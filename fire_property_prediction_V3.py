"""
fire_property_prediction_V3.py: fire-property prediction with poly4mer v3
=========================================================================

Encodes one or more pSMILES to the 768-d latent z and predicts four cone-calorimeter
properties at a given sample thickness and incident heat flux:

    tig    time to ignition             s
    pHRR   peak heat release rate       kW/m^2
    SEA    specific extinction area     m^2/kg
    CO     carbon monoxide yield        kg/kg

Everything is loaded from one file, `checkpoint/poly4mer_v3.ckpt`: the star encoder, the
autoencoder encoder, the four property heads and their standardization statistics. The
decoder is not needed for prediction and is not loaded.

Latent convention
-----------------
The v3 heads consume the RAW latent z. Do not standardize z before passing it in. Only
thickness and flux are standardized, using the statistics stored with the heads. This
differs from the v2 predictor, whose heads were built on a whitened latent and therefore
needed `checkpoint/optimal_model/scalers.pt`.

    tig    MLP([z, thk_norm, flux_norm])   input dim 770
    pHRR   MLP([z, thk_norm, flux_norm])   input dim 770
    SEA    MLP(z)                          input dim 768
    CO     MLP(z)                          input dim 768

    real value = output * y_std[k] + y_mean[k],  k = tig, pkhrr, sea, co

Note that v2 gave SEA 770 inputs; v3 gives it 768. The head shapes in the checkpoint are
authoritative and are checked on load.

INPUT FORM. pSMILES must be RDKit-canonical with bare '*' attachment points, for example
`*CC(*)c1ccccc1`. Bracketed stars and non-canonical orderings of the same molecule do not
encode faithfully. Pass --canonicalize to convert inputs first; the script reports every
string it had to rewrite.

Selecting what to predict
-------------------------
    --smiles  '*CC(*)C'                     one or more pSMILES, comma-separated
    --init_file F --smiles_idx N            row N (1-based, header excluded) of a csv
                                            with a 'smiles' column
    --init_file F --all                     every row of that file

Examples (run from inside this folder):
    python fire_property_prediction_V3.py --smiles '*CC(*)C' --thickness 3 --flux 50
    python fire_property_prediction_V3.py --init_file example_pSMILES_homo.txt --all \
        --thickness 6 --flux 75 --csv predictions.csv
"""

# Deep learning
import torch
import torch.nn as nn

# Data
import pandas as pd

# Standard library
import argparse
import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # repository root
for _p in (SCRIPT_DIR, os.path.join(SCRIPT_DIR, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import star_encoder, AutoEncoderLayer3
from utils1 import load_smi_ted_explicit, find_smi_ted_dir
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------- constants --
NAMES = ["tig", "pkhrr", "sea", "co"]
UNITS = {"tig": "s", "pkhrr": "kW/m^2", "sea": "m^2/kg", "co": "kg/kg"}
PRETTY = {"tig": "Tig", "pkhrr": "pHRR", "sea": "SEA", "co": "CO"}
DECIMALS = {"tig": 4, "pkhrr": 4, "sea": 6, "co": 6}

# which heads also take thickness and flux (input dim 770 instead of 768)
USES_THK_FLUX = {"tig": True, "pkhrr": True, "sea": False, "co": False}

MAX_LEN, EMB_DIM, STAR_TOKEN_ID = 202, 768, 439


def build_regressor(input_dim, device):
    return nn.Sequential(
        nn.Linear(input_dim, 512), nn.GELU(),
        nn.Linear(512, 512), nn.GELU(),
        nn.Linear(512, 256), nn.GELU(),
        nn.Linear(256, 128), nn.GELU(),
        nn.Linear(128, 1)
    ).to(device)



def resolve_heads(ck, which, label="heads"):
    """Pick a head set from a checkpoint that may carry more than one.

    'selected' prefers property_regressors_selected and falls back to property_regressors
    for older checkpoints that only have the one set.
    """
    if which == "selected" and "property_regressors_selected" in ck:
        print(f">> {label}: property_regressors_selected (cross-validation selected)")
        return ck["property_regressors_selected"]
    if which == "selected":
        print(f">> {label}: property_regressors "
              f"(this checkpoint carries no selected set)")
    else:
        print(f">> {label}: property_regressors (the model's own heads)")
    return ck["property_regressors"]


def canonical_psmiles(s):
    """RDKit-canonical pSMILES with bare '*'; None if the string does not parse."""
    m = Chem.MolFromSmiles(str(s).strip())
    return Chem.MolToSmiles(m) if m is not None else None


def get_parser():
    p = argparse.ArgumentParser(description="poly4mer v3 fire-property prediction")

    # what to predict
    p.add_argument("--smiles", default=None,
                   help="One or more pSMILES, comma-separated. Overrides --init_file.")
    p.add_argument("--init_file", default=os.path.join(SCRIPT_DIR, "example_pSMILES_homo.txt"),
                   help="CSV with a 'smiles' column, used when --smiles is absent.")
    p.add_argument("--smiles_idx", type=int, default=1,
                   help="1-based row index into --init_file (header excluded).")
    p.add_argument("--all", action="store_true", help="Predict for every row of --init_file.")
    p.add_argument("--canonicalize", action="store_true",
                   help="Rewrite inputs into RDKit-canonical bare-star form before encoding.")

    # fire test conditions
    p.add_argument("--thickness", type=float, default=3.0, help="Sample thickness, mm.")
    p.add_argument("--flux", type=float, default=50.0, help="Incident heat flux, kW/m^2.")

    # weights
    p.add_argument("--checkpoint", default=os.path.join(SCRIPT_DIR, "checkpoint", "poly4mer_v3.ckpt"))
    p.add_argument("--heads", default="selected", choices=["selected", "joint"],
                   help="Which head set in --checkpoint to use. 'selected' is the cross-validation-selected set and the default; 'joint' is the model's own set, produced with the rest of the model. Ignored when --heads_ckpt names an external file.")
    p.add_argument("--heads_ckpt", default=None,
                   help="Optional separate file holding property_regressors and "
                        "property_stats; by default the heads in --checkpoint are used.")
    p.add_argument("--smi_ted_dir", default=None, help="Defaults to smi_ted_light/ beside this file.")

    # output
    p.add_argument("--csv", default=None, help="Also write the predictions to this CSV.")
    p.add_argument("--batch_size", type=int, default=32)
    return p


def main(config):
    # ------------------------------------------------------------- 1. Device --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Device being used: {device}")

    # ---------------------------------------------- 2. Encoder and the heads --
    print(f">> Loading {os.path.basename(config.checkpoint)}")
    ck = torch.load(config.checkpoint, map_location="cpu", weights_only=False, mmap=True)

    A_encoder = star_encoder(dim=1)
    autoencoder = AutoEncoderLayer3(MAX_LEN * EMB_DIM, EMB_DIM * 4, EMB_DIM * 2, EMB_DIM)
    # The star encoder must be loaded with the rest. An unloaded star_encoder leaves every
    # attachment-point embedding at its random initialization.
    A_encoder.load_state_dict(ck["star_encoder"])
    autoencoder.encoder.load_state_dict(ck["autoencoder_encoder"])
    A_encoder.to(device)
    autoencoder.to(device)

    hk = ck if config.heads_ckpt is None else torch.load(
        config.heads_ckpt, map_location="cpu", weights_only=False)
    if config.heads_ckpt is not None:
        print(f">> Property heads from {os.path.basename(config.heads_ckpt)}")

    print(">> Loading property predictors.")
    src = (hk["property_regressors"] if config.heads_ckpt is not None
           else resolve_heads(ck, config.heads))
    regressors = {}
    for n in NAMES:
        sd = src[n]
        in_dim = sd[next(k for k in sd if k.endswith("weight"))].shape[1]
        expected = 770 if USES_THK_FLUX[n] else 768
        if in_dim != expected:
            raise SystemExit(f"{n} head expects {in_dim} inputs, this script builds {expected}. "
                             f"The checkpoint and the script disagree.")
        r = build_regressor(in_dim, device)
        r.load_state_dict(sd)
        regressors[n] = r

    stats = {k: torch.as_tensor(v, dtype=torch.float32, device=device)
             for k, v in hk["property_stats"].items()}
    y_mean, y_std = stats["y_mean"].view(-1), stats["y_std"].view(-1)

    # Frozen smi-ted-Light.
    smi_ted_dir = config.smi_ted_dir or find_smi_ted_dir()
    Smi_ted, MolTranBertTokenizer = load_smi_ted_explicit()
    Smi_ted_token = Smi_ted(MolTranBertTokenizer(os.path.join(smi_ted_dir, "bert_vocab_curated.txt")))
    Smi_ted_token.load_checkpoint(os.path.join(smi_ted_dir, "smi-ted-Light_40.pt"))

    for m in [A_encoder, autoencoder.encoder, Smi_ted_token, *regressors.values()]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    # ---------------------------------------------------- 3. Test conditions --
    thk_norm = ((torch.tensor(config.thickness, device=device, dtype=torch.float32)
                 - stats["thk_mean"]) / stats["thk_std"]).view(1, 1)
    flux_norm = ((torch.tensor(config.flux, device=device, dtype=torch.float32)
                  - stats["flux_mean"]) / stats["flux_std"]).view(1, 1)

    # ------------------------------------------------------- 4. Input pSMILES --
    if config.smiles is not None:
        smiles = [s.strip() for s in config.smiles.split(",") if s.strip()]
        names = [None] * len(smiles)
    else:
        df = pd.read_csv(config.init_file, delimiter=",")
        if "smiles" not in df.columns:
            raise SystemExit(f"{config.init_file} has no 'smiles' column")
        nm = df["mol_name"].astype(str).tolist() if "mol_name" in df.columns else [None] * len(df)
        if config.all:
            smiles, names = df["smiles"].astype(str).tolist(), nm
        else:
            i = config.smiles_idx - 1
            if not 0 <= i < len(df):
                raise SystemExit(f"--smiles_idx {config.smiles_idx} out of range (1..{len(df)})")
            smiles, names = [str(df["smiles"][i])], [nm[i]]

    canon = [canonical_psmiles(s) for s in smiles]
    bad = [s for s, c in zip(smiles, canon) if c is None]
    for s in bad:
        print(f"!! does not parse, skipped: {s}")
    keep = [(n, s, c) for n, s, c in zip(names, smiles, canon) if c is not None]
    if not keep:
        raise SystemExit("no parsable pSMILES to predict")

    if config.canonicalize:
        changed = sum(1 for _, s, c in keep if s != c)
        if changed:
            print(f">> Canonicalized {changed} of {len(keep)} inputs.")
        inputs = [c for _, _, c in keep]
    else:
        noncanon = [s for _, s, c in keep if s != c]
        if noncanon:
            print(f"!! {len(noncanon)} of {len(keep)} inputs are not in canonical bare-star form. "
                  f"Predictions for these may be unreliable; pass --canonicalize to rewrite them.")
        inputs = [s for _, s, _ in keep]

    print(f">> Conditions: thickness={config.thickness} mm, flux={config.flux} kW/m^2")
    print(f">> Predicting for {len(inputs)} pSMILES.\n")

    # ------------------------------------------------------------ 5. Predict --
    rows = []
    with torch.no_grad():
        for i in range(0, len(inputs), config.batch_size):
            block = inputs[i:i + config.batch_size]
            idx_t, token_emb_t, _ = Smi_ted_token.extract_embeddings(block)
            idx_t, token_emb_t = idx_t.to(device), token_emb_t.to(device)
            pos = torch.nonzero(idx_t == STAR_TOKEN_ID, as_tuple=False)
            if pos.numel():
                token_emb_t[pos[:, 0], pos[:, 1], :] = \
                    A_encoder(idx_t[pos[:, 0], pos[:, 1]].view(-1, 1).float())
            z = autoencoder.encoder(token_emb_t.view(-1, MAX_LEN * EMB_DIM))

            B = z.shape[0]
            in770 = torch.cat([z, thk_norm.expand(B, 1), flux_norm.expand(B, 1)], dim=-1)
            pred = {n: (regressors[n](in770 if USES_THK_FLUX[n] else z).view(-1)
                        * y_std[k] + y_mean[k]).cpu().tolist()
                    for k, n in enumerate(NAMES)}
            for j in range(B):
                rows.append({n: pred[n][j] for n in NAMES})

    # -------------------------------------------------------------- 6. Report --
    for (nm, raw, c), p in zip(keep, rows):
        label = f" [{nm}]" if nm and nm != "nan" else ""
        print(f">> pSMILES{label}: {raw}")
        if raw != c:
            print(f"   canonical: {c}")
        for n in NAMES:
            print(f"   Predicted {PRETTY[n]:<4} {p[n]:>12.{DECIMALS[n]}f}  {UNITS[n]}")
        print()

    if config.csv:
        with open(config.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mol_name", "smiles", "smiles_canonical", "thickness", "flux",
                        "tig", "pkhrr", "sea", "co"])
            for (nm, raw, c), p in zip(keep, rows):
                w.writerow([nm if nm else "", raw, c, config.thickness, config.flux]
                           + [f"{p[n]:.6f}" for n in NAMES])
        print(f">> wrote {config.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main(get_parser().parse_args())
