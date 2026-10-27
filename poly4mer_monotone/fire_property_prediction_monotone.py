"""
fire_property_prediction_monotone.py: monotone-spline fire-property prediction
===============================================================================

Predicts time to ignition (Tig) and peak heat release rate (pHRR) with heads that are
**hard-monotone in heat flux**: Tig can only decrease as flux rises, pHRR can only
increase. The constraint is structural, not a penalty, so it holds for every input.

The encoder is unchanged from the main model. Only the property heads differ, so this
script reads `../checkpoint/poly4mer_v3.ckpt` for the encoder and a monotone head file
from `checkpoint/` for the regressor. SEA and CO have no monotone heads; use
`../fire_property_prediction_V3.py` for those.

How the monotone head works
---------------------------
A plain head maps `[z, thickness_norm, flux_norm]` straight to one number, and nothing
stops it predicting a longer ignition time at a higher flux. The monotone head instead:

1. evaluates the same MLP at 17 fixed flux knots, 20 to 100 kW/m^2 in steps of 5,
   holding the latent and the thickness at their actual values;
2. runs a cumulative minimum over those 17 values for Tig, or a cumulative maximum for
   pHRR, which forces the knot sequence to be monotone in flux;
3. linearly interpolates between the two knots bracketing the requested flux;
4. de-standardizes to physical units.

Requested fluxes are clamped into the knot range, so a query at 10 or 150 kW/m^2
returns the value at 20 or 100.

One head variant ships per property, `last_two_linear`, in which only the last two
linear layers were fine-tuned. It is per-fold: 10 checkpoints, from a 10-fold split
of the experimental rows.
Pick one with --fold, or pass --ensemble to average the predictions of all folds
present, which is the more honest choice for a polymer outside the dataset.

INPUT FORM. pSMILES must be RDKit-canonical with bare '*' attachment points. Pass
--canonicalize to rewrite inputs first.

Examples (run from inside this folder):
    python fire_property_prediction_monotone.py --smiles '*CC(*)C' --thickness 3 --flux 50
    python fire_property_prediction_monotone.py --init_file ../example_pSMILES_homo.txt \
        --all --ensemble --csv monotone_predictions.csv
    python fire_property_prediction_monotone.py --smiles '*CC(*)C' --thickness 3 \
        --flux_sweep 20,100,5 --csv sweep.csv
"""

# Deep learning
import torch
import torch.nn as nn

# Data
import pandas as pd

# Standard library
import argparse
import csv
import glob
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # poly4mer_monotone/
REPO_DIR = os.path.dirname(SCRIPT_DIR)                         # repository root
for _p in (REPO_DIR, os.path.join(REPO_DIR, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import star_encoder, AutoEncoderLayer3
from utils1 import load_smi_ted_explicit, find_smi_ted_dir
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------- constants --
PROPS = ["tig", "pkhrr"]
PRETTY = {"tig": "Tig", "pkhrr": "pHRR"}
UNITS = {"tig": "s", "pkhrr": "kW/m^2"}
CKPT_DIR = {"tig": "tig", "pkhrr": "phrr"}          # folder names on disk

# flux knots the monotone constraint is applied on, kW/m^2
KNOTS = [float(v) for v in range(20, 101, 5)]

# direction of the hard constraint in flux
DIRECTION = {"tig": "cummin", "pkhrr": "cummax"}

MAX_LEN, EMB_DIM, STAR_TOKEN_ID = 202, 768, 439


def build_mlp(device):
    """The knot generator. Same shape as the plain property head: 770 inputs."""
    return nn.Sequential(
        nn.Linear(770, 512), nn.GELU(),
        nn.Linear(512, 512), nn.GELU(),
        nn.Linear(512, 256), nn.GELU(),
        nn.Linear(256, 128), nn.GELU(),
        nn.Linear(128, 1)
    ).to(device)


def canonical_psmiles(s):
    """RDKit-canonical pSMILES with bare '*'; None if the string does not parse."""
    m = Chem.MolFromSmiles(str(s).strip())
    return Chem.MolToSmiles(m) if m is not None else None


def monotone_forward(mlp, z, thk_norm, flux_raw, prop, stats):
    """Monotone prediction in physical units.

    z         [B, 768] raw latent
    thk_norm  [B, 1]   standardized thickness
    flux_raw  [B]      heat flux in kW/m^2, unstandardized
    """
    dev = z.device
    knots = torch.tensor(KNOTS, dtype=torch.float32, device=dev)
    kn = (knots - stats["flux_mean"]) / stats["flux_std"]
    B, K = z.shape[0], knots.numel()

    # evaluate the MLP at every knot, holding latent and thickness fixed
    ctx = torch.cat([z, thk_norm], dim=-1)                       # [B, 769]
    ctx = ctx[:, None, :].expand(-1, K, -1)                      # [B, K, 769]
    feats = torch.cat([ctx, kn[None, :, None].expand(B, -1, -1)], 2).reshape(B * K, 770)
    q = mlp(feats).reshape(B, K)

    # hard monotonicity along the knot axis
    q = torch.cummin(q, 1).values if DIRECTION[prop] == "cummin" else torch.cummax(q, 1).values

    # linear interpolation at the requested flux, clamped into the knot range
    f = flux_raw.clamp(float(knots[0]), float(knots[-1]))
    right = torch.searchsorted(knots, f, right=True).clamp(1, K - 1)
    left = right - 1
    row = torch.arange(B, device=dev)
    w = (f - knots[left]) / (knots[right] - knots[left])
    out = q[row, left] + w * (q[row, right] - q[row, left])

    return out * stats["y_std"][prop] + stats["y_mean"][prop]


def get_parser():
    p = argparse.ArgumentParser(description="poly4mer monotone fire-property prediction")

    p.add_argument("--smiles", default=None,
                   help="One or more pSMILES, comma-separated. Overrides --init_file.")
    p.add_argument("--init_file", default=os.path.join(REPO_DIR, "example_pSMILES_homo.txt"),
                   help="CSV with a 'smiles' column, used when --smiles is absent.")
    p.add_argument("--smiles_idx", type=int, default=1, help="1-based row index into --init_file.")
    p.add_argument("--all", action="store_true", help="Predict for every row of --init_file.")
    p.add_argument("--canonicalize", action="store_true",
                   help="Rewrite inputs into RDKit-canonical bare-star form before encoding.")

    p.add_argument("--properties", default="tig,pkhrr",
                   help="Which monotone properties to predict. SEA and CO have no monotone heads.")
    p.add_argument("--variant", default="last_two_linear",
                   help="Head variant sub-folder under --heads_dir/<property>/.")
    p.add_argument("--fold", type=int, default=0, help="Which fold checkpoint to use (0-9).")
    p.add_argument("--ensemble", action="store_true",
                   help="Average over every fold checkpoint present instead of using one.")

    p.add_argument("--thickness", type=float, default=3.0, help="Sample thickness, mm.")
    p.add_argument("--flux", type=float, default=50.0, help="Incident heat flux, kW/m^2.")
    p.add_argument("--flux_sweep", default=None,
                   help="START,STOP,STEP in kW/m^2; predict across the range instead of one flux. "
                        "Useful for checking the monotone behaviour.")

    p.add_argument("--encoder_checkpoint",
                   default=os.path.join(REPO_DIR, "checkpoint", "poly4mer_v3.ckpt"),
                   help="Supplies the encoder and the standardization statistics.")
    p.add_argument("--heads_dir", default=os.path.join(SCRIPT_DIR, "checkpoint"))
    p.add_argument("--smi_ted_dir", default=None)

    p.add_argument("--csv", default=None, help="Also write the predictions to this CSV.")
    p.add_argument("--batch_size", type=int, default=32)
    return p


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Device being used: {device}")
    props = [p.strip() for p in config.properties.split(",") if p.strip()]
    for p in props:
        if p not in PROPS:
            raise SystemExit(f"'{p}' has no monotone head. Available: {', '.join(PROPS)}. "
                             f"For SEA and CO use ../fire_property_prediction_V3.py.")

    # ------------------------------------------- 1. Encoder and statistics --
    print(f">> Loading encoder from {os.path.relpath(config.encoder_checkpoint, SCRIPT_DIR)}")
    ck = torch.load(config.encoder_checkpoint, map_location="cpu", weights_only=False, mmap=True)

    A_encoder = star_encoder(dim=1)
    autoencoder = AutoEncoderLayer3(MAX_LEN * EMB_DIM, EMB_DIM * 4, EMB_DIM * 2, EMB_DIM)
    A_encoder.load_state_dict(ck["star_encoder"])
    autoencoder.encoder.load_state_dict(ck["autoencoder_encoder"])
    A_encoder.to(device)
    autoencoder.to(device)

    st = {k: torch.as_tensor(v, dtype=torch.float32, device=device)
          for k, v in ck["property_stats"].items()}
    order = ["tig", "pkhrr", "sea", "co"]
    stats = {"flux_mean": st["flux_mean"].reshape(()), "flux_std": st["flux_std"].reshape(()),
             "y_mean": {p: st["y_mean"].view(-1)[order.index(p)] for p in PROPS},
             "y_std": {p: st["y_std"].view(-1)[order.index(p)] for p in PROPS}}
    thk_mean, thk_std = st["thk_mean"].reshape(()), st["thk_std"].reshape(())

    smi_ted_dir = config.smi_ted_dir or find_smi_ted_dir()
    Smi_ted, MolTranBertTokenizer = load_smi_ted_explicit()
    Smi_ted_token = Smi_ted(MolTranBertTokenizer(os.path.join(smi_ted_dir, "bert_vocab_curated.txt")))
    Smi_ted_token.load_checkpoint(os.path.join(smi_ted_dir, "smi-ted-Light_40.pt"))

    for m in [A_encoder, autoencoder.encoder, Smi_ted_token]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    # ----------------------------------------------- 2. Monotone head files --
    print(f">> Loading monotone heads, variant '{config.variant}'.")
    heads = {}
    for prop in props:
        d = os.path.join(config.heads_dir, CKPT_DIR[prop], config.variant)
        files = sorted(glob.glob(os.path.join(d, "fold_*_best_fp32.ckpt")))
        if not files:
            raise SystemExit(
                f"no checkpoints in {d}\n"
                f"Download the monotone heads and place them there; see this folder's README.")
        if not config.ensemble:
            want = os.path.join(d, f"fold_{config.fold}_best_fp32.ckpt")
            if want not in files:
                raise SystemExit(f"fold {config.fold} not found in {d}; present: "
                                 + ", ".join(os.path.basename(f) for f in files))
            files = [want]
        mlps = []
        for f in files:
            c = torch.load(f, map_location="cpu", weights_only=False)
            m = build_mlp(device)
            m.load_state_dict(c["model_state_dict"] if "model_state_dict" in c else c)
            m.eval()
            for q in m.parameters():
                q.requires_grad = False
            mlps.append(m)
        heads[prop] = mlps
        how = f"{len(mlps)} folds averaged" if config.ensemble else f"fold {config.fold}"
        print(f"   {PRETTY[prop]:<4} {how}, monotone by {DIRECTION[prop]} over "
              f"{len(KNOTS)} flux knots {KNOTS[0]:.0f}-{KNOTS[-1]:.0f} kW/m^2")

    # ------------------------------------------------------- 3. Input pSMILES --
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
    for s, c in zip(smiles, canon):
        if c is None:
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
        noncanon = sum(1 for _, s, c in keep if s != c)
        if noncanon:
            print(f"!! {noncanon} of {len(keep)} inputs are not in canonical bare-star form. "
                  f"Pass --canonicalize to rewrite them.")
        inputs = [s for _, s, _ in keep]

    if config.flux_sweep:
        a, b, s = (float(v) for v in config.flux_sweep.split(","))
        fluxes = [a + i * s for i in range(int(round((b - a) / s)) + 1)]
    else:
        fluxes = [config.flux]
    print(f">> Conditions: thickness={config.thickness} mm, "
          f"flux={'%g-%g step %g' % (fluxes[0], fluxes[-1], fluxes[1]-fluxes[0]) if len(fluxes) > 1 else fluxes[0]} kW/m^2")
    print(f">> Predicting for {len(inputs)} pSMILES at {len(fluxes)} flux values.\n")

    # ------------------------------------------------------------ 4. Predict --
    thk_norm_scalar = (torch.tensor(config.thickness, device=device) - thk_mean) / thk_std
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
            thk_norm = thk_norm_scalar.view(1, 1).expand(B, 1)
            for fx in fluxes:
                fr = torch.full((B,), float(fx), device=device)
                pred = {}
                for prop in props:
                    vals = torch.stack([monotone_forward(m, z, thk_norm, fr, prop, stats)
                                        for m in heads[prop]]).mean(0)
                    pred[prop] = vals.cpu().tolist()
                for j in range(B):
                    rows.append({"row": i + j, "flux": fx, **{p: pred[p][j] for p in props}})

    # -------------------------------------------------------------- 5. Report --
    for k, (nm, raw, c) in enumerate(keep):
        label = f" [{nm}]" if nm and nm != "nan" else ""
        print(f">> pSMILES{label}: {raw}")
        if raw != c:
            print(f"   canonical: {c}")
        mine = [r for r in rows if r["row"] == k]
        if len(fluxes) == 1:
            for prop in props:
                print(f"   Predicted {PRETTY[prop]:<4} {mine[0][prop]:>12.4f}  {UNITS[prop]}")
        else:
            header = "   " + "flux".rjust(7) + "".join(f"{PRETTY[p]:>14}" for p in props)
            print(header)
            for r in sorted(mine, key=lambda r: r["flux"]):
                print("   " + f"{r['flux']:>7.0f}" + "".join(f"{r[p]:>14.4f}" for p in props))
        print()

    if config.csv:
        with open(config.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mol_name", "smiles", "smiles_canonical", "thickness", "flux",
                        "variant", "fold"] + props)
            fold = "ensemble" if config.ensemble else config.fold
            for k, (nm, raw, c) in enumerate(keep):
                for r in sorted((r for r in rows if r["row"] == k), key=lambda r: r["flux"]):
                    w.writerow([nm if nm else "", raw, c, config.thickness, r["flux"],
                                config.variant, fold] + [f"{r[p]:.6f}" for p in props])
        print(f">> wrote {config.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main(get_parser().parse_args())
