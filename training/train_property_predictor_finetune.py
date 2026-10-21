"""
train_property_predictor_finetune.py: Phase I fine-tuning of the property predictor
===================================================================================

Lives in training/; models.py, utils1.py, smi_ted_light/ and poly4mer_v3.ckpt are in the
repository root one level up. Run it from the repository root:
    python training/train_property_predictor_finetune.py ...

Freezes the whole encoder module and trains ONLY the four property heads
(tig, pHRR, SEA, CO) on labeled data, for example synthetic (simulated) data.

    pSMILES --> smi-ted (frozen) --> star_encoder (frozen) --> encoder (frozen) --> z [768]
    z (+ thickness, flux for tig / pHRR) --> property heads (TRAINED) --> standardized target

Because nothing upstream of z changes, every polymer is embedded exactly once.
The script embeds all unique pSMILES of the training and validation files,
stores the latents in a cache file, and then trains the heads on the cached
latents. Re-running with the same cache file skips the embedding step.

What is trained
---------------
Only `property_regressors` (one 5-layer MLP per property, see build_regressor()
in train_poly4mer_v3_property_reg.py). The decoder module is not loaded at all.

Loss
----
    loss = sum over selected properties of MSE(head(x), (y - y_mean) / y_std)

Targets are standardized. Missing labels (NaN) are masked per property, so a
row may carry only some of the four properties.

Reproduction mode (--cv)
------------------------
With --cv the script instead reproduces the 10-fold experimental cross-validation that
produced the heads bundled in poly4mer_v3.ckpt, reading data/10folds_data/<property>/.
That protocol differs from phase-I fine-tuning in four ways, so it is a separate path:

  * the loss is the mean RELATIVE error in physical units, not MSE on standardized targets
  * a grid of learning rates is swept per fold and the best one kept
  * the epoch is chosen by the lowest error on that fold's own test rows
  * every head starts from the model's own head in the checkpoint, `property_regressors`,
    not from `property_regressors_selected`; the latter were chosen from these same folds

Points three and four matter when reporting. Selecting the epoch on the test fold means
the per-fold test error is optimistically biased, and the "best overall" criterion that
chose the shipped heads scores each head on the whole property set, roughly 90% of which
it trained on. Both numbers are reproduced here so the selection is auditable, not
because either estimates generalization.

Standardization statistics
--------------------------
The heads only make sense together with the statistics they were trained with.
    * default:            heads AND statistics are taken from the checkpoint
                          (continue training the existing heads)
    * --fresh_heads:      heads start from random weights and the statistics are
                          recomputed from the training file
    * --recompute_stats:  keep the checkpoint heads but recompute the statistics
                          from the training file (only sensible if the new data
                          has a very different range)

Outputs (in --out_dir)
----------------------
    latent_cache.pt        cached latents of every embedded pSMILES (reusable)
    heads_best.ckpt        property_regressors + property_stats + meta (~13 MB)
    history.csv            per-epoch train loss, val loss, val relative errors
    poly4mer_v3_phase1.ckpt  (only with --save_full) a drop-in replacement for
                             poly4mer_v3.ckpt with the new heads

Model selection: if --val_labeled is given, the epoch with the lowest validation
loss is kept; otherwise the last epoch is kept. --patience N stops training when
the validation loss has not improved for N epochs.

Example
-------
    python training/train_property_predictor_finetune.py \\
        --train_labeled my_data/synthetic_train.csv \\
        --val_labeled   my_data/synthetic_val.csv \\
        --out_dir       runs/phase1 \\
        --epochs 50 --lr 1e-4 --batch_size 256

Evaluate heads without training (any checkpoint that carries heads):
    python training/train_property_predictor_finetune.py --eval_only --checkpoint runs/phase1/heads_best.ckpt \\
        --val_labeled my_data/synthetic_val.csv --out_dir runs/phase1_eval

Data format: identical to train_poly4mer_v3_property_reg.py (--train_labeled / --val_labeled),
see README.md. Units: thickness mm, flux kW/m^2, tig s, pHRR kW/m^2, CO kg/kg.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # .../training
REPO_DIR = os.path.dirname(SCRIPT_DIR)                           # repository root: models.py, utils1.py, smi_ted_light/
for _p in (REPO_DIR, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import star_encoder, AutoEncoderLayer3
from utils1 import load_smi_ted_explicit, find_smi_ted_dir
from train_poly4mer_v3_property_reg import (MAX_LEN, EMB_DIM, STAR_TOKEN_ID, PROPERTY_NAMES, PROP_USES_THK_FLUX,
                               parse_paths, load_labeled, compute_property_stats, build_regressor,
                               count_params, lr_at_epoch, set_lr)


# =============================================================================
# Frozen encoder module and latent cache
# =============================================================================

def load_frozen_encoder(ckpt, device):
    """smi-ted + star_encoder + autoencoder.encoder, all frozen, from a poly4mer_v3-style checkpoint."""
    SmiTed, Tokenizer = load_smi_ted_explicit()
    d = find_smi_ted_dir()
    smi_ted = SmiTed(Tokenizer(os.path.join(d, "bert_vocab_curated.txt")))
    smi_ted.load_checkpoint(os.path.join(d, "smi-ted-Light_40.pt"))
    smi_ted.eval()
    A = star_encoder(dim=1)
    A.load_state_dict(ckpt["star_encoder"])
    ae = AutoEncoderLayer3(feature_size=MAX_LEN * EMB_DIM, mid_size=EMB_DIM * 4, mid_size_2=EMB_DIM * 2, latent_size=EMB_DIM)
    ae.encoder.load_state_dict(ckpt["autoencoder_encoder"])
    enc = ae.encoder
    for m in (smi_ted, A, enc):
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad = False
    return smi_ted, A, enc


@torch.no_grad()
def encode(smiles, smi_ted, A, enc, device):
    """List of pSMILES -> latent z [B, 768]. Mirrors forward_pass() in train_poly4mer_v3_property_reg.py."""
    idx, emb, _ = smi_ted.extract_embeddings(smiles)
    idx = idx.to(device)
    emb = emb.to(device).clone()
    pos = torch.nonzero(idx == STAR_TOKEN_ID, as_tuple=False)
    if pos.numel() > 0:
        emb[pos[:, 0], pos[:, 1], :] = A(idx[pos[:, 0], pos[:, 1]].view(-1, 1).float())
    return enc(emb.view(-1, MAX_LEN * EMB_DIM))


def embed_unique(smiles, cache_path, ckpt, device, batch_size):
    """Return {pSMILES: z (cpu float32)} for every string in `smiles`, using / extending the cache file."""
    cache = {}
    if cache_path and os.path.isfile(cache_path):
        c = torch.load(cache_path, map_location="cpu", weights_only=False)
        cache = {s: z for s, z in zip(c["smiles"], c["latents"])}
        print(f">> latent cache: {len(cache)} polymers loaded from {cache_path}")
    todo = [s for s in dict.fromkeys(smiles) if s not in cache]
    if todo:
        print(f">> embedding {len(todo)} new polymers with the frozen encoder (batch {batch_size}) ...", flush=True)
        smi_ted, A, enc = load_frozen_encoder(ckpt, device)
        t0 = time.time()
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            z = encode(batch, smi_ted, A, enc, device).cpu()
            for s, zz in zip(batch, z):
                cache[s] = zz
            if (i // batch_size) % 100 == 0:
                print(f"   {min(i + batch_size, len(todo))}/{len(todo)}  [{time.time() - t0:.0f}s]", flush=True)
        del smi_ted, A, enc
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if cache_path:
            keys = list(cache)
            torch.save({"smiles": keys, "latents": torch.stack([cache[k] for k in keys])}, cache_path)
            print(f">> latent cache saved: {len(keys)} polymers -> {cache_path}")
    return cache


# =============================================================================
# Data as tensors (everything fits in memory once latents are cached)
# =============================================================================

def frame_to_tensors(df, cache, device):
    z = torch.stack([cache[s] for s in df["smiles_canonicalized"].astype(str)]).to(device)
    thk = torch.tensor(df["thickness"].values, dtype=torch.float32, device=device)
    flux = torch.tensor(df["flux"].values, dtype=torch.float32, device=device)
    y = torch.tensor(df[PROPERTY_NAMES].values, dtype=torch.float32, device=device)   # NaN allowed
    return z, thk, flux, y


def head_inputs(z, thk, flux, stats):
    thk_n = (thk - stats["thk_mean"]) / stats["thk_std"]
    flux_n = (flux - stats["flux_mean"]) / stats["flux_std"]
    return torch.cat([z, thk_n.view(-1, 1), flux_n.view(-1, 1)], dim=-1), z


def masked_loss(z, thk, flux, y, heads, stats, props):
    """Sum over `props` of standardized MSE, NaN labels masked per property. Also returns per-property terms."""
    x770, x768 = head_inputs(z, thk, flux, stats)
    total = z.new_zeros(())
    parts = {}
    for name in props:
        k = PROPERTY_NAMES.index(name)
        pred = heads[name](x770 if PROP_USES_THK_FLUX[name] else x768).view(-1)
        target = (y[:, k] - stats["y_mean"][k]) / stats["y_std"][k]
        mask = ~torch.isnan(target)
        if mask.any():
            mse = F.mse_loss(pred[mask], target[mask])
            total = total + mse
            parts[name] = float(mse)
    return total, parts


@torch.no_grad()
def evaluate(z, thk, flux, y, heads, stats, props, batch=4096):
    """Validation loss plus per-property mean and median relative error in physical units."""
    for h in heads.values():
        h.eval()
    loss_sum, n_rows = 0.0, 0
    rel = {p: [] for p in props}
    for i in range(0, len(z), batch):
        sl = slice(i, min(i + batch, len(z)))
        bs = sl.stop - sl.start
        loss, _ = masked_loss(z[sl], thk[sl], flux[sl], y[sl], heads, stats, props)
        loss_sum += float(loss) * bs
        n_rows += bs
        x770, x768 = head_inputs(z[sl], thk[sl], flux[sl], stats)
        for name in props:
            k = PROPERTY_NAMES.index(name)
            pred = heads[name](x770 if PROP_USES_THK_FLUX[name] else x768).view(-1) * stats["y_std"][k] + stats["y_mean"][k]
            yy = y[sl, k]
            m = (~torch.isnan(yy)) & (yy.abs() > 1e-12)
            if m.any():
                rel[name].append(((pred[m] - yy[m]).abs() / yy[m].abs()).cpu())
    out = {"val_loss": loss_sum / max(n_rows, 1)}
    for name in props:
        r = torch.cat(rel[name]) if rel[name] else torch.tensor([float("nan")])
        out[f"{name}_rel_mean"] = float(r.mean())
        out[f"{name}_rel_median"] = float(r.median())
        out[f"{name}_n"] = int(r.numel()) if rel[name] else 0
    return out


def fmt_metrics(m, props):
    return "  ".join(f"{p}: mean {m[f'{p}_rel_mean'] * 100:6.2f}% median {m[f'{p}_rel_median'] * 100:6.2f}% (n={m[f'{p}_n']})" for p in props)


# =============================================================================
# Main
# =============================================================================

# ---------------------------------------------------------------------------
# Reproduction of the 10-fold experimental cross-validation (--cv)
# ---------------------------------------------------------------------------

# Per-property recipe of the runs that selected the heads shipped in the checkpoint.
# grid order matters: the sweep keeps the first learning rate that attains the best
# test error, so a different order can pick a different winner on ties.
CV_RECIPE = {
    "tig":   dict(lr_grid=[1e-4, 5e-5, 1e-5, 5e-6, 1e-6], epochs=5000, batch_size=32),
    "pkhrr": dict(lr_grid=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6], epochs=2000, batch_size=64),
    "sea":   dict(lr_grid=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6], epochs=2000, batch_size=32),
    "co":    dict(lr_grid=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6], epochs=2000, batch_size=32),
}
# column holding the target in each property's fold CSVs
CV_TARGET = {"tig": "tig", "pkhrr": "pkhrr", "sea": "SEA", "co": "CO"}


def _parse_folds(spec):
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _mean_rel(pred, y):
    m = y.abs() > 1e-12
    return (torch.abs(pred[m] - y[m]) / torch.abs(y[m])).mean()


def run_cv(args, props, device):
    """Re-run the 10-fold sweep that produced the per-fold heads."""
    import pandas as pd
    from torch.utils.data import DataLoader, TensorDataset

    ckpt_path = args.encoder_checkpoint or args.checkpoint
    print(f">> encoder and initial heads from {os.path.basename(ckpt_path)}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    smi_ted, A, enc = load_frozen_encoder(ck, device)
    stats = {k: torch.as_tensor(v, dtype=torch.float32, device=device)
             for k, v in ck["property_stats"].items()}
    thk_m, thk_s = stats["thk_mean"].reshape(()), stats["thk_std"].reshape(())
    flx_m, flx_s = stats["flux_mean"].reshape(()), stats["flux_std"].reshape(())
    y_mean, y_std = stats["y_mean"].view(-1), stats["y_std"].view(-1)

    folds = _parse_folds(args.cv_folds)
    summary = {}

    for prop in props:
        k_idx = PROPERTY_NAMES.index(prop)
        rec = dict(CV_RECIPE[prop])
        if args.cv_lr_grid:
            rec["lr_grid"] = [float(x) for x in args.cv_lr_grid.split(",")]
        if args.cv_epochs:
            rec["epochs"] = args.cv_epochs
        if args.cv_batch_size:
            rec["batch_size"] = args.cv_batch_size
        ddir = os.path.join(args.cv_data_dir, prop)
        if not os.path.isdir(ddir):
            print(f"!! no fold data for {prop} at {ddir}, skipped")
            continue
        if args.cv_init_heads == "selected" and "property_regressors_selected" in ck:
            init_src = ck["property_regressors_selected"]
            print("!! starting from the selected heads; those were chosen from these same "
                  "folds, so the result is not an independent reproduction")
        else:
            init_src = ck["property_regressors"]
        init_sd = {k: v.to(device) for k, v in init_src[prop].items()}
        in_dim = 770 if PROP_USES_THK_FLUX[prop] else 768
        tcol = CV_TARGET[prop]
        out_root = os.path.join(args.out_dir, prop)
        os.makedirs(out_root, exist_ok=True)
        print(f"\n=== {prop}: {len(folds)} folds, lrs {rec['lr_grid']}, "
              f"{rec['epochs']} epochs, batch {rec['batch_size']} ===", flush=True)

        def make_xy(df):
            df = df.copy()
            cols = ["thickness", "flux", tcol] if PROP_USES_THK_FLUX[prop] else [tcol]
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["smiles_canonicalized"] + cols).reset_index(drop=True)
            smis = df["smiles_canonicalized"].astype(str).tolist()
            z = torch.cat([encode(smis[i:i + args.embed_batch_size], smi_ted, A, enc, device)
                           for i in range(0, len(smis), args.embed_batch_size)])
            y = torch.tensor(df[tcol].values, dtype=torch.float32, device=device).view(-1, 1)
            if not PROP_USES_THK_FLUX[prop]:
                return z, y
            thk = torch.tensor(df["thickness"].values, dtype=torch.float32, device=device).view(-1, 1)
            flx = torch.tensor(df["flux"].values, dtype=torch.float32, device=device).view(-1, 1)
            return torch.cat([z, (thk - thk_m) / thk_s, (flx - flx_m) / flx_s], dim=-1), y

        rows = []
        for fold in folds:
            tr = pd.read_csv(os.path.join(ddir, f"data_{fold}", "train.csv"))
            te = pd.read_csv(os.path.join(ddir, f"data_{fold}", "test.csv"))
            Xtr, ytr = make_xy(tr)
            Xte, yte = make_xy(te)                      # embedded once, reused across lrs

            best = dict(test_rel=float("inf"), train_rel=float("nan"), lr=None, epoch=-1, state=None)
            for lr in rec["lr_grid"]:
                head = build_regressor(in_dim).to(device)
                head.load_state_dict(init_sd)
                opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=args.wd)
                loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=rec["batch_size"], shuffle=True)

                @torch.no_grad()
                def rel(X, y):
                    head.eval()
                    return _mean_rel(head(X).view(-1, 1) * y_std[k_idx] + y_mean[k_idx], y).item()

                b = dict(te=float("inf"), tr=float("nan"), ep=-1, state=None)
                for ep in range(rec["epochs"]):
                    head.train()
                    for xb, yb in loader:
                        opt.zero_grad()
                        loss = _mean_rel(head(xb).view(-1, 1) * y_std[k_idx] + y_mean[k_idx], yb)
                        loss.backward()
                        opt.step()
                    tr_e, te_e = rel(Xtr, ytr), rel(Xte, yte)
                    if te_e < b["te"]:                  # epoch selected on the test fold
                        b = dict(te=te_e, tr=tr_e, ep=ep,
                                 state={k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
                print(f"   fold {fold}  lr={lr:.0e}: best test_rel={100*b['te']:6.2f}% "
                      f"(ep{b['ep']}, train {100*b['tr']:.2f}%)", flush=True)
                if b["te"] < best["test_rel"]:
                    best = dict(test_rel=b["te"], train_rel=b["tr"], lr=lr, epoch=b["ep"], state=b["state"])

            torch.save(best["state"], os.path.join(out_root, f"fold_{fold}.ckpt"))
            with open(os.path.join(out_root, f"fold_{fold}.txt"), "w") as f:
                f.write(f"best_lr={best['lr']} best_epoch={best['epoch']} "
                        f"train_rel={best['train_rel']:.6f} test_rel={best['test_rel']:.6f}\n")
            rows.append(dict(fold=fold, n_train=len(ytr), n_test=len(yte), lr=best["lr"],
                             epoch=best["epoch"], train_rel=100 * best["train_rel"],
                             test_rel=100 * best["test_rel"]))
            print(f"   -> fold {fold}: lr={best['lr']:.0e} ep={best['epoch']} "
                  f"train={100*best['train_rel']:.2f}% test={100*best['test_rel']:.2f}%", flush=True)

        if rows:
            t = np.array([r["test_rel"] for r in rows])
            top = np.argsort(t)[:5]
            summary[prop] = dict(per_fold=rows, top5_folds=sorted(int(rows[i]["fold"]) for i in top),
                                 top5_mean=float(t[top].mean()), all_mean=float(t.mean()))
            print(f"   best-5 {summary[prop]['top5_folds']}: {summary[prop]['top5_mean']:.2f}%   "
                  f"all-{len(t)} mean {summary[prop]['all_mean']:.2f}%")
            with open(os.path.join(out_root, "summary.json"), "w") as f:
                json.dump(summary[prop], f, indent=1)

    print("\n>> heads written under", args.out_dir)
    print(">> score them with evaluate_fire_prediction_heads.py --heads_dir " + args.out_dir)
    return summary



def main():
    ap = argparse.ArgumentParser(description="Phase I: fine-tune the property heads on labeled data with the encoder frozen.")
    ap.add_argument("--train_labeled", default=None, help="Comma-separated labeled files (required unless --eval_only).")
    ap.add_argument("--val_labeled", default=None, help="Comma-separated labeled files for validation and model selection.")
    ap.add_argument("--checkpoint", default=os.path.join(REPO_DIR, "checkpoint", "poly4mer_v3.ckpt"),
                    help="Checkpoint providing the frozen encoder and (by default) the initial heads and statistics.")
    ap.add_argument("--encoder_checkpoint", default=None,
                    help="If --checkpoint is a heads-only file (heads_best.ckpt), the full checkpoint to take the encoder from "
                         "(default: poly4mer_v3.ckpt next to this script).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--properties", default="tig,pkhrr,sea,co", help="Subset of heads to train, e.g. 'tig,pkhrr'.")
    ap.add_argument("--fresh_heads", action="store_true", help="Random initial heads; statistics recomputed from the training data.")
    ap.add_argument("--recompute_stats", action="store_true", help="Recompute standardization statistics from the training data.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--scheduler_step", type=int, default=10, help="Multiply lr by scheduler_gamma every this many epochs.")
    ap.add_argument("--scheduler_gamma", type=float, default=0.7)
    ap.add_argument("--wd", type=float, default=1e-5, help="Adam weight decay.")
    ap.add_argument("--patience", type=int, default=0, help="Early stopping on validation loss; 0 disables.")
    ap.add_argument("--seed", type=int, default=1995)
    ap.add_argument("--embed_batch_size", type=int, default=64, help="Batch size for the frozen encoder.")
    ap.add_argument("--latent_cache", default=None, help="Latent cache file (default: <out_dir>/latent_cache.pt).")
    ap.add_argument("--save_full", action="store_true", help="Also write a full drop-in checkpoint with the new heads.")
    ap.add_argument("--eval_only", action="store_true", help="Only evaluate the checkpoint's heads on --val_labeled.")
    cv = ap.add_argument_group("reproduction of the 10-fold experimental cross-validation")
    cv.add_argument("--cv", action="store_true",
                    help="Reproduce the 10-fold runs that selected the heads in the checkpoint.")
    cv.add_argument("--cv_data_dir", default=os.path.join(REPO_DIR, "data", "10folds_data"))
    cv.add_argument("--cv_folds", default="0-9", help="Folds to run, e.g. '0-9' or '8' or '0,3,8'.")
    cv.add_argument("--cv_lr_grid", default=None,
                    help="Override the per-property learning-rate grid, comma-separated.")
    cv.add_argument("--cv_epochs", type=int, default=None, help="Override the per-property epoch budget.")
    cv.add_argument("--cv_batch_size", type=int, default=None, help="Override the per-property batch size.")
    cv.add_argument("--cv_init_heads", default="joint", choices=["joint", "selected"],
                    help="Which head set in --checkpoint each fold starts from. The original "
                         "protocol starts from the model's own heads, 'joint', which is the "
                         "default. Starting from 'selected' is circular, because those heads "
                         "were themselves chosen from these folds.")
    args = ap.parse_args()

    props = [p.strip() for p in args.properties.split(",") if p.strip()]
    for p in props:
        if p not in PROPERTY_NAMES:
            ap.error(f"unknown property {p!r}; choose from {PROPERTY_NAMES}")
    if not args.eval_only and not args.cv and not args.train_labeled:
        ap.error("--train_labeled is required unless --eval_only or --cv")
    if args.eval_only and not args.val_labeled:
        ap.error("--eval_only needs --val_labeled")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)
    if args.cv:
        return run_cv(args, props, device)
    cache_path = args.latent_cache or os.path.join(args.out_dir, "latent_cache.pt")

    # ---- checkpoint(s) -----------------------------------------------------
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if "autoencoder_encoder" in ck:
        enc_ck = ck
    else:  # heads-only file: encoder comes from the full checkpoint
        enc_path = args.encoder_checkpoint or os.path.join(REPO_DIR, "checkpoint", "poly4mer_v3.ckpt")
        enc_ck = torch.load(enc_path, map_location="cpu", weights_only=False, mmap=True)
        print(f">> encoder taken from {enc_path}")

    # ---- data --------------------------------------------------------------
    train_df = load_labeled(parse_paths(args.train_labeled)) if args.train_labeled else None
    val_df = load_labeled(parse_paths(args.val_labeled)) if args.val_labeled else None
    all_smiles = []
    for df in (train_df, val_df):
        if df is not None:
            all_smiles += df["smiles_canonicalized"].astype(str).tolist()
    cache = embed_unique(all_smiles, cache_path, enc_ck, device, args.embed_batch_size)
    del enc_ck

    # ---- heads and statistics ---------------------------------------------
    heads = {n: build_regressor(770 if PROP_USES_THK_FLUX[n] else 768).to(device) for n in PROPERTY_NAMES}
    if args.fresh_heads:
        print(">> heads: random initialization")
        stats = compute_property_stats(train_df, device)
        print(">> statistics: computed from the training data")
    else:
        for n in PROPERTY_NAMES:
            heads[n].load_state_dict(ck["property_regressors"][n])
        print(f">> heads: loaded from {args.checkpoint}")
        if args.recompute_stats:
            stats = compute_property_stats(train_df, device)
            print(">> statistics: recomputed from the training data")
        else:
            stats = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in ck["property_stats"].items()}
            print(">> statistics: taken from the checkpoint")
    print("   y_mean:", [round(float(v), 4) for v in stats["y_mean"]], " y_std:", [round(float(v), 4) for v in stats["y_std"]])

    val = frame_to_tensors(val_df, cache, device) if val_df is not None else None
    if args.eval_only:
        m = evaluate(*val, heads, stats, props)
        print(f">> eval ({len(val_df)} rows): val_loss {m['val_loss']:.5f}\n   " + fmt_metrics(m, props))
        json.dump(m, open(os.path.join(args.out_dir, "eval_metrics.json"), "w"), indent=1)
        return

    z, thk, flux, y = frame_to_tensors(train_df, cache, device)
    n = len(z)
    print(f">> training rows {n}; validation rows {0 if val is None else len(val[0])}; heads trained: {props}")
    for p in props:
        print(f"   {p}: {int((~torch.isnan(y[:, PROPERTY_NAMES.index(p)])).sum())} labeled rows, head params {count_params(heads[p]):,}")

    params = [q for p in props for q in heads[p].parameters()]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    # ---- training loop -----------------------------------------------------
    hist_path = os.path.join(args.out_dir, "history.csv")
    cols = ["epoch", "lr", "train_loss"] + [f"train_{p}" for p in props] + ["val_loss"] + [f"val_{p}_rel_median" for p in props] + [f"val_{p}_rel_mean" for p in props]
    hist = open(hist_path, "w", newline="")
    writer = csv.DictWriter(hist, fieldnames=cols)
    writer.writeheader()

    def save_heads(path, epoch, metrics):
        torch.save({
            "property_regressors": {k: {kk: vv.detach().cpu() for kk, vv in h.state_dict().items()} for k, h in heads.items()},
            "property_stats": {k: v.detach().cpu() for k, v in stats.items()},
            "properties_trained": props,
            "epoch": epoch,
            "val_metrics": metrics,
            "meta": {"name": "poly4mer_v3 property heads, phase I fine-tune (encoder frozen)",
                     "source_checkpoint": os.path.abspath(args.checkpoint),
                     "train_files": parse_paths(args.train_labeled), "val_files": parse_paths(args.val_labeled),
                     "args": vars(args)},
        }, path)

    best_path = os.path.join(args.out_dir, "heads_best.ckpt")
    best_val, best_epoch, bad_epochs = float("inf"), -1, 0
    if val is not None:
        m0 = evaluate(*val, heads, stats, props)
        print(f">> before training: val_loss {m0['val_loss']:.5f}\n   " + fmt_metrics(m0, props))

    for ep in range(args.epochs):
        t0 = time.time()
        lr = lr_at_epoch(args.lr, ep, args.scheduler_step, args.scheduler_gamma)
        set_lr(optimizer, lr)
        for p in props:
            heads[p].train()
        perm = torch.randperm(n, device=device)
        loss_sum, part_sum, nb = 0.0, {p: 0.0 for p in props}, 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss, parts = masked_loss(z[idx], thk[idx], flux[idx], y[idx], heads, stats, props)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss)
            for p in props:
                part_sum[p] += parts.get(p, 0.0)
            nb += 1
        row = {"epoch": ep, "lr": lr, "train_loss": loss_sum / nb, **{f"train_{p}": part_sum[p] / nb for p in props}}
        msg = f"Epoch {ep:3d} ({time.time() - t0:5.1f}s) lr {lr:.2e} | train loss {row['train_loss']:.5f}"
        metrics = None
        if val is not None:
            metrics = evaluate(*val, heads, stats, props)
            row["val_loss"] = metrics["val_loss"]
            for p in props:
                row[f"val_{p}_rel_median"] = metrics[f"{p}_rel_median"]
                row[f"val_{p}_rel_mean"] = metrics[f"{p}_rel_mean"]
            msg += f" | val loss {metrics['val_loss']:.5f}\n   " + fmt_metrics(metrics, props)
            if metrics["val_loss"] < best_val:
                best_val, best_epoch, bad_epochs = metrics["val_loss"], ep, 0
                save_heads(best_path, ep, metrics)
                msg += "\n   >> new best, saved heads_best.ckpt"
            else:
                bad_epochs += 1
        else:
            save_heads(best_path, ep, None)
            best_epoch = ep
        writer.writerow(row)
        hist.flush()
        print(msg, flush=True)
        if args.patience and bad_epochs >= args.patience:
            print(f">> early stopping: no validation improvement for {args.patience} epochs")
            break
    hist.close()
    print(f">> done. best epoch {best_epoch}" + (f", best val loss {best_val:.5f}" if val is not None else "") + f" -> {best_path}")

    # ---- optional full drop-in checkpoint ---------------------------------
    if args.save_full:
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        full = {k: ck[k] for k in ("star_encoder", "autoencoder_encoder", "decoder", "predictor")} if "decoder" in ck else None
        if full is None:
            enc_path = args.encoder_checkpoint or os.path.join(REPO_DIR, "checkpoint", "poly4mer_v3.ckpt")
            src = torch.load(enc_path, map_location="cpu", weights_only=False, mmap=True)
            full = {k: src[k] for k in ("star_encoder", "autoencoder_encoder", "decoder", "predictor")}
        full["property_regressors"] = best["property_regressors"]
        full["property_stats"] = best["property_stats"]
        full["meta"] = {"name": "poly4mer_v3 with phase-I fine-tuned property heads", "phase1": best["meta"], "best_epoch": best["epoch"]}
        out = os.path.join(args.out_dir, "poly4mer_v3_phase1.ckpt")
        torch.save(full, out)
        print(f">> full checkpoint written: {out}")


if __name__ == "__main__":
    main()
