"""
evaluate_fire_prediction_heads.py: 10-fold evaluation of the fire property heads
=================================================================================

For each property, loads the 10 per-fold heads from
`checkpoint/fire_property_heads/<property>/fold_<k>.ckpt`, runs each one on its own
held-out test fold in `data/10folds_data/<property>/data_<k>/test.csv`, and reports:

  * the full per-fold table: train and test mean relative error
  * the mean over the best 5 folds, which is the number quoted in the paper
  * the mean over all 10 folds
  * the best overall relative error: every fold's head scored on the WHOLE property
    set, with the lowest value reported

If a head is accompanied by a `fold_<k>.txt` recording the learning rate, epoch and
errors it was trained with, those are printed alongside for comparison. The bundled
heads ship without them; heads written by `--cv` carry them.

The last one needs care. A fold's head has trained on roughly 90% of the rows it is
then scored on, so the overall error is optimistic by construction. It is the criterion
that picked the heads bundled in poly4mer_v3.ckpt, and it is reported here so that
choice is reproducible, not because it estimates generalization. Quote the best-5 or
all-10 test means for that.

The encoder is frozen and comes from `checkpoint/poly4mer_v3.ckpt`, which also supplies
the standardization statistics. tig and pHRR heads take [z, thickness, flux] and so need
those columns in the CSV; SEA and CO heads take the latent alone.

Examples (run from inside this folder):
    python evaluate_fire_prediction_heads.py
    python evaluate_fire_prediction_heads.py --properties tig,pkhrr --csv folds.csv
"""
import argparse, os, re, sys
import numpy as np, pandas as pd, torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # repository root
for _p in (SCRIPT_DIR, os.path.join(SCRIPT_DIR, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import star_encoder, AutoEncoderLayer3
from utils1 import load_smi_ted_explicit, find_smi_ted_dir

PROPS = ["tig", "pkhrr", "sea", "co"]
PRETTY = {"tig": "tig", "pkhrr": "pHRR", "sea": "SEA", "co": "CO"}
# which heads also take thickness and flux (input dim 770 instead of 768)
USES_THK_FLUX = {"tig": True, "pkhrr": True, "sea": False, "co": False}
# the target column name in each property's CSVs
TARGET = {"tig": "tig", "pkhrr": "pkhrr", "sea": "SEA", "co": "CO"}
MAX_LEN, EMB_DIM, STAR_TOKEN_ID = 202, 768, 439


def build_regressor(input_dim):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, 512), nn.GELU(),
        nn.Linear(512, 512), nn.GELU(),
        nn.Linear(512, 256), nn.GELU(),
        nn.Linear(256, 128), nn.GELU(),
        nn.Linear(128, 1))


def mean_rel(pred, y):
    m = np.abs(y) > 1e-12
    return float(np.mean(np.abs(pred[m] - y[m]) / np.abs(y[m])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--properties", default=",".join(PROPS))
    ap.add_argument("--checkpoint", default=os.path.join(SCRIPT_DIR, "checkpoint", "poly4mer_v3.ckpt"))
    ap.add_argument("--heads_dir", default=os.path.join(SCRIPT_DIR, "checkpoint", "fire_property_heads"))
    ap.add_argument("--data_dir", default=os.path.join(SCRIPT_DIR, "data", "10folds_data"))
    ap.add_argument("--top_k", type=int, default=5, help="how many best folds to average")
    ap.add_argument("--csv", default=None, help="write the per-fold table here")
    ap.add_argument("--batch_size", type=int, default=128)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> device {dev}")
    print(f">> encoder {os.path.relpath(a.checkpoint, SCRIPT_DIR)}")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False, mmap=True)

    A = star_encoder(dim=1)
    ae = AutoEncoderLayer3(MAX_LEN * EMB_DIM, EMB_DIM * 4, EMB_DIM * 2, EMB_DIM)
    A.load_state_dict(ck["star_encoder"])
    ae.encoder.load_state_dict(ck["autoencoder_encoder"])
    A.to(dev).eval(); ae.to(dev).eval()

    st = {k: torch.as_tensor(v, dtype=torch.float32, device=dev) for k, v in ck["property_stats"].items()}
    thk_m, thk_s = st["thk_mean"].reshape(()), st["thk_std"].reshape(())
    flx_m, flx_s = st["flux_mean"].reshape(()), st["flux_std"].reshape(())
    y_mean, y_std = st["y_mean"].view(-1), st["y_std"].view(-1)

    smi_dir = find_smi_ted_dir()
    Smi, MolTok = load_smi_ted_explicit()
    S = Smi(MolTok(os.path.join(smi_dir, "bert_vocab_curated.txt")))
    S.load_checkpoint(os.path.join(smi_dir, "smi-ted-Light_40.pt"))
    S.eval()
    for m in (A, ae.encoder, S):
        for p in m.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def embed(smiles):
        out = []
        for i in range(0, len(smiles), a.batch_size):
            idx, te, _ = S.extract_embeddings(smiles[i:i + a.batch_size])
            idx, te = idx.to(dev), te.to(dev)
            pos = torch.nonzero(idx == STAR_TOKEN_ID, as_tuple=False)
            if pos.numel():
                te[pos[:, 0], pos[:, 1], :] = A(idx[pos[:, 0], pos[:, 1]].view(-1, 1).float())
            out.append(ae.encoder(te.view(-1, MAX_LEN * EMB_DIM)))
        return torch.cat(out)

    def make_xy(df, prop):
        df = df.copy()
        cols = ["thickness", "flux", TARGET[prop]] if USES_THK_FLUX[prop] else [TARGET[prop]]
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["smiles_canonicalized"] + cols).reset_index(drop=True)
        z = embed(df["smiles_canonicalized"].astype(str).tolist())
        if USES_THK_FLUX[prop]:
            thk = torch.tensor(df["thickness"].values, dtype=torch.float32, device=dev).view(-1, 1)
            flx = torch.tensor(df["flux"].values, dtype=torch.float32, device=dev).view(-1, 1)
            X = torch.cat([z, (thk - thk_m) / thk_s, (flx - flx_m) / flx_s], dim=-1)
        else:
            X = z
        return X, df[TARGET[prop]].values.astype(np.float64)

    all_rows, summary = [], {}
    for prop in [p.strip() for p in a.properties.split(",") if p.strip()]:
        k_idx = PROPS.index(prop)
        ddir = os.path.join(a.data_dir, prop)
        hdir = os.path.join(a.heads_dir, prop)
        if not os.path.isdir(ddir):
            print(f"!! no data for {prop} at {ddir}, skipped"); continue

        # cache the features per fold, and build the union set for the overall score
        folds, seen, uX, uy = {}, set(), [], []
        for k in range(10):
            tr = pd.read_csv(os.path.join(ddir, f"data_{k}", "train.csv"))
            te = pd.read_csv(os.path.join(ddir, f"data_{k}", "test.csv"))
            folds[k] = (make_xy(tr, prop), make_xy(te, prop))
            for _, r in te.iterrows():                       # test folds are disjoint
                seen.add(len(seen))
            Xte, yte = folds[k][1]
            uX.append(Xte); uy.append(yte)
        Xall, yall = torch.cat(uX), np.concatenate(uy)

        @torch.no_grad()
        def predict(head, X):
            return (head(X).view(-1) * y_std[k_idx] + y_mean[k_idx]).cpu().numpy().astype(np.float64)

        rows = []
        for k in range(10):
            f = os.path.join(hdir, f"fold_{k}.ckpt")
            if not os.path.exists(f):
                print(f"!! missing {f}"); continue
            head = build_regressor(770 if USES_THK_FLUX[prop] else 768).to(dev)
            head.load_state_dict(torch.load(f, map_location=dev))
            head.eval()
            (Xtr, ytr), (Xte, yte) = folds[k]
            rec = dict(property=prop, fold=k, n_train=len(ytr), n_test=len(yte),
                       train_rel=100 * mean_rel(predict(head, Xtr), ytr),
                       test_rel=100 * mean_rel(predict(head, Xte), yte),
                       overall_rel=100 * mean_rel(predict(head, Xall), yall))
            # the recorded values that shipped with the head, for cross-checking
            t = os.path.join(hdir, f"fold_{k}.txt")
            if os.path.exists(t):
                g = dict(re.findall(r"(\w+)=([-\d.eE+]+)", open(t).read()))
                rec["recorded_test_rel"] = 100 * float(g["test_rel"])
                rec["best_lr"] = float(g["best_lr"])
                rec["best_epoch"] = int(float(g["best_epoch"]))
            rows.append(rec)
            del head
        if not rows:
            continue
        all_rows += rows

        te = np.array([r["test_rel"] for r in rows])
        top = np.argsort(te)[:a.top_k]
        ov = np.array([r["overall_rel"] for r in rows])
        best_ov = int(np.argmin(ov))
        summary[prop] = dict(top_folds=sorted(int(rows[i]["fold"]) for i in top),
                             top_mean=float(te[top].mean()), top_sd=float(te[top].std(ddof=1)),
                             all_mean=float(te.mean()), all_sd=float(te.std(ddof=1)),
                             best_overall_fold=int(rows[best_ov]["fold"]),
                             best_overall=float(ov[best_ov]))

        print(f"\n=== {PRETTY[prop]}   ({len(yall)} rows across the 10 test folds) ===")
        has_rec = "recorded_test_rel" in rows[0]
        head_fmt = f"{'fold':>5}{'n_tr':>6}{'n_te':>6}"
        if has_rec:
            head_fmt += f"{'lr':>9}{'epoch':>7}"
        head_fmt += f"{'train %':>10}{'test %':>9}{'overall %':>11}"
        print(head_fmt + ("   recorded" if has_rec else ""))
        for r in rows:
            line = f"{r['fold']:>5}{r['n_train']:>6}{r['n_test']:>6}"
            if has_rec:
                line += f"{r['best_lr']:>9.0e}{r['best_epoch']:>7}"
            line += f"{r['train_rel']:>10.2f}{r['test_rel']:>9.2f}{r['overall_rel']:>11.2f}"
            if has_rec:
                d = r["test_rel"] - r["recorded_test_rel"]
                line += f"   {r['recorded_test_rel']:7.2f} ({d:+.3f})"
            if r["fold"] in summary[prop]["top_folds"]:
                line += "  *"
            print(line)
        s = summary[prop]
        print(f"  best-{a.top_k} folds {s['top_folds']}: {s['top_mean']:.2f} +/- {s['top_sd']:.2f} %")
        print(f"  all-10 mean: {s['all_mean']:.2f} +/- {s['all_sd']:.2f} %")
        print(f"  best overall relative error: fold {s['best_overall_fold']} = {s['best_overall']:.2f} %")

    print("\n" + "=" * 78)
    print(f"{'property':<10}{'best-'+str(a.top_k):>12}{'all-10':>12}{'best overall':>16}{'overall fold':>14}")
    for prop, s in summary.items():
        print(f"{PRETTY[prop]:<10}{s['top_mean']:>12.2f}{s['all_mean']:>12.2f}"
              f"{s['best_overall']:>16.2f}{s['best_overall_fold']:>14}")
    print("=" * 78)
    print("best-k and all-10 are held-out test errors. 'best overall' scores each head on the\n"
          "whole property set, about 90% of which it trained on, so it is optimistic by\n"
          "construction; it is the criterion that selected the heads in poly4mer_v3.ckpt.")

    if a.csv and all_rows:
        pd.DataFrame(all_rows).to_csv(a.csv, index=False)
        print(f"\n>> wrote {a.csv} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
