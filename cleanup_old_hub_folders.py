"""Delete the four folders from the old HuggingFace layout.

They held a mix of plain and monotone heads under names that did not say which:
tig_10folds and phrr_10folds were monotone, sea_10folds and co_10folds were plain.
The replacements are property_heads_10folds/ and monotone_heads_10folds/.

Run this only after the new folders are uploaded. It is a dry run unless --confirm.
"""
import argparse
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="fishmoon1234/poly4mer")
ap.add_argument("--confirm", action="store_true", help="actually delete")
a = ap.parse_args()

api = HfApi()
present = set(api.list_repo_files(a.repo, repo_type="model"))
targets = [f"poly4mer_pretrained_v3/{p}_10folds" for p in ("tig", "phrr", "sea", "co")]

for t in targets:
    n = sum(1 for f in present if f.startswith(t + "/"))
    if n == 0:
        print(f"   already gone: {t}")
        continue
    if not a.confirm:
        print(f"   would delete: {t}  ({n} files)")
        continue
    api.delete_folder(path_in_repo=t, repo_id=a.repo, repo_type="model")
    print(f"   deleted: {t}  ({n} files)")

if not a.confirm:
    print("\nDry run. Re-run with --confirm to delete.")
