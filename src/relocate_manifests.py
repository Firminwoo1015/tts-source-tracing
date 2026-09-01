"""Rewrite the dataset prefix of `prompt_wav` / `real_wav` in data/manifests/*.jsonl.

The manifests were generated on the original machine and therefore contain
absolute LibriSpeech/VCTK paths. Run once after setting TTS_ANAL_DATASETS:

    python src/relocate_manifests.py --old /home/nas5/minwoolee/datasets [--dry-run]

(default --new = $TTS_ANAL_DATASETS, or <repo>/datasets). Idempotent.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="dataset prefix currently stored in the manifests")
    ap.add_argument("--new", default=os.environ.get("TTS_ANAL_DATASETS", str(ROOT / "datasets")))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    old, new = a.old.rstrip("/"), a.new.rstrip("/")
    for f in sorted((ROOT / "data/manifests").glob("*.jsonl")):
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        n = 0
        for r in rows:
            for k in ("prompt_wav", "real_wav"):
                if k in r and r[k].startswith(old + "/"):
                    r[k] = new + r[k][len(old):]
                    n += 1
        print(f"{f.name}: {n} fields rewritten {old} -> {new}" + (" (dry-run)" if a.dry_run else ""))
        if not a.dry_run and n:
            f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


if __name__ == "__main__":
    main()
