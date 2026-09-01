"""N. Nuisance-removal stress test: embeddings extracted after deterministic silence trimming
(librosa top_db=35) and per-utterance RMS normalization (`extract_embeddings.py --trim-norm`,
tags <ssl>_tn) vs the untrimmed main embeddings, under the Table-1 nested speaker-disjoint
protocol. Output: <RESULTS>/normalization_control.csv
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
sys.path.insert(0, str(Path(__file__).parent))
from analyze_cloning import load_all, nested_spk_disjoint, speaker_bootstrap_f1, CLONING

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = os.environ.get("TTS_ANAL_RESULTS", "results/paper5c17")


def main():
    rows = []; rng = np.random.default_rng(0)
    for ssl in ["wavlm", "w2vbert"]:
        kept = {}
        for tag, name in [(ssl, "untrimmed"), (f"{ssl}_tn", "trim+rms")]:
            if not (ROOT / "results/embeddings" / tag / "real.npz").exists():
                print("missing", tag); continue
            for setting, conds in [("6way", ["real"] + CLONING), ("ttsonly", CLONING)]:
                embs, y, spk, uid = load_all(tag, conds)
                pred, chosen = nested_spk_disjoint(embs, y, spk)
                f1 = f1_score(y, pred, average="macro"); lo, hi = speaker_bootstrap_f1(y, pred, spk)
                row = {"ssl": ssl, "preprocessing": name, "setting": setting, "macro_f1": f1, "ci_lo": lo, "ci_hi": hi, "layers": str(chosen)}
                # paired speaker-bootstrap CI of (untrimmed - trim+rms) macro-F1 on the same utterances
                if name == "untrimmed":
                    kept[setting] = (y, pred, spk, uid)
                elif setting in kept:
                    y0, p0, s0, u0 = kept[setting]
                    assert len(y0) == len(y) and (u0 == uid).all() and (y0 == y).all()
                    diffs = []
                    for _ in range(1000):
                        s = rng.choice(np.unique(spk), len(np.unique(spk)), replace=True)
                        idx = np.concatenate([np.where(spk == x)[0] for x in s])
                        diffs.append(f1_score(y0[idx], p0[idx], average="macro")
                                     - f1_score(y[idx], pred[idx], average="macro"))
                    row["diff_vs_untrimmed"] = f1_score(y0, p0, average="macro") - f1
                    row["diff_lo"], row["diff_hi"] = np.percentile(diffs, [2.5, 97.5])
                rows.append(row)
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / RESULTS / "normalization_control.csv", index=False)
    print("saved", ROOT / RESULTS / "normalization_control.csv")


if __name__ == "__main__":
    main()
