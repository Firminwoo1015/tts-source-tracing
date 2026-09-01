"""ASR-confound controls for the deep-layer signal (reviewer item).

(a) per-system WER/CER summary
(b) ASR-error-feature baseline: classify systems (TTS-only, speaker-disjoint
    GroupKFold, cosine kNN on standardized features) from
    [wer, sub, del, ins, cer] alone
(c) WER=0 subset: keep only utterances whose whisper transcript exactly
    matches the reference (per condition sample); recompute TTS-only
    deep-band (17-21, best in-domain layer within band) LOO accuracy per
    encoder on that subset vs the full set.
Output: results/paper/asr_summary.csv, asr_feature_baseline.csv,
        asr_perfect_deepband.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, LeaveOneOut, cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = os.environ.get("TTS_ANAL_RESULTS", "results/paper5c17")
SYSTEMS = os.environ.get("TTS_ANAL_SYSTEMS", "f5tts,xtts,cosyvoice3,chatterbox,indextts").split(",")
SYS = SYSTEMS
SSLS = ["wavlm", "hubert", "xlsr", "w2v2lv60", "w2vbert"]
DEEP = range(17, 22)
EXCLUDE = set((ROOT / os.environ.get("TTS_ANAL_EXCLUDE", "data/manifests/exclude17.txt")).read_text().split())
RNG = np.random.default_rng(0)


def main():
    df = pd.read_csv(ROOT / RESULTS / "asr_wer.csv")

    summ = df.groupby("cond")[["wer", "cer"]].agg(["mean", "median"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ["wer0_rate"] = df.groupby("cond")["wer"].apply(lambda x: (x == 0).mean())
    summ.to_csv(ROOT / RESULTS / "asr_summary.csv")
    print(summ.round(3))

    # (b) ASR-error-feature baseline, TTS-only + 6-way
    feats = ["wer", "sub", "del", "ins", "cer"]
    out = []
    for tag, sub in [(f"{df.cond.nunique()}way", df), ("ttsonly", df[df.cond != "real"])]:
        X = sub[feats].to_numpy(dtype=float)
        # force plain numpy object arrays (pyarrow-backed pandas strings
        # break sklearn's index-based splitting)
        y = np.array(sub["cond"], dtype=object)
        g = np.array(sub["speaker"].astype(str), dtype=object)
        clf = make_pipeline(StandardScaler(),
                            KNeighborsClassifier(5, metric="cosine"))
        pred = cross_val_predict(clf, X, y, cv=GroupKFold(5), groups=g, n_jobs=8)
        f1 = f1_score(y, pred, average="macro")
        sps = np.unique(g)
        boots = [f1_score(*(lambda idx: (y[idx], pred[idx]))(
            np.concatenate([np.where(g == s)[0]
                            for s in RNG.choice(sps, len(sps), True)])),
            average="macro") for _ in range(1000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append({"setting": tag, "macro_f1": f1, "ci_lo": lo, "ci_hi": hi})
        print(f"ASR-error features {tag}: {f1:.3f} [{lo:.3f},{hi:.3f}]")
    pd.DataFrame(out).to_csv(ROOT / RESULTS / "asr_feature_baseline.csv",
                             index=False)

    # (c) WER=0 subset deep-band attribution (audit fix): keep only utterance
    # IDs that EVERY TTS system read with WER=0 (balanced intersection), and
    # use NESTED layer selection within the deep band (no selection leakage).
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from analyze_bands import band_pred

    tts_df = df[df.cond != "real"]
    perfect_all = set.intersection(*[
        set(tts_df[(tts_df.cond == c) & (tts_df.wer == 0)].utt_id) for c in SYS])
    rows = []
    for ssl in SSLS:
        X, y, g, uid = [], [], [], []
        for c in SYS:
            d = np.load(ROOT / "results/embeddings" / ssl / f"{c}.npz")
            keep = ~np.isin(d["utt_ids"], list(EXCLUDE))
            X.append(d["embs"][keep])
            y += [c] * keep.sum()
            g += list(d["speakers"][keep])
            uid += list(d["utt_ids"][keep])
        X = np.concatenate(X)
        y, g, uid = np.array(y), np.array(g), np.array(uid)
        mask = np.isin(uid, list(perfect_all))
        full = f1_score(y, band_pred(X, y, g, list(DEEP)), average="macro")
        sub = f1_score(y[mask], band_pred(X[mask], y[mask], g[mask], list(DEEP)),
                       average="macro")
        rows.append({"ssl": ssl, "deepband_full": full, "deepband_wer0_all": sub,
                     "n_ids_wer0_all": len(perfect_all), "n_samples": int(mask.sum())})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(ROOT / RESULTS / "asr_perfect_deepband.csv",
                              index=False)


if __name__ == "__main__":
    main()
