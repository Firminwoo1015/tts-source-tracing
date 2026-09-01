"""Speaker-disjoint deep-band analysis (reviewer item: 4.3 numbers were
in-domain LOO). For each encoder and band (early 1-5, deep 17-21), TTS-only
GroupKFold(5) with nested inner layer selection RESTRICTED to the band;
macro-F1, speaker bootstrap CI, and paired early-minus-deep speaker CI.
Output: results/paper/bands_spkdisjoint.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = os.environ.get("TTS_ANAL_RESULTS", "results/paper5c17")
SYSTEMS = os.environ.get("TTS_ANAL_SYSTEMS", "f5tts,xtts,cosyvoice3,chatterbox,indextts").split(",")
SYS = SYSTEMS
SSLS = ["wavlm", "hubert", "xlsr", "w2v2lv60", "w2vbert"]
BANDS = {"early": range(1, 6), "deep": range(17, 22)}
EXCLUDE = set((ROOT / os.environ.get("TTS_ANAL_EXCLUDE", "data/manifests/exclude17.txt")).read_text().split())
RNG = np.random.default_rng(0)


def load_all(ssl):
    X, y, g = [], [], []
    for c in SYS:
        d = np.load(ROOT / "results/embeddings" / ssl / f"{c}.npz")
        keep = ~np.isin(d["utt_ids"], list(EXCLUDE))
        X.append(d["embs"][keep])
        y += [c] * keep.sum()
        g += list(d["speakers"][keep])
    return np.concatenate(X), np.array(y), np.array(g)


def knn():
    return KNeighborsClassifier(5, metric="cosine")


def band_pred(embs, y, g, band):
    outer = GroupKFold(5)
    pred = np.empty(len(y), dtype=object)
    for tr, te in outer.split(embs[:, 0], y, groups=g):
        best_l, best_f = band[0], -1
        inner = GroupKFold(4)
        for l in band:
            Xl = normalize(embs[tr][:, l])
            f1s = [f1_score(y[tr][ite],
                            knn().fit(Xl[itr], y[tr][itr]).predict(Xl[ite]),
                            average="macro")
                   for itr, ite in inner.split(Xl, y[tr], groups=g[tr])]
            if np.mean(f1s) > best_f:
                best_f, best_l = np.mean(f1s), l
        pred[te] = knn().fit(normalize(embs[tr][:, best_l]), y[tr]).predict(
            normalize(embs[te][:, best_l]))
    return pred.astype(str)


def main():
    rows = []
    for ssl in SSLS:
        embs, y, g = load_all(ssl)
        speakers = np.unique(g)
        preds = {}
        for bname, band in BANDS.items():
            preds[bname] = band_pred(embs, y, g, list(band))
        boots = {"early": [], "deep": [], "diff": []}
        for _ in range(1000):
            s = RNG.choice(speakers, len(speakers), replace=True)
            idx = np.concatenate([np.where(g == x)[0] for x in s])
            fe = f1_score(y[idx], preds["early"][idx], average="macro")
            fd = f1_score(y[idx], preds["deep"][idx], average="macro")
            boots["early"].append(fe)
            boots["deep"].append(fd)
            boots["diff"].append(fe - fd)
        row = {"ssl": ssl}
        for bname in BANDS:
            row[f"{bname}_f1"] = f1_score(y, preds[bname], average="macro")
            row[f"{bname}_lo"], row[f"{bname}_hi"] = np.percentile(
                boots[bname], [2.5, 97.5])
        row["earlyminusdeep_lo"], row["earlyminusdeep_hi"] = np.percentile(
            boots["diff"], [2.5, 97.5])
        rows.append(row)
        print({k: round(v, 3) if isinstance(v, float) else v
               for k, v in row.items()})
    pd.DataFrame(rows).to_csv(ROOT / RESULTS / "bands_spkdisjoint.csv",
                              index=False)


if __name__ == "__main__":
    main()
