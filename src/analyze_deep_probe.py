"""A1. Deep-layer representation decodability beyond kNN (see analysis_plan.md).

For each encoder and band (early 1-5, deep 17-21), TTS-only 5-way source tracing under the
SAME outer speaker-disjoint GroupKFold(5) as Table 1, with three classifiers on L2-normalized
mean-pooled embeddings:
  knn      cosine kNN, k=5 (reference protocol)
  logreg   multinomial logistic regression (L2, lbfgs), C in {0.1, 1, 10} chosen by inner CV
  centroid nearest centroid (cosine; centroids from the outer-training speakers only)
Layer (and C) are selected by inner GroupKFold(4) on the training speakers within the band.
Output: <RESULTS>/deep_probe.csv (macro-F1, speaker-bootstrap CI, chosen layers per fold).
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import normalize
sys.path.insert(0, str(Path(__file__).parent))
from analyze_bands import load_all, SSLS, BANDS

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = os.environ.get("TTS_ANAL_RESULTS", "results/paper5c17")
RNG = np.random.default_rng(0)
CGRID = [0.1, 1.0, 10.0]


def make(clf, C=1.0):
    if clf == "knn":
        return KNeighborsClassifier(5, metric="cosine")
    if clf == "logreg":
        return LogisticRegression(C=C, max_iter=2000)
    return NearestCentroid()


def nested(embs, y, g, band, clf):
    outer = GroupKFold(5); preds = np.empty(len(y), dtype=object); chosen = []
    for tr, te in outer.split(embs[:, 0], y, groups=g):
        best, best_f1 = None, -1
        for L in band:
            for C in (CGRID if clf == "logreg" else [1.0]):
                f1s = []
                for itr, ite in GroupKFold(4).split(embs[tr][:, L], y[tr], groups=g[tr]):
                    Xtr = normalize(embs[tr][itr][:, L]); Xte = normalize(embs[tr][ite][:, L])
                    m = make(clf, C).fit(Xtr, y[tr][itr])
                    f1s.append(f1_score(y[tr][ite], m.predict(Xte), average="macro"))
                if np.mean(f1s) > best_f1:
                    best_f1, best = np.mean(f1s), (L, C)
        L, C = best
        m = make(clf, C).fit(normalize(embs[tr][:, L]), y[tr])
        preds[te] = m.predict(normalize(embs[te][:, L])); chosen.append((int(L), C))
    return preds.astype(str), chosen


def boot_f1(y, pred, g, n=1000):
    sps = np.unique(g); vals = []
    for _ in range(n):
        s = RNG.choice(sps, len(sps), replace=True)
        idx = np.concatenate([np.where(g == x)[0] for x in s])
        vals.append(f1_score(y[idx], pred[idx], average="macro"))
    return np.percentile(vals, [2.5, 97.5])


def main():
    out = ROOT / RESULTS / "deep_probe.csv"
    if out.exists() and not os.environ.get("TTS_ANAL_FORCE"):
        n = len(pd.read_csv(out))
        if n >= len(SSLS) * len(BANDS) * 3:
            print(f"skip: {out} complete ({n} rows); set TTS_ANAL_FORCE=1 to recompute"); return
    rows = []
    for ssl in SSLS:
        embs, y, g = load_all(ssl)
        for bname, band in BANDS.items():
            for clf in ["knn", "logreg", "centroid"]:
                pred, chosen = nested(embs, y, g, list(band), clf)
                f1 = f1_score(y, pred, average="macro"); lo, hi = boot_f1(y, pred, g)
                rows.append({"ssl": ssl, "band": bname, "classifier": clf, "macro_f1": f1,
                             "ci_lo": lo, "ci_hi": hi, "chosen": str(chosen)})
                print(ssl, bname, clf, f"{f1:.3f} [{lo:.3f},{hi:.3f}]", chosen, flush=True)
        pd.DataFrame(rows).to_csv(ROOT / RESULTS / "deep_probe.csv", index=False)
    print("saved", ROOT / RESULTS / "deep_probe.csv")


if __name__ == "__main__":
    main()
