"""Decoder-only estimand analysis.

All voc_* conditions decode the IDENTICAL log-mel per utterance, so any
separability between them is decoder-caused. Reports, per layer:
  - K-way LOO accuracy over vocoder conditions (in-domain)
  - K-way speaker-disjoint GroupKFold macro-F1 (+ speaker bootstrap CI at the
    pre-set layer)
  - pairwise minimum separability
Also, for each vocoder output, the speaker-excluded kNN assignment over the
reference classes (real + TTS systems) at layers 0 and 4.
Output: results/paper/deconly_<ssl>.csv, voc_vs_reference_<ssl>.csv
"""

import argparse
from itertools import combinations
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, LeaveOneOut, cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = os.environ.get("TTS_ANAL_RESULTS", "results/paper5c17")
SYSTEMS = os.environ.get("TTS_ANAL_SYSTEMS", "f5tts,xtts,cosyvoice3,chatterbox,indextts").split(",")
EXCLUDE = set((ROOT / os.environ.get("TTS_ANAL_EXCLUDE", "data/manifests/exclude17.txt")).read_text().split())
RNG = np.random.default_rng(0)


def load(ssl, cond, layer=None):
    d = np.load(ROOT / "results/embeddings" / ssl / f"{cond}.npz")
    keep = ~np.isin(d["utt_ids"], list(EXCLUDE))
    e = d["embs"][keep]
    return (e if layer is None else e[:, layer]), d["speakers"][keep]


def knn():
    return KNeighborsClassifier(5, metric="cosine")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssl", default="wavlm")
    ap.add_argument("--conds", nargs="+",
                    default=["voc_pwg", "voc_melgan", "voc_mbmelgan",
                             "voc_hifigan", "voc_stylemelgan"])
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 2, 4, 8, 19])
    ap.add_argument("--ci-layer", type=int, default=0)
    args = ap.parse_args()

    conds = [c for c in args.conds
             if (ROOT / "results/embeddings" / args.ssl / f"{c}.npz").exists()]
    print("conditions:", conds)
    data = {c: load(args.ssl, c) for c in conds}
    real = load(args.ssl, "real")

    records = []
    for layer in args.layers:
        X = normalize(np.concatenate([data[c][0][:, layer] for c in conds]))
        y = np.concatenate([[c] * len(data[c][0]) for c in conds])
        g = np.concatenate([data[c][1] for c in conds])
        loo = cross_val_score(knn(), X, y, cv=LeaveOneOut(), n_jobs=8).mean()
        pred = cross_val_predict(knn(), X, y, cv=GroupKFold(5), groups=g, n_jobs=8)
        f1 = f1_score(y, pred, average="macro")
        pmin, pmin_pair = 1.0, ""
        for a, b in combinations(conds, 2):
            Xp = normalize(np.concatenate([data[a][0][:, layer],
                                           data[b][0][:, layer]]))
            yp = np.array([0] * len(data[a][0]) + [1] * len(data[b][0]))
            acc = cross_val_score(knn(), Xp, yp, cv=LeaveOneOut(), n_jobs=8).mean()
            if acc < pmin:
                pmin, pmin_pair = acc, f"{a}|{b}"
        row = {"layer": layer, "kway_loo": loo, "kway_spkdisjoint_f1": f1,
               "pairwise_min": pmin, "pairwise_min_pair": pmin_pair}
        if layer == args.ci_layer:
            sps = np.unique(g)
            boots = []
            for _ in range(1000):
                s = RNG.choice(sps, len(sps), replace=True)
                idx = np.concatenate([np.where(g == x)[0] for x in s])
                boots.append(f1_score(y[idx], pred[idx], average="macro"))
            row["f1_ci_lo"], row["f1_ci_hi"] = np.percentile(boots, [2.5, 97.5])
        records.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v)
               for k, v in row.items()})

    df = pd.DataFrame(records)
    out = ROOT / RESULTS / f"deconly_{args.ssl}.csv"
    df.to_csv(out, index=False)
    print("saved", out)

    # vocoder outputs vs. the reference classes (real + TTS systems): speaker-excluded
    # kNN assignment of every voc_* utterance at layers 0 and 4
    refs = ["real"] + SYSTEMS
    ref = {c: load(args.ssl, c) for c in refs
           if (ROOT / "results/embeddings" / args.ssl / f"{c}.npz").exists()}
    vrows = []
    for layer in [0, 4]:
        Xr = normalize(np.concatenate([ref[c][0][:, layer] for c in ref]))
        yr = np.concatenate([[c] * len(ref[c][0]) for c in ref])
        gr = np.concatenate([ref[c][1] for c in ref])
        for c in conds:
            Xq, gq = normalize(data[c][0][:, layer]), data[c][1]
            pred = np.empty(len(gq), dtype=object)
            for spk in np.unique(gq):
                clf = knn().fit(Xr[gr != spk], yr[gr != spk])
                pred[gq == spk] = clf.predict(Xq[gq == spk])
            row = {"voc": c, "layer": layer}
            row.update({r: float((pred == r).mean()) for r in ref})
            vrows.append(row)
    vout = ROOT / RESULTS / f"voc_vs_reference_{args.ssl}.csv"
    pd.DataFrame(vrows).to_csv(vout, index=False)
    print("saved", vout)


if __name__ == "__main__":
    main()
