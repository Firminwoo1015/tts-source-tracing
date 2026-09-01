"""B. Cross-condition stress test with a fair three-condition comparison (see analysis_plan.md).

Per perturbation and outer speaker-disjoint fold (GroupKFold(5) as in Table 1):
  clean->clean        reference = clean train speakers,      query = clean test speakers
  perturbed->perturbed reference = perturbed train speakers, query = perturbed test speakers (matched)
  clean->perturbed    reference = clean train speakers,      query = the SAME perturbed test files (mismatched)
Layer: chosen per fold by inner GroupKFold(4) on the CLEAN training speakers (cosine kNN k=5,
all 25 layers) and applied unchanged to the three conditions. Metric: 6-way macro-F1 over the
pooled outer predictions; for stochastic perturbations (noise20, phaserand) every available
realization (tags <ssl>_p_<pert>, _s2, _s3) is evaluated and averaged. CI: speaker-paired
bootstrap of (matched - mismatched). Output: <RESULTS>/robustB3_<ssl>.csv
"""
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
sys.path.insert(0, str(Path(__file__).parent))
from analyze_cloning import load_all, CLONING

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = os.environ.get("TTS_ANAL_RESULTS", "results/paper5c17")
RNG = np.random.default_rng(0)
CONDS = ["real"] + CLONING
STOCH = {"noise20", "phaserand"}


def knn():
    return KNeighborsClassifier(5, metric="cosine")


def clean_layers(embs, y, g):
    """per outer fold: layer chosen by inner CV on the clean training speakers"""
    folds, layers = [], []
    for tr, te in GroupKFold(5).split(embs[:, 0], y, groups=g):
        best, bf = 0, -1
        for L in range(embs.shape[1]):
            f = []
            for itr, ite in GroupKFold(4).split(embs[tr][:, L], y[tr], groups=g[tr]):
                m = knn().fit(normalize(embs[tr][itr][:, L]), y[tr][itr])
                f.append(f1_score(y[tr][ite], m.predict(normalize(embs[tr][ite][:, L])), average="macro"))
            if np.mean(f) > bf: bf, best = np.mean(f), L
        folds.append((tr, te)); layers.append(best)
    return folds, layers


def align(ref_uid, ref_cond, X, uid, cond):
    """reorder (X) to the reference (uid, cond) ordering; returns None if incomplete"""
    key = {(u, c): i for i, (u, c) in enumerate(zip(uid, cond))}
    try:
        idx = [key[(u, c)] for u, c in zip(ref_uid, ref_cond)]
    except KeyError:
        return None
    return X[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssl", default="wavlm")
    ap.add_argument("--perts", nargs="+", default=["common", "common_sym", "mp3_64k", "lp4k", "hp2k", "noise20", "phaserand"])
    a = ap.parse_args()
    Xc, y, g, uid = load_all(a.ssl, CONDS)
    folds, layers = clean_layers(Xc, y, g)
    print("clean-selected layers per fold:", layers, flush=True)
    rows = []
    for pert in a.perts:
        tags = [f"{a.ssl}_p_{pert}"] + ([f"{a.ssl}_p_{pert}_s{i}" for i in (2, 3)] if pert in STOCH else [])
        per_real = {"cc": [], "pp": [], "cp": []}; spk_pairs = []
        for tag in tags:
            if not (ROOT / "results/embeddings" / tag / "real.npz").exists():
                print("  missing", tag); continue
            Xp_raw, yp, gp, uidp = load_all(tag, CONDS)
            Xp = align(uid, y, Xp_raw, uidp, yp)
            if Xp is None:
                print("  incomplete", tag); continue
            pred = {k: np.empty(len(y), dtype=object) for k in per_real}
            for (tr, te), L in zip(folds, layers):
                pred["cc"][te] = knn().fit(normalize(Xc[tr][:, L]), y[tr]).predict(normalize(Xc[te][:, L]))
                pred["pp"][te] = knn().fit(normalize(Xp[tr][:, L]), y[tr]).predict(normalize(Xp[te][:, L]))
                pred["cp"][te] = knn().fit(normalize(Xc[tr][:, L]), y[tr]).predict(normalize(Xp[te][:, L]))
            for k in per_real: per_real[k].append(pred[k].astype(str))
        if not per_real["pp"]:
            continue
        def f1(p, idx=None):
            idx = np.arange(len(y)) if idx is None else idx
            return f1_score(y[idx], p[idx], average="macro")
        res = {k: float(np.mean([f1(p) for p in per_real[k]])) for k in per_real}
        # speaker-paired bootstrap of matched - mismatched (averaged over realizations)
        sps = np.unique(g); diffs = []
        for _ in range(1000):
            s = RNG.choice(sps, len(sps), replace=True); idx = np.concatenate([np.where(g == x)[0] for x in s])
            diffs.append(np.mean([f1(pp, idx) - f1(cp, idx) for pp, cp in zip(per_real["pp"], per_real["cp"])]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append({"pert": pert, "n_realizations": len(per_real["pp"]), "layers": str(layers),
                     "clean_to_clean": res["cc"], "perturbed_to_perturbed": res["pp"], "clean_to_perturbed": res["cp"],
                     "matched_minus_mismatched": res["pp"] - res["cp"], "diff_ci_lo": lo, "diff_ci_hi": hi})
        print(pert, {k: round(v, 3) for k, v in res.items()}, f"diff CI [{lo:.3f},{hi:.3f}]", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / RESULTS / f"robustB3_{a.ssl}.csv", index=False)
    print("saved", ROOT / RESULTS / f"robustB3_{a.ssl}.csv")


if __name__ == "__main__":
    for ssl in (sys.argv[sys.argv.index("--ssl") + 1:sys.argv.index("--ssl") + 2] if "--ssl" in sys.argv else ["wavlm", "w2vbert"]):
        if "--ssl" not in sys.argv:
            sys.argv += ["--ssl", ssl]; main(); sys.argv = sys.argv[:-2]
        else:
            main(); break
