"""Post hoc voting sensitivity of the primary nested kNN evaluation (Table 1).

Added 2026-09-22 after submission-stage review; not a pre-specified analysis.
For each SSL encoder and setting (six-way, TTS-only) this script first reproduces
analyze_cloning.nested_spk_disjoint exactly: outer GroupKFold(5) over speakers,
inner GroupKFold(4) layer selection, cosine kNN k=5, uniform vote with ties to the
first class in sorted label order. It checks the reproduced outer folds against
data/manifests/speaker_folds.json, the chosen layers against cloning_summary_<ssl>.csv,
the pooled macro-F1 and speaker-bootstrap CI against the published values, and every
query's uniform-vote prediction against sklearn's predict.

On the same outer folds, chosen layers and top-5 neighbour sets it then compares
  A  tie-break sensitivity: only queries with a tied uniform vote change; the tied class
     whose top-5 neighbours have the smallest mean cosine distance wins (sorted-label
     rule if the distances are exactly equal);
  B  inverse-distance voting on every query, w = 1 / max(d_cos, 1e-6) as in
     analyze_knn_mechanism.py, exact ties to the first class in sorted label order.
Macro-F1 differences carry speaker-paired bootstrap CIs (1,000 replicates, same
resampling as the primary CI). These CIs hold the folds and chosen layers fixed, so they
exclude layer-selection uncertainty.

Outputs (results/final/, separate from the primary cloning_summary_<ssl>.csv):
  primary_knn_sensitivity.csv            one row per encoder x setting x method
  primary_knn_sensitivity_perclass.csv   per-class recall under each method
  primary_knn_sensitivity_confusion.csv  row-normalised confusion, long format
  primary_knn_sensitivity_queries.csv.gz per-query fold, layer, neighbours' votes, predictions
  primary_knn_sensitivity_config.json    protocol constants and reproduction checks
Usage: python src/analyze_primary_knn_sensitivity.py [--ssls wavlm hubert ...]
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent))
from analyze_cloning import CLONING, knn, load_all  # noqa: E402

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
PUB = ROOT / "results" / "final"
OUT = ROOT / os.environ.get("TTS_ANAL_RESULTS", "results/final")
SSLS = ["wavlm", "hubert", "xlsr", "w2v2lv60", "w2vbert"]
N_BOOT = 1000
EPS = 1e-6


def nested_with_neighbours(embs, y, spk):
    """analyze_cloning.nested_spk_disjoint, keeping the fold, layer and top-5 neighbours."""
    n = len(y)
    fold = np.full(n, -1); layer_of = np.full(n, -1)
    p_sk = np.empty(n, dtype=object)
    nb_lab = np.empty((n, 5), dtype=object); nb_dist = np.zeros((n, 5))
    chosen, fold_spk = [], []
    for f, (tr, te) in enumerate(GroupKFold(n_splits=5).split(embs[:, 0, :], y, groups=spk)):
        best_layer, best_f1 = 0, -1
        inner = GroupKFold(n_splits=4)
        for layer in range(embs.shape[1]):
            Xl = normalize(embs[tr][:, layer, :])
            f1s = []
            for itr, ite in inner.split(Xl, y[tr], groups=spk[tr]):
                p = knn().fit(Xl[itr], y[tr][itr]).predict(Xl[ite])
                f1s.append(f1_score(y[tr][ite], p, average="macro"))
            m = np.mean(f1s)
            if m > best_f1:
                best_f1, best_layer = m, layer
        chosen.append(best_layer); fold_spk.append(sorted(set(spk[te])))
        Xtr = normalize(embs[tr][:, best_layer, :]); Xte = normalize(embs[te][:, best_layer, :])
        clf = knn().fit(Xtr, y[tr])
        p_sk[te] = clf.predict(Xte)
        dist, ind = clf.kneighbors(Xte)
        nb_lab[te] = y[tr][ind]; nb_dist[te] = dist
        fold[te] = f; layer_of[te] = best_layer
    return p_sk.astype(str), nb_lab.astype(str), nb_dist, fold, layer_of, chosen, fold_spk


def votes(nb_lab, nb_dist, classes):
    """Uniform vote (ties -> first sorted class), rule A and rule B on fixed neighbour sets."""
    n, C = len(nb_lab), len(classes)
    idx = np.searchsorted(classes, nb_lab)
    uni = np.empty(n, int); a = np.empty(n, int); b = np.empty(n, int); tie = np.zeros(n, bool)
    for i in range(n):
        counts = np.bincount(idx[i], minlength=C)
        winners = np.flatnonzero(counts == counts.max())
        uni[i] = winners[0]; tie[i] = len(winners) > 1
        if tie[i]:
            meand = np.array([nb_dist[i][idx[i] == c].mean() for c in winners])
            a[i] = winners[np.argmin(meand)]          # argmin keeps sorted order on exact equality
        else:
            a[i] = uni[i]
        w = np.bincount(idx[i], weights=1.0 / np.maximum(nb_dist[i], EPS), minlength=C)
        b[i] = np.argmax(w)                           # first maximum = sorted-label rule
    return classes[uni], classes[a], classes[b], tie


def boot_ci(y, preds, spk, rng, n=N_BOOT):
    """Percentile CI of macro-F1 for each prediction vector and of each difference to preds[0],
    all on the same speaker resamples."""
    speakers = np.unique(spk)
    rows = {k: [] for k in preds}
    base = list(preds)[0]
    for _ in range(n):
        s = rng.choice(speakers, len(speakers), replace=True)
        idx = np.concatenate([np.where(spk == sp)[0] for sp in s])
        f = {k: f1_score(y[idx], p[idx], average="macro") for k, p in preds.items()}
        for k in preds:
            rows[k].append(f[k] - f[base] if k != base else f[k])
    return {k: np.percentile(v, [2.5, 97.5]) for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssls", nargs="+", default=SSLS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    folds_manifest = json.loads((ROOT / "data/manifests/speaker_folds.json").read_text())
    manifest_partition = sorted(sorted(k for k, v in folds_manifest.items() if v == f) for f in set(folds_manifest.values()))
    summ, perclass, conf, qrows, checks = [], [], [], [], []
    for ssl in args.ssls:
        rng_pub = np.random.default_rng(0)            # analyze_cloning.RNG, consumed six-way then TTS-only
        pub = pd.read_csv(PUB / f"cloning_summary_{ssl}.csv").set_index("setting")
        for ti, (tag, conds) in enumerate([(f"{len(CLONING) + 1}way", ["real"] + CLONING), ("ttsonly", CLONING)]):
            embs, y, spk, uid = load_all(ssl, conds)
            p_sk, nb_lab, nb_dist, fold, layer_of, chosen, fold_spk = nested_with_neighbours(embs, y, spk)
            classes = np.unique(y)
            p_uni, p_a, p_b, tie = votes(nb_lab, nb_dist, classes)
            # reproduction checks
            f_uni = f1_score(y, p_uni, average="macro")
            lo_pub, hi_pub = boot_ci(y, {"u": p_uni}, spk, rng_pub)["u"]
            pr = pub.loc[tag]
            chk = {"ssl": ssl, "setting": tag,
                   "uniform_equals_sklearn_predict": bool((p_uni == p_sk).all()),
                   "folds_match_manifest": sorted(fold_spk) == manifest_partition,
                   "chosen_layers": chosen, "chosen_layers_published": pr.chosen_layers,
                   "chosen_layers_match": str(chosen) == pr.chosen_layers,
                   "macro_f1": f_uni, "macro_f1_published": float(pr.spkdisjoint_macro_f1),
                   "ci": [float(lo_pub), float(hi_pub)], "ci_published": [float(pr.ci_lo), float(pr.ci_hi)]}
            chk["macro_f1_match"] = abs(chk["macro_f1"] - chk["macro_f1_published"]) < 1e-12
            chk["ci_match"] = bool(np.allclose(chk["ci"], chk["ci_published"], atol=1e-12))
            checks.append(chk); print(json.dumps(chk), flush=True)
            # sensitivity
            preds = {"uniform": p_uni, "tiebreak_distance": p_a, "inverse_distance": p_b}
            ci = boot_ci(y, preds, spk, np.random.default_rng([20260922, SSLS.index(ssl), ti]))
            for m, p in preds.items():
                ch = p != p_uni
                summ.append({"ssl": ssl, "setting": tag, "method": m, "n_queries": len(y),
                             "n_ties_uniform": int(tie.sum()), "tie_rate_uniform": float(tie.mean()),
                             "changed_rate": float(ch.mean()),
                             "changed_rate_tied": float(ch[tie].mean()) if tie.any() else np.nan,
                             "changed_rate_untied": float(ch[~tie].mean()),
                             "macro_f1": f1_score(y, p, average="macro"),
                             "diff_vs_uniform": f1_score(y, p, average="macro") - f_uni,
                             "diff_ci_lo": float(ci[m][0]) if m != "uniform" else np.nan,
                             "diff_ci_hi": float(ci[m][1]) if m != "uniform" else np.nan,
                             "chosen_layers": str(chosen)})
                rec = recall_score(y, p, labels=conds, average=None)
                rec_u = recall_score(y, p_uni, labels=conds, average=None)
                for c, r, ru in zip(conds, rec, rec_u):
                    perclass.append({"ssl": ssl, "setting": tag, "method": m, "class": c,
                                     "recall": r, "recall_change_vs_uniform": r - ru})
                for t in conds:
                    for q in conds:
                        conf.append({"ssl": ssl, "setting": tag, "method": m, "true": t, "pred": q,
                                     "rate": float(np.mean(p[y == t] == q))})
            for i in range(len(y)):
                qrows.append({"ssl": ssl, "setting": tag, "uid": uid[i], "speaker": spk[i], "true": y[i],
                              "outer_fold": int(fold[i]), "layer": int(layer_of[i]),
                              "neighbour_labels": "|".join(nb_lab[i]),
                              "neighbour_dist": "|".join(f"{d:.6f}" for d in nb_dist[i]),
                              "tie_uniform": bool(tie[i]), "pred_uniform": p_uni[i],
                              "pred_tiebreak_distance": p_a[i], "pred_inverse_distance": p_b[i]})
    pd.DataFrame(summ).to_csv(OUT / "primary_knn_sensitivity.csv", index=False)
    pd.DataFrame(perclass).to_csv(OUT / "primary_knn_sensitivity_perclass.csv", index=False)
    pd.DataFrame(conf).to_csv(OUT / "primary_knn_sensitivity_confusion.csv", index=False)
    # mtime 0 keeps the gzip header, and so the file's bytes, independent of when it is written
    pd.DataFrame(qrows).to_csv(OUT / "primary_knn_sensitivity_queries.csv.gz", index=False,
                               compression={"method": "gzip", "mtime": 0})
    cfg = {"k": 5, "metric": "cosine", "normalization": "L2 per layer", "tie_rule_uniform": "first class in sorted label order",
           "rule_A": "tied queries only; tied class with smallest mean cosine distance among its top-5 neighbours; exact equality -> sorted-label rule",
           "rule_B": "all queries; class weight = sum of 1/max(d_cos, 1e-6); exact equality -> sorted-label rule",
           "bootstrap": f"{N_BOOT} speaker-paired replicates, percentile 95%, rng default_rng([20260922, ssl_index, setting_index])",
           "ci_scope": "folds and chosen layers fixed; excludes layer-selection uncertainty",
           "status": "post hoc sensitivity analysis added 2026-09-22; not pre-specified",
           "checks": checks}
    (OUT / "primary_knn_sensitivity_config.json").write_text(json.dumps(cfg, indent=1, default=str))
    print("saved", OUT)


if __name__ == "__main__":
    main()
