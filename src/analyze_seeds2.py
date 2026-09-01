"""D. Generation-seed sensitivity, cross-seed transfer protocol (see analysis_plan.md).

Data: data/manifests/seedsub20.jsonl (20 speakers, 4 per outer fold, x 3 utterances), wrapper
seeds 101-103 for each system (embeddings tag <ssl>_seeds20, conditions <system>_s<seed>),
plus the original-run outputs of the same IDs (main embeddings, tag <ssl>).
For each outer fold (GroupKFold(5) over all speakers, as in Table 1): reference = ORIGINAL-run
TTS-only embeddings of the training speakers at the layer chosen by inner CV on the original run;
queries = original-run / seed-101 / -102 / -103 outputs of the fold's test speakers among the 20.
Reports per-query-set macro-F1 and per-system recall with speaker-paired bootstrap CIs (20 query
speakers), the drop original->seed vs original->original, the geometry check (same-system
cross-seed distance vs cross-system same-utterance distance, original layer 0 and the chosen
layers) and a waveform-level stochasticity check (SHA-256 / mean abs sample difference).
Output: <RESULTS>/seeds2_<ssl>.csv, seeds2_geometry.csv, seeds2_stochasticity.csv
"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
sys.path.insert(0, str(Path(__file__).parent))
from analyze_cloning import load_all, CLONING

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = os.environ.get("TTS_ANAL_RESULTS", "results/paper5c17")
RNG = np.random.default_rng(0)
SEEDS = [101, 102, 103]


def knn():
    return KNeighborsClassifier(5, metric="cosine")


def load_seed(ssl, sysname, seed, sub_ids):
    d = np.load(ROOT / "results/embeddings" / f"{ssl}_seeds20" / f"{sysname}_s{seed}.npz")
    keep = np.isin(d["utt_ids"], sub_ids)
    return d["embs"][keep], d["utt_ids"][keep], d["speakers"][keep]


def main():
    sub = [json.loads(l) for l in open(ROOT / "data/manifests/seedsub20.jsonl")]
    sub_ids = np.array([r["utt_id"] for r in sub])
    out_rows, geo_rows = [], []
    for ssl in ["wavlm", "w2vbert", "hubert", "xlsr", "w2v2lv60"]:
        if not (ROOT / "results/embeddings" / f"{ssl}_seeds20").exists():
            print("missing seeds20 embeddings for", ssl); continue
        X, y, g, uid = load_all(ssl, CLONING)  # original run, TTS-only
        # per-fold layer on the original run (inner CV on training speakers)
        folds = list(GroupKFold(5).split(X[:, 0], y, groups=g)); layers = []
        for tr, te in folds:
            best, bf = 0, -1
            for L in range(X.shape[1]):
                f = []
                for itr, ite in GroupKFold(4).split(X[tr][:, L], y[tr], groups=g[tr]):
                    m = knn().fit(normalize(X[tr][itr][:, L]), y[tr][itr])
                    f.append(f1_score(y[tr][ite], m.predict(normalize(X[tr][ite][:, L])), average="macro"))
                if np.mean(f) > bf: bf, best = np.mean(f), L
            layers.append(best)
        # query sets: orig + seeds, restricted to the 60 IDs (test speakers of each fold)
        q = {"orig": {}, **{f"s{s}": {} for s in SEEDS}}
        for sysname in CLONING:
            m_orig = np.isin(uid, sub_ids) & (y == sysname)
            q["orig"][sysname] = (X[m_orig], uid[m_orig], g[m_orig])
            for s in SEEDS:
                q[f"s{s}"][sysname] = load_seed(ssl, sysname, s, sub_ids)
        preds = {k: [] for k in q}; truth = []; qspk = []; per_done = False
        for (tr, te), L in zip(folds, layers):
            test_spk = set(g[te])
            clf = knn().fit(normalize(X[tr][:, L]), y[tr])
            for sysname in CLONING:
                for k in q:
                    Xq, uq, gq = q[k][sysname]; m = np.isin(gq, list(test_spk))
                    if m.sum() == 0: continue
                    preds[k].append((clf.predict(normalize(Xq[m][:, L])), np.array([sysname] * m.sum()), gq[m]))
        P = {k: (np.concatenate([p[0] for p in preds[k]]), np.concatenate([p[1] for p in preds[k]]), np.concatenate([p[2] for p in preds[k]])) for k in q}
        sps = np.unique(P["orig"][2])
        for k in q:
            pr, yt, gs = P[k]
            f1 = f1_score(yt, pr, average="macro")
            rec = {c: recall_score(yt == c, pr == c) for c in CLONING}
            # paired bootstrap over query speakers: difference vs orig
            diffs = []
            for _ in range(1000):
                s = RNG.choice(sps, len(sps), replace=True)
                i1 = np.concatenate([np.where(gs == x)[0] for x in s]); i0 = np.concatenate([np.where(P["orig"][2] == x)[0] for x in s])
                diffs.append(f1_score(yt[i1], pr[i1], average="macro") - f1_score(P["orig"][1][i0], P["orig"][0][i0], average="macro"))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            out_rows.append({"ssl": ssl, "query": k, "macro_f1": f1, "drop_vs_orig": f1_score(P["orig"][1], P["orig"][0], average="macro") - f1,
                             "drop_ci_lo": -hi, "drop_ci_hi": -lo, "layers": str(layers), **{f"recall_{c}": rec[c] for c in CLONING}})
            print(ssl, k, f"F1 {f1:.3f}", {c: round(v, 2) for c, v in rec.items()}, flush=True)
        # geometry at layer 0 and at the modal chosen layer: same-system cross-seed vs cross-system same-utterance distances
        Lm = int(pd.Series(layers).mode()[0])
        for L in sorted({0, Lm}):
            same, cross = [], []
            for u in sub_ids:
                vec = {}
                for sysname in CLONING:
                    for k in q:
                        Xq, uq, gq = q[k][sysname]; m = uq == u
                        if m.sum(): vec[(sysname, k)] = normalize(Xq[m][:, L])[0]
                for sysname in CLONING:
                    ks = [k for k in q if (sysname, k) in vec]
                    for i in range(len(ks)):
                        for j in range(i + 1, len(ks)):
                            same.append(1 - vec[(sysname, ks[i])] @ vec[(sysname, ks[j])])
                    for other in CLONING:
                        if other == sysname: continue
                        for k in q:
                            if (sysname, k) in vec and (other, k) in vec:
                                cross.append(1 - vec[(sysname, k)] @ vec[(other, k)])
            geo_rows.append({"ssl": ssl, "layer": L, "same_system_cross_seed_dist": float(np.mean(same)),
                             "cross_system_same_utt_dist": float(np.mean(cross)), "ratio": float(np.mean(same) / np.mean(cross))})
            print(ssl, "L", L, "same-system cross-seed", round(np.mean(same), 4), "cross-system", round(np.mean(cross), 4))
        pd.DataFrame(out_rows).to_csv(ROOT / RESULTS / "seeds2_transfer.csv", index=False)
        pd.DataFrame(geo_rows).to_csv(ROOT / RESULTS / "seeds2_geometry.csv", index=False)
    # stochasticity check on waveforms: hash equality and mean abs difference between seeds (per system)
    import soundfile as sf
    st = []
    for sysname in CLONING:
        eq = 0; mad = []; n = 0
        for r in sub:
            wavs = {}
            for s in SEEDS:
                p = ROOT / "data/generated_seeds" / f"{sysname}_s{s}" / f"{r['utt_id']}.wav"
                if p.exists(): wavs[s] = sf.read(str(p))[0]
            ks = sorted(wavs)
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    a, b = wavs[ks[i]], wavs[ks[j]]; n += 1
                    if hashlib.sha256(a.tobytes()).hexdigest() == hashlib.sha256(b.tobytes()).hexdigest(): eq += 1
                    m = min(len(a), len(b)); mad.append(float(np.mean(np.abs(a[:m] - b[:m]))) if m else np.nan)
        st.append({"system": sysname, "pairs": n, "identical_pairs": eq, "mean_abs_sample_diff": float(np.nanmean(mad)) if mad else np.nan,
                   "length_varies": bool(n and len({len(v) for r in sub for s in SEEDS for v in [sf.read(str(ROOT / 'data/generated_seeds' / f'{sysname}_s{s}' / f"{r['utt_id']}.wav"))[0]] if (ROOT / 'data/generated_seeds' / f'{sysname}_s{s}' / f"{r['utt_id']}.wav").exists()}) > 1)})
        print(st[-1], flush=True)
    pd.DataFrame(st).to_csv(ROOT / RESULTS / "seeds2_stochasticity.csv", index=False)
    print("saved seeds2_* in", ROOT / RESULTS)


if __name__ == "__main__":
    main()
