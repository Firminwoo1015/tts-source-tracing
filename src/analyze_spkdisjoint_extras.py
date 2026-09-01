"""Audit fixes: (a) nested speaker-disjoint 6-way F1 for common-channel tags;
(b) speaker-disjoint pairwise separability (GroupKFold acc per layer 0-4);
(b2) in-domain leave-one-out pairwise separability (same pairs/layers; the
     "uniquely closest pair" reading in the paper is in-domain only);
(c) alternative deep bands (13-24, 20-24) under nested speaker-disjoint selection.
Outputs: results/paper/common_spkdisjoint.csv, pairwise_spkdisjoint.csv,
         pairwise_indomain.csv, bands_alt.csv
"""
import sys
import os
from pathlib import Path
from itertools import combinations
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
sys.path.insert(0, str(Path(__file__).parent))
from analyze_cloning import nested_spk_disjoint, load_all as load_cloning, CLONING
from analyze_bands import band_pred, load_all as load_tts, SSLS

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = os.environ.get("TTS_ANAL_RESULTS", "results/paper5c17")
EXCLUDE = set((ROOT / os.environ.get("TTS_ANAL_EXCLUDE", "data/manifests/exclude17.txt")).read_text().split())
CONDS = ["real"] + CLONING
RNG = np.random.default_rng(0)

def knn(): return KNeighborsClassifier(5, metric="cosine")

def _pair_embs():
    E = {}
    for c in CLONING:
        d = np.load(ROOT / f"results/embeddings/wavlm/{c}.npz")
        keep = ~np.isin(d["utt_ids"], list(EXCLUDE))
        E[c] = (d["embs"][keep], d["speakers"][keep])
    return E


def pairwise_indomain():
    """(b2) in-domain LOO pairwise separability, WavLM layers 0-4."""
    E = _pair_embs(); prow = []
    for a, b in combinations(CLONING, 2):
        accs = []
        for L in range(5):
            X = normalize(np.concatenate([E[a][0][:, L], E[b][0][:, L]]))
            y = np.array([0] * len(E[a][0]) + [1] * len(E[b][0]))
            pred = cross_val_predict(knn(), X, y, cv=LeaveOneOut(), n_jobs=8)
            accs.append((pred == y).mean())
        prow.append({"pair": f"{a}|{b}", **{f"L{i}": v for i, v in enumerate(accs)}, "min": min(accs), "max": max(accs)})
        print("indomain", prow[-1]["pair"], [round(v, 3) for v in accs])
    pd.DataFrame(prow).to_csv(ROOT / RESULTS / "pairwise_indomain.csv", index=False)


def main():
    # (a) common-channel, nested spk-disjoint
    rows = []
    for ssl in ["wavlm", "w2vbert"]:
        for tag, name in [(ssl, "clean"), (f"{ssl}_p_common", "common"), (f"{ssl}_p_common_sym", "common_sym")]:
            embs, cond, spk, uid = load_cloning(tag, CONDS)
            pred, chosen = nested_spk_disjoint(embs, cond, spk)
            f1 = f1_score(cond, pred, average="macro")
            sps = np.unique(spk); boots = []
            for _ in range(1000):
                s = RNG.choice(sps, len(sps), replace=True)
                idx = np.concatenate([np.where(spk == x)[0] for x in s])
                boots.append(f1_score(cond[idx], pred[idx], average="macro"))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"ssl": ssl, "channel": name, "macro_f1": f1, "ci_lo": lo, "ci_hi": hi, "layers": str(chosen)})
            print(rows[-1])
    pd.DataFrame(rows).to_csv(ROOT / RESULTS / "common_spkdisjoint.csv", index=False)

    # (b) speaker-disjoint pairwise (wavlm, layers 0-4)
    E = {}
    for c in CLONING:
        d = np.load(ROOT / f"results/embeddings/wavlm/{c}.npz")
        keep = ~np.isin(d["utt_ids"], list(EXCLUDE))
        E[c] = (d["embs"][keep], d["speakers"][keep])
    prow = []
    for a, b in combinations(CLONING, 2):
        accs = []
        for L in range(5):
            X = normalize(np.concatenate([E[a][0][:, L], E[b][0][:, L]]))
            y = np.array([0] * len(E[a][0]) + [1] * len(E[b][0]))
            g = np.concatenate([E[a][1], E[b][1]])
            pred = cross_val_predict(knn(), X, y, cv=GroupKFold(5), groups=g, n_jobs=8)
            accs.append((pred == y).mean())
        prow.append({"pair": f"{a}|{b}", **{f"L{i}": v for i, v in enumerate(accs)}, "min": min(accs), "max": max(accs)})
        print(prow[-1]["pair"], [round(v, 3) for v in accs])
    pd.DataFrame(prow).to_csv(ROOT / RESULTS / "pairwise_spkdisjoint.csv", index=False)
    pairwise_indomain()

    # (c) alternative bands, nested spk-disjoint
    brow = []
    for ssl in SSLS:
        embs, y, g = load_tts(ssl)
        for bname, band in [("17-21", range(17, 22)), ("20-24", range(20, 25)), ("13-24", range(13, 25))]:
            pred = band_pred(embs, y, g, list(band))
            brow.append({"ssl": ssl, "band": bname, "macro_f1": f1_score(y, pred, average="macro")})
        print(ssl, [round(r["macro_f1"], 3) for r in brow if r["ssl"] == ssl])
    pd.DataFrame(brow).to_csv(ROOT / RESULTS / "bands_alt.csv", index=False)

if __name__ == "__main__":
    main()
