"""A2. Paired-reference kNN mechanism (see analysis_plan.md).

Fixed layers (default WavLM/HuBERT L17/L19/L21), TTS-only 5-way. For each protocol
(ordinary LOO, leave-utterance-out, leave-speaker-out), k in {1,3,5,7,11} and voting
(uniform, inverse-distance), report macro-F1; plus mechanism statistics under ordinary LOO:
fraction of the top-k neighbours that share the query's utterance ID (other systems) and the
uniform-vote tie rate. Output: <RESULTS>/knn_mechanism.csv
"""
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
sys.path.insert(0, str(Path(__file__).parent))
from analyze_cloning import load_all, CLONING

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = os.environ.get("TTS_ANAL_RESULTS", "results/paper5c17")


def predict(S, y, uid, spk, protocol, k, weighting):
    n = len(y); preds = np.empty(n, dtype=object); same_uid_frac = np.zeros(n); ties = np.zeros(n, bool)
    classes = np.unique(y)
    for i in range(n):
        s = S[i].copy(); s[i] = -np.inf
        if protocol == "leave_utt":
            s[uid == uid[i]] = -np.inf
        elif protocol == "leave_spk":
            s[spk == spk[i]] = -np.inf
        top = np.argpartition(-s, k)[:k]
        same_uid_frac[i] = np.mean(uid[top] == uid[i])
        w = np.ones(k) if weighting == "uniform" else 1.0 / np.maximum(1 - s[top], 1e-6)
        votes = {c: w[y[top] == c].sum() for c in classes}
        best = max(votes.values()); winners = [c for c, v in votes.items() if v == best]
        ties[i] = len(winners) > 1
        preds[i] = winners[0] if weighting == "uniform" else max(votes, key=votes.get)
    return preds.astype(str), same_uid_frac, ties


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssls", nargs="+", default=["wavlm", "hubert"])
    ap.add_argument("--layers", nargs="+", type=int, default=[17, 19, 21])
    a = ap.parse_args()
    out = ROOT / RESULTS / "knn_mechanism.csv"
    if out.exists() and not os.environ.get("TTS_ANAL_FORCE") and len(pd.read_csv(out)) >= len(a.ssls) * len(a.layers) * 3 * 5 * 2:
        print(f"skip: {out} complete; set TTS_ANAL_FORCE=1 to recompute"); return
    rows = []
    for ssl in a.ssls:
        embs, y, spk, uid = load_all(ssl, CLONING)
        for L in a.layers:
            X = normalize(embs[:, L]); S = X @ X.T
            for protocol in ["loo", "leave_utt", "leave_spk"]:
                for k in [1, 3, 5, 7, 11]:
                    for weighting in ["uniform", "distance"]:
                        pred, same, ties = predict(S, y, uid, spk, protocol, k, weighting)
                        rows.append({"ssl": ssl, "layer": L, "protocol": protocol, "k": k, "weighting": weighting,
                                     "macro_f1": f1_score(y, pred, average="macro"),
                                     "same_utt_frac_topk": float(same.mean()), "tie_rate": float(ties.mean())})
            print(ssl, L, [f"{r['protocol']}:k{r['k']}:{r['weighting'][:3]}={r['macro_f1']:.2f}" for r in rows[-30:] if r['k'] == 5], flush=True)
    df = pd.DataFrame(rows); df.to_csv(ROOT / RESULTS / "knn_mechanism.csv", index=False)
    print("saved", ROOT / RESULTS / "knn_mechanism.csv")


if __name__ == "__main__":
    main()
