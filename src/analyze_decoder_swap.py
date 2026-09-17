"""Within-system decoder swap on the same mel (analysis_plan.md, N4; §4.2 "Same-mel decoder swap").

For each (native path, same-mel Griffin-Lim path, target), paired by utterance ID:
  D_G  = G_t(native)  - G_t(GL)      primary
  D_dP = dP_t(native) - dP_t(GL)     = P(pred=t | native) - P(pred=t | GL); the clean-real baseline cancels
  D_T  = T_t(native)  - T_t(GL)
with 95% speaker-paired bootstrap CIs (10,000 replicates). G_t, T_t and the kNN predictions use exactly the
rules of analyze_centroid2.py and analyze_transplant2.py (query-speaker-excluded support, centroids from the
other speakers, per-utterance change relative to the same clean real utterance).
Output: <results>/decoder_swap_<ssl>_L<layer>.csv
"""
import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = os.environ.get("TTS_ANAL_RESULTS", "results/final")
PAIRS = [  # (native, gl, target, role)
    ("resynth_vocos", "resynth_glvocos", "f5tts", "primary"),
    ("resynth_hift3", "resynth_glc3", "cosyvoice3", "primary"),
    ("resynth_hiftcb", "resynth_glcb", "chatterbox", "primary"),
    ("resynth_s3vc3", "resynth_glc3", "cosyvoice3", "descriptive (token round trip vs same-mel GL)"),
    ("resynth_s3vccb", "resynth_glcb", "chatterbox", "descriptive (token round trip vs same-mel GL)"),
]


def _module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssl", default="wavlm"); ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--nboot", type=int, default=10000)
    a = ap.parse_args()
    T2, C2 = _module("analyze_transplant2"), _module("analyze_centroid2")
    rng = np.random.default_rng(20260917)

    # kNN predictions per utterance (same reference set and exclusion key as analyze_transplant2)
    refs = {c: T2.load(a.ssl, c) for c in T2.REFS}
    Xref = np.concatenate([r[0][:, a.layer] for r in refs.values()])
    yref = np.concatenate([[c] * len(r[0]) for c, r in refs.items()])
    eref = np.concatenate([r[2] for r in refs.values()])
    # per-utterance centroid shifts (same rule as analyze_centroid2)
    real_X, real_u, real_s = C2.load(a.ssl, "real", a.layer)
    cls = {c: C2.load(a.ssl, c, a.layer) for c in C2.TTS}
    order = {u: i for i, u in enumerate(real_u)}
    dr = {c: C2.dist_to_centroid(real_X, real_s, cls[c][0], cls[c][2]) for c in C2.TTS}

    cache = {}
    def per_utt(p, tgt):
        if (p, tgt) in cache: return cache[(p, tgt)]
        Xp, up, sp = T2.load(a.ssl, p)
        pred = T2.knn_predict_excl(Xref, yref, eref, Xp[:, a.layer], sp)
        P, pu, ps = C2.load(a.ssl, p, a.layer)
        assert (pu == up).all() and (ps == sp).all()
        sh = {c: C2.dist_to_centroid(P, ps, cls[c][0], cls[c][2]) - np.array([dr[c][order[u]] for u in pu]) for c in C2.TTS}
        others = [c for c in C2.TTS if c != tgt]
        df = pd.DataFrame({"utt": pu, "spk": ps, "hit": (pred == tgt).astype(float),
                           "T": -sh[tgt], "G": -(sh[tgt] - np.mean([sh[c] for c in others], axis=0))})
        cache[(p, tgt)] = df
        return df

    rows = []
    for native, gl, tgt, role in PAIRS:
        n, g = per_utt(native, tgt), per_utt(gl, tgt)
        m = n.merge(g, on=["utt", "spk"], suffixes=("_n", "_g"))
        assert len(m) == len(n) == len(g), (native, gl, len(m), len(n), len(g))
        d = {"dP": m.hit_n - m.hit_g, "T": m.T_n - m.T_g, "G": m.G_n - m.G_g}
        spk = m.spk.values; speakers = np.unique(spk)
        idx = {s: np.where(spk == s)[0] for s in speakers}
        boots = {k: np.empty(a.nboot) for k in d}
        arr = {k: v.values for k, v in d.items()}
        for b in range(a.nboot):
            sel = np.concatenate([idx[s] for s in rng.choice(speakers, len(speakers), replace=True)])
            for k in d: boots[k][b] = arr[k][sel].mean()
        row = {"native": native, "gl": gl, "target": tgt, "role": role, "n_utt": len(m), "n_spk": len(speakers),
               "G_native": m.G_n.mean(), "G_gl": m.G_g.mean(), "T_native": m.T_n.mean(), "T_gl": m.T_g.mean(),
               "P_native": m.hit_n.mean(), "P_gl": m.hit_g.mean()}
        for k in ["G", "dP", "T"]:
            lo, hi = np.percentile(boots[k], [2.5, 97.5])
            row[f"D_{k}"], row[f"D_{k}_lo"], row[f"D_{k}_hi"] = arr[k].mean(), lo, hi
        rows.append(row)
    # exploratory cross-mel comparison (not pre-specified): every Griffin-Lim path scored toward F5-TTS
    rng2 = np.random.default_rng(1); xr = []
    for gl in ["resynth_glvocos", "resynth_griffinlim", "resynth_glc3", "resynth_glcb"]:
        try: g = per_utt(gl, "f5tts")
        except FileNotFoundError: continue
        spk = g.spk.values; speakers = np.unique(spk); idx = {s_: np.where(spk == s_)[0] for s_ in speakers}
        bt = {k: [] for k in ["hit", "T", "G"]}
        for _ in range(1000):
            sel = np.concatenate([idx[s_] for s_ in rng2.choice(speakers, len(speakers), replace=True)])
            for k in bt: bt[k].append(g[k].values[sel].mean())
        r_ = {"gl": gl, "scored_toward": "f5tts"}
        for k, nm in [("hit", "P_f5"), ("T", "T_f5"), ("G", "G_f5")]:
            lo, hi = np.percentile(bt[k], [2.5, 97.5]); r_[nm], r_[nm + "_lo"], r_[nm + "_hi"] = g[k].mean(), lo, hi
        xr.append(r_)
    pd.DataFrame(xr).to_csv(ROOT / RESULTS / f"gl_crossmel_{a.ssl}_L{a.layer}.csv", index=False)
    out = pd.DataFrame(rows)
    path = ROOT / RESULTS / f"decoder_swap_{a.ssl}_L{a.layer}.csv"
    path.parent.mkdir(parents=True, exist_ok=True); out.to_csv(path, index=False)
    pd.set_option("display.width", 250); print(out.round(5).to_string(index=False)); print("saved", path)


if __name__ == "__main__":
    main()
