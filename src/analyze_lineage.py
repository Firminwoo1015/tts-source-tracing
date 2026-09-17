"""Sibling-system overlap between the CosyVoice3 and Chatterbox matched paths (§4.2 "Related lineages").

Chatterbox's token-to-waveform stack derives from CosyVoice2 and shares the HiFT decoder family with
CosyVoice3. For each HiFT path and token round trip of the two systems, report the prediction-rate
shift and the relative alignment G toward the path's own target and toward the sibling system, and
the paired own-minus-sibling difference, with speaker-bootstrap CIs. The kNN rule, exclusion key,
centroid estimate and bootstrap unit are those of analyze_transplant2.py and analyze_centroid2.py,
so the point estimates equal theirs (own-minus-sibling prediction shift = S_t whenever the sibling is
the strongest competing class).

Output: results/final/lineage_<ssl>_L<layer>.csv
"""
import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = os.environ.get("TTS_ANAL_RESULTS", "results/final")
PATHS = {"resynth_hift3": "cosyvoice3", "resynth_s3vc3": "cosyvoice3",
         "resynth_hiftcb": "chatterbox", "resynth_s3vccb": "chatterbox"}
SIB = {"cosyvoice3": "chatterbox", "chatterbox": "cosyvoice3"}


def _module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssl", default="wavlm"); ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--nboot", type=int, default=1000)
    a = ap.parse_args()
    T2, C2 = _module("analyze_transplant2"), _module("analyze_centroid2")
    rng = np.random.default_rng(0)
    ci = lambda v: np.percentile(v, [2.5, 97.5])

    refs = {c: T2.load(a.ssl, c) for c in T2.REFS}
    Xref = np.concatenate([r[0][:, a.layer] for r in refs.values()])
    yref = np.concatenate([[c] * len(r[0]) for c, r in refs.items()])
    eref = np.concatenate([r[2] for r in refs.values()])
    Xr, _, sr_ = refs["real"]
    base = T2.knn_predict_excl(Xref, yref, eref, Xr[:, a.layer], sr_)

    real_X, real_u, real_s = C2.load(a.ssl, "real", a.layer)
    cls = {c: C2.load(a.ssl, c, a.layer) for c in C2.TTS}
    order = {u: i for i, u in enumerate(real_u)}

    rows = []
    for p, tgt in PATHS.items():
        sib = SIB[tgt]
        # prediction-rate shifts
        Xp, _, sp = T2.load(a.ssl, p)
        pred = T2.knn_predict_excl(Xref, yref, eref, Xp[:, a.layer], sp)
        dP = lambda pp, bb, c: (pp == c).mean() - (bb == c).mean()
        # centroid geometry
        P, pu, ps = C2.load(a.ssl, p, a.layer)
        assert (ps == sp).all()
        sh = {}
        for c in C2.TTS:
            Tx, _, Ts = cls[c]
            dr = C2.dist_to_centroid(real_X, real_s, Tx, Ts)
            sh[c] = C2.dist_to_centroid(P, ps, Tx, Ts) - np.array([dr[order[u]] for u in pu])
        G = lambda c, idx: -(sh[c][idx] - np.mean([sh[o][idx] for o in C2.TTS if o != c], axis=0)).mean()
        spk = np.unique(sp); allq = np.arange(len(sp))
        bP = {"own": [], "sib": [], "diff": []}; bG = {"own": [], "sib": [], "diff": []}
        for _ in range(a.nboot):
            s = rng.choice(spk, len(spk), replace=True)
            qi = np.concatenate([np.where(sp == x)[0] for x in s]); bi = np.concatenate([np.where(sr_ == x)[0] for x in s])
            po, pz = dP(pred[qi], base[bi], tgt), dP(pred[qi], base[bi], sib)
            go, gz = G(tgt, qi), G(sib, qi)
            for dct, o, z in [(bP, po, pz), (bG, go, gz)]:
                dct["own"].append(o); dct["sib"].append(z); dct["diff"].append(o - z)
        row = {"probe": p, "target": tgt, "sibling": sib}
        for name, dct, full in [("dP", bP, (dP(pred, base, tgt), dP(pred, base, sib))),
                                ("G", bG, (G(tgt, allq), G(sib, allq)))]:
            vals = {"own": full[0], "sib": full[1], "diff": full[0] - full[1]}
            for k in ["own", "sib", "diff"]:
                row[f"{name}_{k}"] = vals[k]
                row[f"{name}_{k}_lo"], row[f"{name}_{k}_hi"] = ci(dct[k])
        rows.append(row)
    df = pd.DataFrame(rows)
    out = ROOT / RESULTS / f"lineage_{a.ssl}_L{a.layer}.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250); print(df.round(4).to_string(index=False)); print("saved", out)


if __name__ == "__main__":
    main()
