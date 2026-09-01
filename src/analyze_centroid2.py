"""Centroid analysis v2 (audit fix): per-probe paired cosine-distance shift to
EVERY TTS-class centroid (speaker-excluded), for matched AND unmatched probes.
Reports absolute target shift, the target-specific CONTRAST (target shift minus
mean shift to the other TTS centroids), shifts of unmatched controls toward
each class, and a matched-vs-unmatched contrast with speaker bootstrap CIs.
Output: results/paper/centroid2_<ssl>_L<layer>.csv
"""
import argparse
import os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import normalize

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = os.environ.get("TTS_ANAL_RESULTS", "results/paper5c17")
SYSTEMS = os.environ.get("TTS_ANAL_SYSTEMS", "f5tts,xtts,cosyvoice3,chatterbox,indextts").split(",")
TTS = SYSTEMS
PROBES = ["resynth_vocos", "resynth_hift", "resynth_s3vc", "resynth_bigvgan",
          "resynth_glvocos", "resynth_griffinlim", "resynth_encodec", "resynth_dac"]
TARGET = {"resynth_vocos": "f5tts", "resynth_hift": "cosyvoice2",
          "resynth_s3vc": "cosyvoice2", "resynth_bigvgan": "indextts",
          "resynth_glvocos": "f5tts", "resynth_griffinlim": "f5tts"}
# probes for optional newer systems (only active when the target system is in TTS_ANAL_SYSTEMS)
_EXTRA_TARGET = {"resynth_hift3": "cosyvoice3", "resynth_s3vc3": "cosyvoice3", "resynth_qwencodec": "qwen3tts"}
TARGET.update({p: t for p, t in _EXTRA_TARGET.items() if t in SYSTEMS})
TARGET = {p: t for p, t in TARGET.items() if t in SYSTEMS}
PROBES = [p for p in PROBES if p in TARGET or p in ("resynth_encodec", "resynth_dac")] + [p for p in _EXTRA_TARGET if p in TARGET]
EXCLUDE = set((ROOT / os.environ.get("TTS_ANAL_EXCLUDE", "data/manifests/exclude17.txt")).read_text().split())
RNG = np.random.default_rng(0)

def load(ssl, cond, layer):
    d = np.load(ROOT / "results/embeddings" / ssl / f"{cond}.npz")
    keep = ~np.isin(d["utt_ids"], list(EXCLUDE))
    return normalize(d["embs"][keep][:, layer]), d["utt_ids"][keep], d["speakers"][keep]

def dist_to_centroid(X, sq, Tx, Ts):
    out = np.empty(len(X))
    for sp in np.unique(sq):
        cen = Tx[Ts != sp].mean(0); cen /= np.linalg.norm(cen)
        m = sq == sp; out[m] = 1 - X[m] @ cen
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ssl", default="wavlm"); ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--nboot-min", type=int, default=10000,
                    help="bootstrap replicates for the strongest-control G_min rows (the Vocos "
                         "lower bound sits near zero, so these use more replicates than the default 1,000)")
    a = ap.parse_args()
    real_X, real_u, real_s = load(a.ssl, "real", a.layer)
    cls = {c: load(a.ssl, c, a.layer) for c in TTS}
    order = {u: i for i, u in enumerate(real_u)}
    # per-probe, per-class paired shift arrays
    shifts = {}  # probe -> class -> array
    spk = {}
    for p in PROBES:
        try: P, pu, ps = load(a.ssl, p, a.layer)
        except FileNotFoundError: continue
        spk[p] = ps; shifts[p] = {}
        for c in TTS:
            Tx, _, Ts = cls[c]
            dp = dist_to_centroid(P, ps, Tx, Ts)
            dr = dist_to_centroid(real_X, real_s, Tx, Ts)
            shifts[p][c] = dp - np.array([dr[order[u]] for u in pu])
    def boot(fn, n=1000):
        sps = np.unique(np.concatenate([spk[p] for p in shifts]))
        vals = []
        for _ in range(n):
            s = RNG.choice(sps, len(sps), replace=True)
            vals.append(fn(s))
        return np.percentile(vals, [2.5, 97.5])
    def sel(arr, ps, s):
        return np.concatenate([arr[ps == x] for x in s])
    rows = []
    for p in shifts:
        row = {"probe": p, "target": TARGET.get(p, "")}
        for c in TTS: row[f"shift_{c}"] = shifts[p][c].mean()
        tgt = TARGET.get(p)
        if tgt:
            others = [c for c in TTS if c != tgt]
            contrast = shifts[p][tgt] - np.mean([shifts[p][c] for c in others], axis=0)
            row["contrast"] = contrast.mean()
            row["contrast_lo"], row["contrast_hi"] = boot(lambda s: sel(contrast, spk[p], s).mean())
            # positive-is-stronger conventions used in the paper:
            #   T = -shift_target  (translation toward the target centroid)
            #   G = -contrast      (relative alignment: mean shift to other TTS centroids minus target shift)
            row["T"] = -shifts[p][tgt].mean()
            row["T_lo"], row["T_hi"] = sorted(-np.array(boot(lambda s: sel(shifts[p][tgt], spk[p], s).mean())))
            row["G"], row["G_lo"], row["G_hi"] = -row["contrast"], -row["contrast_hi"], -row["contrast_lo"]
        else:
            row["max_abs_shift_class"] = min(TTS, key=lambda c: shifts[p][c].mean())
            row["min_shift"] = min(shifts[p][c].mean() for c in TTS)
            row["mean_shift"] = np.mean([shifts[p][c].mean() for c in TTS])
        rows.append(row)
    # matched-vs-unmatched, two estimands (both reported, named explicitly):
    #  (i) matched_minus_unmatched_target_shift: matched ABSOLUTE target shift
    #      minus unmatched probes' absolute shift toward that same target
    #  (ii) contrast_of_contrasts: matched target-specific contrast minus the
    #      mean unmatched target-specific contrast toward the same target
    unm = [p for p in ["resynth_encodec", "resynth_dac"] if p in shifts]
    def contrast_to(p, tgt):
        others = [c for c in TTS if c != tgt]
        return shifts[p][tgt] - np.mean([shifts[p][c] for c in others], axis=0)
    for p in [q for q in ["resynth_vocos", "resynth_hift", "resynth_s3vc"] if q in TARGET] + [q for q in _EXTRA_TARGET if q in TARGET]:
        if p not in shifts: continue
        tgt = TARGET[p]
        def stat_abs(s):
            m = sel(shifts[p][tgt], spk[p], s).mean()
            u = np.mean([sel(shifts[q][tgt], spk[q], s).mean() for q in unm])
            return m - u
        def stat_cc(s):
            m = sel(contrast_to(p, tgt), spk[p], s).mean()
            u = np.mean([sel(contrast_to(q, tgt), spk[q], s).mean() for q in unm])
            return m - u
        full = np.unique(spk[p])
        lo, hi = boot(stat_abs); lo2, hi2 = boot(stat_cc)
        rows.append({"probe": f"{p}_vs_unmatched", "target": tgt,
                     "matched_minus_unmatched_target_shift": stat_abs(full),
                     "mmu_lo": lo, "mmu_hi": hi,
                     "contrast_of_contrasts": stat_cc(full),
                     "coc_lo": lo2, "coc_hi": hi2,
                     "G_adj": -stat_cc(full), "G_adj_lo": -hi2, "G_adj_hi": -lo2})
        # conservative strongest-control statistic (2026-08-24 amendment, see analysis_plan.md):
        # G_min = G_matched - max over the full off-target neural control set
        # {EnCodec, DAC, BigVGAN-v2} of G_u, with the max recomputed inside every
        # speaker-bootstrap replicate. G_adj (pre-specified, codecs only) is unchanged.
        strong = [q for q in ["resynth_encodec", "resynth_dac", "resynth_bigvgan"]
                  if q in shifts and TARGET.get(q) != tgt]
        def stat_min(s):
            gm = -sel(contrast_to(p, tgt), spk[p], s).mean()
            gu = max(-sel(contrast_to(q, tgt), spk[q], s).mean() for q in strong)
            return gm - gu
        lo3, hi3 = boot(stat_min, n=a.nboot_min)
        rows.append({"probe": f"{p}_vs_strongest", "target": tgt,
                     "controls": "+".join(q.replace("resynth_", "") for q in strong),
                     "n_boot": a.nboot_min,
                     "G_min": stat_min(full), "G_min_lo": lo3, "G_min_hi": hi3})
    # unmatched/off-target controls evaluated under each architecture-derived target (same
    # target as the matched probe they are compared with): T and G with speaker-bootstrap CIs
    for q in unm + [x for x in ["resynth_bigvgan"] if x in shifts]:
        for tgt in sorted(set(v for v in TARGET.values() if v)):
            if TARGET.get(q) == tgt: continue
            ct = contrast_to(q, tgt)
            tlo, thi = sorted(-np.array(boot(lambda s: sel(shifts[q][tgt], spk[q], s).mean())))
            glo, ghi = sorted(-np.array(boot(lambda s: sel(ct, spk[q], s).mean())))
            rows.append({"probe": f"{q}_under_{tgt}", "target": tgt,
                         "T": -shifts[q][tgt].mean(), "T_lo": tlo, "T_hi": thi,
                         "G": -ct.mean(), "G_lo": glo, "G_hi": ghi})
    df = pd.DataFrame(rows)
    out = ROOT / RESULTS / f"centroid2_{a.ssl}_L{a.layer}.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250); print(df.round(4).to_string(index=False)); print("saved", out)

if __name__ == "__main__":
    main()
