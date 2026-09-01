"""Analysis-synthesis interventions v2 (reviewer-hardened).

Fixes vs v1:
  - reference = cloning-only space (real + 5 cloning systems; no Kokoro)
  - utterance-ID exclusion: a probe never sees ANY reference sample sharing its
    utterance ID (removes paired-recording and same-text leakage)
  - clean-real baseline probe classified under the same exclusion rule
  - Delta_target = P(pred=target | intervention) - P(pred=target | clean real),
    with speaker-clustered bootstrap CIs
Probes and their lineage-matched targets:
  resynth_vocos      -> f5tts      (Vocos IS F5-TTS's decoder)
  resynth_hift       -> cosyvoice2 (HiFT IS CosyVoice2's decoder; decoder-only)
  resynth_s3vc       -> cosyvoice2 (S3 tokens+flow+HiFT round trip)
  resynth_bigvgan    -> indextts   (BigVGAN2 lineage)
  resynth_encodec    -> none       (codec used by no system here)
  resynth_dac        -> none       (codec used by no system here)
  resynth_griffinlim -> none       (mel bottleneck, no neural decoder)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = os.environ.get("TTS_ANAL_RESULTS", "results/paper5c17")
SYSTEMS = os.environ.get("TTS_ANAL_SYSTEMS", "f5tts,xtts,cosyvoice3,chatterbox,indextts").split(",")
REFS = ["real"] + SYSTEMS
TARGET = {"resynth_vocos": "f5tts", "resynth_hift": "cosyvoice2",
          "resynth_s3vc": "cosyvoice2", "resynth_bigvgan": "indextts",
          # mel-bottleneck probes: griffinlim = generic librosa mel at F5's
          # frame config; glvocos = Vocos's exact analysis inverted by
          # Griffin-Lim (the true F5 mel-only ablation). Architecture-derived
          # hypothesized target for both is f5tts; EnCodec/DAC are unmatched.
          "resynth_griffinlim": "f5tts", "resynth_glvocos": "f5tts",
          "resynth_encodec": None, "resynth_dac": None}
# probes for optional newer systems (only active when the target system is in TTS_ANAL_SYSTEMS)
_EXTRA_TARGET = {"resynth_hift3": "cosyvoice3", "resynth_s3vc3": "cosyvoice3", "resynth_qwencodec": "qwen3tts"}
TARGET.update({p: t for p, t in _EXTRA_TARGET.items() if t in SYSTEMS})
# keep only probes whose hypothesized target is present (unmatched probes keep target None)
TARGET = {p: t for p, t in TARGET.items() if t is None or t in SYSTEMS}
EXCLUDE = set((ROOT / os.environ.get("TTS_ANAL_EXCLUDE", "data/manifests/exclude17.txt")).read_text().split())
RNG = np.random.default_rng(0)


def load(ssl, cond):
    d = np.load(ROOT / "results/embeddings" / ssl / f"{cond}.npz")
    keep = ~np.isin(d["utt_ids"], list(EXCLUDE))
    return d["embs"][keep], d["utt_ids"][keep], d["speakers"][keep]


def knn_predict_excl(Xref, yref, eref, Xq, eq, k=5):
    """kNN with cosine distance, excluding reference rows whose exclusion key
    matches the query's. With speaker keys this makes every probe
    speaker-disjoint from its reference (and, since LibriSpeech utterance IDs
    are speaker-specific, also utterance- and text-disjoint)."""
    sims = normalize(Xq) @ normalize(Xref).T
    preds = []
    for i in range(len(Xq)):
        mask = eref == eq[i]
        s = sims[i].copy()
        s[mask] = -np.inf
        top = np.argpartition(-s, k)[:k]
        vals, counts = np.unique(yref[top], return_counts=True)
        preds.append(vals[np.argmax(counts)])
    return np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssl", default="wavlm")
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 2, 4, 8, 19])
    args = ap.parse_args()

    refs = {c: load(args.ssl, c) for c in REFS}
    probes = {}
    for c in TARGET:
        try:
            probes[c] = load(args.ssl, c)
        except FileNotFoundError:
            print(f"skip missing {c}")

    records = []
    for layer in args.layers:
        Xref = np.concatenate([r[0][:, layer] for r in refs.values()])
        yref = np.concatenate([[c] * len(r[0]) for c, r in refs.items()])
        # speaker-disjoint exclusion key (subsumes utterance/text exclusion)
        eref = np.concatenate([r[2] for r in refs.values()])

        # clean-real baseline probe under identical speaker-disjoint exclusion
        Xr, ur, sr_ = refs["real"]
        base_pred = knn_predict_excl(Xref, yref, eref, Xr[:, layer], sr_)
        base_frac = {c: float((base_pred == c).mean()) for c in REFS}
        records.append({"layer": layer, "probe": "clean_real", **base_frac})

        for probe, (Xp, up, sp) in probes.items():
            pred = knn_predict_excl(Xref, yref, eref, Xp[:, layer], sp)
            frac = {c: float((pred == c).mean()) for c in REFS}
            # per-class shift vs the clean-real baseline
            dP = {f"dP_{c}": frac[c] - base_frac[c] for c in REFS}
            tts = [c for c in REFS if c != "real"]
            row = {"layer": layer, "probe": probe, **frac, **dP}
            # unified concentration margin for ALL probes: largest TTS-class
            # shift minus the second largest (diffuse probes -> small margin)
            shifts = sorted((dP[f"dP_{c}"] for c in tts), reverse=True)
            row["margin"] = shifts[0] - shifts[1]
            top_class = max(tts, key=lambda c: dP[f"dP_{c}"])
            row["top_class"] = top_class
            tgt = TARGET[probe]
            row["target_aligned"] = bool(tgt) and (top_class == tgt)

            speakers = np.unique(sp)

            def boot_stat(stat_fn, n=1000):
                vals = []
                for _ in range(n):
                    s = RNG.choice(speakers, len(speakers), replace=True)
                    qi = np.concatenate([np.where(sp == x)[0] for x in s])
                    bi = np.concatenate([np.where(sr_ == x)[0] for x in s])
                    vals.append(stat_fn(pred[qi], base_pred[bi]))
                return np.percentile(vals, [2.5, 97.5])

            def margin_stat(p, b):
                d = {c: (p == c).mean() - (b == c).mean() for c in tts}
                sh = sorted(d.values(), reverse=True)
                return sh[0] - sh[1]

            row["margin_ci_lo"], row["margin_ci_hi"] = boot_stat(margin_stat)
            # signed target margin S_t = dP_t - max_{j in TTS \ {t}} dP_j, for every
            # architecture-derived target (matched probe vs unmatched controls under the
            # same target); the probe's own hypothesized target gets a speaker-bootstrap CI
            def signed_margin(p, b, t):
                d = {c: (p == c).mean() - (b == c).mean() for c in tts}
                return d[t] - max(v for c, v in d.items() if c != t)
            for t in sorted(set(v for v in TARGET.values() if v)):
                row[f"S_{t}"] = signed_margin(pred, base_pred, t)
            # speaker-bootstrap CIs for the control probes under the fixed matched-probe
            # targets (EnCodec/DAC/BigVGAN rows of Table 2 under the F5 and CosyVoice3 targets)
            if probe in ("resynth_encodec", "resynth_dac", "resynth_bigvgan"):
                for t in [x for x in ("f5tts", "cosyvoice3") if x in SYSTEMS and x != tgt]:
                    row[f"dP_{t}_lo"], row[f"dP_{t}_hi"] = boot_stat(
                        lambda p, b, t=t: (p == t).mean() - (b == t).mean())
                    row[f"S_{t}_lo"], row[f"S_{t}_hi"] = boot_stat(
                        lambda p, b, t=t: signed_margin(p, b, t))
            if tgt:
                row["delta_target"] = dP[f"dP_{tgt}"]
                row["delta_ci_lo"], row["delta_ci_hi"] = boot_stat(
                    lambda p, b: (p == tgt).mean() - (b == tgt).mean())
                row["S_target"] = row[f"S_{tgt}"]
                row["S_ci_lo"], row["S_ci_hi"] = boot_stat(lambda p, b: signed_margin(p, b, tgt))
            records.append(row)

    df = pd.DataFrame(records)
    out = ROOT / RESULTS / f"intervention_{args.ssl}.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
