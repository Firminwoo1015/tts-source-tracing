"""MFCC baseline on the cloning-only set (persisted script for the paper's
Table 2 numbers; the old mfcc_baseline.csv was the deprecated Kokoro-era
5-way run). Speaker-disjoint GroupKFold, cosine kNN, speaker bootstrap.
Output: results/paper/mfcc_cloning.csv
"""

import json
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = os.environ.get("TTS_ANAL_RESULTS", "results/paper5c17")
SYSTEMS = os.environ.get("TTS_ANAL_SYSTEMS", "f5tts,xtts,cosyvoice3,chatterbox,indextts").split(",")
CONDS = ["real"] + SYSTEMS
EXCLUDE = set((ROOT / os.environ.get("TTS_ANAL_EXCLUDE", "data/manifests/exclude17.txt")).read_text().split())
RNG = np.random.default_rng(0)


def main():
    rows = [json.loads(l) for l in open(ROOT / "data/manifests/main.jsonl")]
    X, y, spk = [], [], []
    for r in rows:
        if r["utt_id"] in EXCLUDE:
            continue
        for c in CONDS:
            p = (r["real_wav"] if c == "real"
                 else str(ROOT / "data/generated" / c / f"{r['utt_id']}.wav"))
            wav, sr = sf.read(p, dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            m = librosa.feature.mfcc(y=wav, sr=16000, n_mfcc=40)
            X.append(np.concatenate([m.mean(1), m.std(1)]))
            y.append(c)
            spk.append(r["speaker"])
    X = normalize(np.array(X))
    y, spk = np.array(y), np.array(spk)

    out = []
    for tag, mask in [(f"{len(SYSTEMS) + 1}way", np.ones(len(y), bool)), ("ttsonly", y != "real")]:
        pred = cross_val_predict(KNeighborsClassifier(5, metric="cosine"),
                                 X[mask], y[mask], cv=GroupKFold(5),
                                 groups=spk[mask], n_jobs=8)
        f1 = f1_score(y[mask], pred, average="macro")
        sps = np.unique(spk[mask])
        boots = []
        for _ in range(1000):
            s = RNG.choice(sps, len(sps), replace=True)
            idx = np.concatenate([np.where(spk[mask] == x)[0] for x in s])
            boots.append(f1_score(y[mask][idx], pred[idx], average="macro"))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append({"setting": tag, "macro_f1": f1, "ci_lo": lo, "ci_hi": hi})
        print(f"MFCC cloning {tag}: {f1:.3f} [{lo:.3f},{hi:.3f}]")
    pd.DataFrame(out).to_csv(ROOT / RESULTS / "mfcc_cloning.csv", index=False)


if __name__ == "__main__":
    main()
