"""Simple-cue baseline: can trivial per-utterance descriptors explain the
deep-layer attribution signal?

Descriptors: duration, silence ratio, F0 mean/std/range (pyin), RMS
energy mean/std, chars-per-second. Speaker-disjoint GroupKFold kNN, same
protocol as the SSL probes. If this baseline is far below deep-layer SSL
accuracy, length/prosody/rate cues alone do not account for the signal.
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = os.environ.get("TTS_ANAL_RESULTS", "results/paper5c17")
SYSTEMS = os.environ.get("TTS_ANAL_SYSTEMS", "f5tts,xtts,cosyvoice3,chatterbox,indextts").split(",")
CONDS = ["real"] + SYSTEMS
EXCLUDE = set((ROOT / os.environ.get("TTS_ANAL_EXCLUDE", "data/manifests/exclude17.txt")).read_text().split())
RNG = np.random.default_rng(0)


def descriptors(path, text):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    dur = len(wav) / 16000
    intervals = librosa.effects.split(wav, top_db=35)
    voiced = sum(e - s for s, e in intervals) / 16000
    sil_ratio = 1 - voiced / max(dur, 1e-6)
    f0 = librosa.yin(wav, fmin=60, fmax=400, sr=16000, frame_length=1024)
    f0 = f0[(f0 > 60) & (f0 < 400)]
    f0m, f0s, f0r = (np.log(f0).mean(), np.log(f0).std(),
                     np.log(f0).max() - np.log(f0).min()) if len(f0) > 10 else (0, 0, 0)
    rms = librosa.feature.rms(y=wav)[0]
    return [dur, sil_ratio, f0m, f0s, f0r, rms.mean(), rms.std(),
            len(text) / max(voiced, 0.3)]


def main():
    rows = [json.loads(l) for l in open(ROOT / "data/manifests/main.jsonl")]
    X, y, spk = [], [], []
    for r in rows:
        if r["utt_id"] in EXCLUDE:
            continue
        for c in CONDS:
            p = (r["real_wav"] if c == "real"
                 else str(ROOT / "data/generated" / c / f"{r['utt_id']}.wav"))
            X.append(descriptors(p, r["text"]))
            y.append(c)
            spk.append(r["speaker"])
    # scaler inside the CV pipeline -> fold-local fitting (no test leakage);
    # cosine metric to match the SSL probes
    X = np.array(X)
    y, spk = np.array(y), np.array(spk)

    out = []
    for tag, mask in [(f"{len(SYSTEMS) + 1}way", np.ones(len(y), bool)), ("ttsonly", y != "real")]:
        clf = make_pipeline(StandardScaler(),
                            KNeighborsClassifier(5, metric="cosine"))
        pred = cross_val_predict(clf, X[mask], y[mask],
                                 cv=GroupKFold(5), groups=spk[mask], n_jobs=8)
        f1 = f1_score(y[mask], pred, average="macro")
        sps = np.unique(spk[mask])
        boots = []
        for _ in range(1000):
            s = RNG.choice(sps, len(sps), replace=True)
            idx = np.concatenate([np.where(spk[mask] == x)[0] for x in s])
            boots.append(f1_score(y[mask][idx], pred[idx], average="macro"))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append({"setting": tag, "macro_f1": f1, "ci_lo": lo, "ci_hi": hi})
        print(f"simple-cues {tag}: {f1:.3f} [{lo:.3f},{hi:.3f}]")
    pd.DataFrame(out).to_csv(ROOT / RESULTS / "simplecues.csv", index=False)


if __name__ == "__main__":
    main()
