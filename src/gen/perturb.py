"""Create perturbed copies of all 5 conditions to locate the fingerprint.

Perturbations (CPU):
  lp2k / lp4k / lp6k : low-pass at 2/4/6 kHz    (is the signature in high bands?)
  hp2k               : high-pass at 2 kHz        (or in low bands?)
  phaserand          : STFT magnitude kept, phase randomized (GAN phase artifacts?)
  mp3_64k            : MP3 64 kbps round-trip    (robustness / practical forensics)
  noise20            : white noise at 20 dB SNR  (robustness)

Output: data/perturbed/<pert>/<cond>/<utt_id>.wav (16 kHz mono)
"""

import json
import subprocess
import tempfile
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfiltfilt

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[2]))
CONDS = ["real", "f5tts", "kokoro", "xtts", "cosyvoice2"]
PERTS = ["lp2k", "lp4k", "lp6k", "hp2k", "phaserand", "mp3_64k", "noise20",
         "common", "common_sym"]
RNG = np.random.default_rng(0)


def load16k(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav


def apply(pert, wav):
    if pert == "common_sym":
        # symmetric-channel control: EVERY condition (input already unified to
        # 16 kHz by load16k) passes the same 16->24->16 kHz resampler chain,
        # then the common trim/RMS/16-bit post-processing. This equalizes the
        # final resampling history across classes (real speech never existed
        # at 24 kHz, so a perfectly identical full history is impossible;
        # this shares the last two resampling hops).
        up = librosa.resample(wav, orig_sr=16000, target_sr=24000)
        wav = librosa.resample(up, orig_sr=24000, target_sr=16000)
        return apply("common", wav)
    if pert == "common":
        # common-channel control: identical resampler (already librosa/soxr in
        # load16k), silence trim, RMS target, 16-bit quantization for ALL
        # conditions — kills file-pipeline confounds (SR, bit depth, levels).
        idx = np.where(np.abs(wav) > 10 ** (-35 / 20) * np.abs(wav).max())[0]
        if len(idx) > 1:
            wav = wav[idx[0]: idx[-1] + 1]
        wav = wav * (0.05 / (np.sqrt((wav ** 2).mean()) + 1e-8))
        return (np.round(np.clip(wav, -1, 1) * 32767) / 32767).astype(np.float32)
    if pert.startswith("lp"):
        fc = int(pert[2:-1]) * 1000
        sos = butter(8, fc, btype="low", fs=16000, output="sos")
        return sosfiltfilt(sos, wav).astype(np.float32)
    if pert.startswith("hp"):
        fc = int(pert[2:-1]) * 1000
        sos = butter(8, fc, btype="high", fs=16000, output="sos")
        return sosfiltfilt(sos, wav).astype(np.float32)
    if pert == "phaserand":
        spec = librosa.stft(wav, n_fft=1024, hop_length=256)
        mag = np.abs(spec)
        phase = np.exp(1j * RNG.uniform(-np.pi, np.pi, spec.shape))
        return librosa.istft(mag * phase, hop_length=256, length=len(wav)).astype(np.float32)
    if pert == "noise20":
        rms = np.sqrt((wav ** 2).mean())
        noise = RNG.normal(0, rms / (10 ** (20 / 20)), len(wav))
        return (wav + noise).astype(np.float32)
    if pert == "mp3_64k":
        with tempfile.TemporaryDirectory() as td:
            w, m = Path(td) / "a.wav", Path(td) / "a.mp3"
            sf.write(w, wav, 16000)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-i", w,
                            "-b:a", "64k", m], check=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-i", m,
                            "-ar", "16000", "-ac", "1", w], check=True)
            out, _ = sf.read(w, dtype="float32")
            return out[: len(wav)]
    raise ValueError(pert)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="+", default=CONDS)
    ap.add_argument("--perts", nargs="+", default=PERTS)
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for stochastic perturbations (noise20, phaserand)")
    ap.add_argument("--suffix", default="", help="output dir suffix, e.g. _s2 -> data/perturbed/noise20_s2/")
    args = ap.parse_args()
    global RNG
    RNG = np.random.default_rng(args.seed)

    rows = [json.loads(l) for l in open(ROOT / "data/manifests/main.jsonl")]
    for pert in args.perts:
        for c in args.conds:
            out_dir = ROOT / "data" / "perturbed" / (pert + args.suffix) / c
            out_dir.mkdir(parents=True, exist_ok=True)
            for r in rows:
                out = out_dir / f"{r['utt_id']}.wav"
                if out.exists():
                    continue
                src = (r["real_wav"] if c == "real"
                       else ROOT / "data/generated" / c / f"{r['utt_id']}.wav")
                sf.write(out, apply(pert, load16k(str(src))), 16000)
        print(f"{pert} done")


if __name__ == "__main__":
    main()
