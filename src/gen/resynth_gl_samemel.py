"""Same-mel Griffin-Lim decoder-swap controls (specification fixed before generation: analysis_plan.md, N4).

  resynth_glc3 : real -> CosyVoice3 load_wav + Matcha mel (CosyVoice3 config) -> Griffin-Lim   (env tts_cosy)
  resynth_glcb : real -> librosa.load + Chatterbox mel_spectrogram (fmax 8 kHz) -> Griffin-Lim  (env tts_chatter)

The mel is computed with the very functions the native HiFT paths use (resynth_cosy3.py:
cosy.frontend._extract_speech_feat = load_wav + feat_extractor; resynth_chatterbox.py: librosa.load +
s3gen.mel_extractor). Only the reconstruction changes. Inversion mirrors src/gen/resynth_glvocos.py:
pseudo-inverse of the mel filter bank on the linear mel, clamp at zero, torchaudio GriffinLim with
power 1 and 32 iterations (momentum 0.99, random phase init), here seeded per utterance.
The Matcha framing (center=False, reflect pad (n_fft-hop)/2) centres frame t at sample hop*t + 240,
centred Griffin-Lim at hop*t, so the output is delayed by 240 samples and cut to the input length.
"""
import argparse
import json
import os
import sys
import zlib
from pathlib import Path

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[2]))
SR, N_FFT, HOP, WIN, N_MELS = 24000, 1920, 480, 1920, 80
CFG = {"c3": dict(fmax=None, out="resynth_glc3"), "cb": dict(fmax=8000, out="resynth_glcb")}


def frontend(system):
    """Return (load(path) -> (1, N) float tensor at 24 kHz, mel(y) -> (1, 80, T) log mel), system-native."""
    import torch
    if system == "c3":
        cosy = ROOT / "third_party" / "CosyVoice"
        sys.path.insert(0, str(cosy)); sys.path.insert(0, str(cosy / "third_party" / "Matcha-TTS"))
        from cosyvoice.utils.file_utils import load_wav
        from matcha.utils.audio import mel_spectrogram
        load = lambda p: load_wav(p, SR)
        mel = lambda y: mel_spectrogram(y, n_fft=N_FFT, num_mels=N_MELS, sampling_rate=SR, hop_size=HOP,
                                        win_size=WIN, fmin=0, fmax=None, center=False)
    else:
        import librosa
        from chatterbox.models.s3gen.utils.mel import mel_spectrogram
        load = lambda p: torch.from_numpy(librosa.load(p, sr=SR)[0]).float()[None]
        mel = lambda y: mel_spectrogram(y)   # Chatterbox defaults: 1920/480/80, fmax 8000, center=False
    return load, mel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["c3", "cb"], required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/main.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import librosa
    import torch
    import torchaudio

    load, mel_fn = frontend(a.system)
    fmax = CFG[a.system]["fmax"]
    basis = torch.from_numpy(librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=fmax)).float()
    basis_inv = torch.linalg.pinv(basis)                         # (n_freqs, n_mels)
    gl = torchaudio.transforms.GriffinLim(n_fft=N_FFT, win_length=WIN, hop_length=HOP, power=1.0, n_iter=32)
    shift = 240                                                 # see docstring

    out_dir = ROOT / "data/generated" / CFG[a.system]["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(a.manifest)]
    if a.limit:
        rows = rows[: a.limit]
    with torch.inference_mode():
        for i, r in enumerate(rows):
            out = out_dir / f"{r['utt_id']}.wav"
            if out.exists():
                continue
            y = load(r["real_wav"])                              # (1, N) at 24 kHz
            logmel = mel_fn(y)                                   # (1, 80, T)
            spec = (basis_inv @ torch.exp(logmel.squeeze(0))).clamp_min(0.0)   # (n_freqs, T)
            torch.manual_seed(zlib.crc32(r["utt_id"].encode()))
            rec = gl(spec.unsqueeze(0)).squeeze(0)               # centred framing
            rec = torch.cat([torch.zeros(shift), rec])[: y.shape[-1]]
            if rec.shape[0] < y.shape[-1]:
                rec = torch.cat([rec, torch.zeros(y.shape[-1] - rec.shape[0])])
            torchaudio.save(str(out), rec[None].float(), SR)
            if i % 50 == 0:
                print(f"{a.system} [{i + 1}/{len(rows)}] mel {tuple(logmel.shape)} -> {rec.shape[0]} samples", flush=True)
    print(f"{CFG[a.system]['out']} done", flush=True)


if __name__ == "__main__":
    main()
