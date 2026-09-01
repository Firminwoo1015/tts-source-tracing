"""True F5 mel-only ablation: Griffin-Lim inversion of VOCOS'S OWN analysis.

Uses the exact torchaudio MelSpectrogram inside Vocos's feature extractor
(same STFT params and mel filterbank F5-TTS conditions on), inverts the mel
filterbank by pseudo-inverse and reconstructs phase with Griffin-Lim. Unlike
resynth_griffinlim (librosa power-mel, slaney filterbank), this shares F5's
actual analysis representation, so it is the legitimate mel-bottleneck probe.

Output: data/generated/resynth_glvocos/<utt_id>.wav (24 kHz)
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[2]))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/main.jsonl"))
    args = ap.parse_args()

    from vocos import Vocos

    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    msp = vocos.feature_extractor.mel_spec  # torchaudio MelSpectrogram (power=1)
    fb = msp.mel_scale.fb                   # (n_freqs, n_mels)
    # torchaudio: mel = fb.T @ spec  =>  spec ~= pinv(fb.T) @ mel
    fb_inv = torch.linalg.pinv(fb.T)        # (n_freqs, n_mels)
    gl = torchaudio.transforms.GriffinLim(
        n_fft=msp.n_fft, hop_length=msp.hop_length,
        win_length=msp.win_length, power=1.0, n_iter=32)

    out_dir = ROOT / "data/generated/resynth_glvocos"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.manifest)]
    for r in rows:
        out = out_dir / f"{r['utt_id']}.wav"
        if out.exists():
            continue
        wav, sr = torchaudio.load(r["real_wav"])
        wav = wav.mean(0, keepdim=True)
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, sr, 24000)
        mel = msp(wav)                       # (1, n_mels, T) linear-magnitude mel
        spec = (fb_inv @ mel.squeeze(0)).clamp_min(0.0)  # (n_freqs, T)
        y = gl(spec.unsqueeze(0))
        torchaudio.save(str(out), y, 24000)
    print("glvocos done:", len(list(out_dir.glob('*.wav'))))


if __name__ == "__main__":
    main()
