"""More analysis-synthesis interventions (tts_anal env).

  resynth_bigvgan    : real -> BigVGAN mel (24k/100-band) -> BigVGAN v2
                       (lineage-matched to IndexTTS's BigVGAN2 decoder family)
  resynth_griffinlim : real -> mel (24k/100-band) -> Griffin-Lim
                       (analysis-only control: same mel bottleneck, no neural decoder)
"""

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))


def load24k(path):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != 24000:
        wav = torchaudio.functional.resample(wav, sr, 24000)
    return wav


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/main.jsonl"))
    ap.add_argument("--which", nargs="+", default=["bigvgan", "griffinlim"])
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest)]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "bigvgan" in args.which:
        from third_party.BigVGAN import bigvgan as bigvgan_mod
        from third_party.BigVGAN.meldataset import get_mel_spectrogram

        model = bigvgan_mod.BigVGAN.from_pretrained(
            "nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False
        ).eval().to(device)
        model.remove_weight_norm()
        out_dir = ROOT / "data/generated/resynth_bigvgan"
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in rows:
            out = out_dir / f"{r['utt_id']}.wav"
            if out.exists():
                continue
            wav = load24k(r["real_wav"])
            mel = get_mel_spectrogram(wav, model.h).to(device)
            y = model(mel).squeeze(0).cpu()
            torchaudio.save(str(out), y, 24000)
        print("bigvgan done")

    if "griffinlim" in args.which:
        import librosa

        out_dir = ROOT / "data/generated/resynth_griffinlim"
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in rows:
            out = out_dir / f"{r['utt_id']}.wav"
            if out.exists():
                continue
            wav = load24k(r["real_wav"]).squeeze(0).numpy()
            mel = librosa.feature.melspectrogram(
                y=wav, sr=24000, n_fft=1024, hop_length=256, n_mels=100)
            y = librosa.feature.inverse.mel_to_audio(
                mel, sr=24000, n_fft=1024, hop_length=256, n_iter=32)
            torchaudio.save(str(out), torch.from_numpy(y).unsqueeze(0), 24000)
        print("griffinlim done")


if __name__ == "__main__":
    main()
