"""Phase 2 (decoder transplant): pass REAL speech through waveform decoders only.

If the per-system clusters are carried by the final waveform decoder, real speech
resynthesized by a system's decoder should move into (or toward) that system's
cluster, despite having perfectly natural prosody/content/speaker.

Conditions:
  resynth_vocos   : real -> mel -> Vocos (exactly F5-TTS's vocoder path)
  resynth_encodec : real -> EnCodec 24k encode/decode at 6 kbps (codec-LM decoder family)
  resynth_dac     : real -> Descript Audio Codec 24k encode/decode
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[2]))


def load_resampled(path, sr):
    wav, in_sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)
    if in_sr != sr:
        wav = torchaudio.functional.resample(wav, in_sr, sr)
    return wav


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/main.jsonl"))
    ap.add_argument("--which", nargs="+", default=["vocos", "encodec", "dac"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [json.loads(l) for l in open(args.manifest)]

    if "vocos" in args.which:
        from vocos import Vocos

        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        out_dir = ROOT / "data/generated/resynth_vocos"
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in rows:
            out = out_dir / f"{r['utt_id']}.wav"
            if out.exists():
                continue
            wav = load_resampled(r["real_wav"], 24000).to(device)
            y = vocos(wav)
            torchaudio.save(str(out), y.cpu(), 24000)
        print("vocos done")

    if "encodec" in args.which:
        from transformers import EncodecModel

        codec = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
        out_dir = ROOT / "data/generated/resynth_encodec"
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in rows:
            out = out_dir / f"{r['utt_id']}.wav"
            if out.exists():
                continue
            wav = load_resampled(r["real_wav"], 24000).to(device)
            enc = codec.encode(wav.unsqueeze(0), bandwidth=6.0)
            y = codec.decode(enc.audio_codes, enc.audio_scales)[0]
            torchaudio.save(str(out), y.squeeze(0).cpu(), 24000)
        print("encodec done")

    if "dac" in args.which:
        import dac

        model = dac.DAC.load(dac.utils.download(model_type="24khz")).to(device)
        out_dir = ROOT / "data/generated/resynth_dac"
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in rows:
            out = out_dir / f"{r['utt_id']}.wav"
            if out.exists():
                continue
            wav = load_resampled(r["real_wav"], 24000).to(device)
            x = model.preprocess(wav.unsqueeze(0), 24000)
            z, *_ = model.encode(x)
            y = model.decode(z)
            torchaudio.save(str(out), y.squeeze(0).cpu(), 24000)
        print("dac done")


if __name__ == "__main__":
    main()
