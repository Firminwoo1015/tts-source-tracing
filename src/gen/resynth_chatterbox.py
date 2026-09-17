"""Analysis-synthesis interventions using Chatterbox's own S3Gen stack (tts_chatter env).

Mirrors src/gen/resynth_cosy3.py (CosyVoice3):
  resynth_hiftcb : real -> Chatterbox S3Gen mel extractor (24 kHz, 80 mel, hop 480)
                   -> HiFTGenerator (decoder path only; counterpart of resynth_hift3)
  resynth_s3vccb : real -> S3TokenizerV2 tokens -> flow -> HiFT via ChatterboxVC with
                   the SAME speaker's enrollment prompt (full analysis-synthesis round
                   trip, no text/LM; counterpart of resynth_s3vc3)
Hypothesized target for both: chatterbox. Specification fixed before generation: analysis_plan.md, section N3.
"""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/main.jsonl"))
    ap.add_argument("--which", nargs="+", default=["hiftcb", "s3vccb"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import librosa
    import torch
    import torchaudio
    import perth

    # same as src/gen/chatterbox_gen.py: no audio watermark (it would confound the analysis)
    perth.PerthImplicitWatermarker = perth.DummyWatermarker
    from chatterbox.vc import ChatterboxVC
    from chatterbox.models.s3gen import S3GEN_SR

    vc = ChatterboxVC.from_pretrained(device="cuda")
    s3gen = vc.s3gen
    assert type(vc.watermarker).__name__ == "DummyWatermarker", type(vc.watermarker)

    rows = [json.loads(l) for l in open(args.manifest)]
    if args.limit:
        rows = rows[: args.limit]

    if "hiftcb" in args.which:
        out_dir = ROOT / "data/generated/resynth_hiftcb"
        out_dir.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            for i, r in enumerate(rows):
                out = out_dir / f"{r['utt_id']}.wav"
                if out.exists():
                    continue
                wav24, _ = librosa.load(r["real_wav"], sr=S3GEN_SR)
                y = torch.from_numpy(wav24).float().to(s3gen.device)[None]
                mel = s3gen.mel_extractor(y).to(dtype=s3gen.dtype)  # (1, 80, T)
                wav, _ = s3gen.hift_inference(speech_feat=mel)
                torchaudio.save(str(out), wav.detach().float().cpu().reshape(1, -1), S3GEN_SR)
                if i % 50 == 0:
                    print(f"hiftcb [{i + 1}/{len(rows)}] mel {tuple(mel.shape)} -> wav {tuple(wav.shape)}", flush=True)
        print("hiftcb done", flush=True)

    if "s3vccb" in args.which:
        out_dir = ROOT / "data/generated/resynth_s3vccb"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(rows):
            out = out_dir / f"{r['utt_id']}.wav"
            if out.exists():
                continue
            wav = vc.generate(r["real_wav"], target_voice_path=r["prompt_wav"])
            torchaudio.save(str(out), wav.float().cpu().reshape(1, -1), vc.sr)
            if i % 50 == 0:
                print(f"s3vccb [{i + 1}/{len(rows)}] wav {tuple(wav.shape)}", flush=True)
        print("s3vccb done", flush=True)


if __name__ == "__main__":
    main()
