"""Generate speech with Chatterbox (Resemble AI) — zero-shot cloning.

Scientifically interesting: Chatterbox's s3gen decoder shares the CosyVoice
lineage (S3 tokens + flow + HiFT), so if the decoder carries the fingerprint,
chatterbox should sit close to cosyvoice2 in low SSL layers despite a
completely different text/LM stack. Run inside `tts_chatter` env.
"""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/main.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/generated/chatterbox"))
    args = ap.parse_args()

    import torchaudio
    import perth

    # PerthImplicitWatermarker is broken in this install AND an audio watermark
    # would itself confound the fingerprint analysis — use the no-op watermarker.
    perth.PerthImplicitWatermarker = perth.DummyWatermarker
    from chatterbox.tts import ChatterboxTTS

    model = ChatterboxTTS.from_pretrained(device="cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.manifest)]
    for i, r in enumerate(rows):
        out = out_dir / f"{r['utt_id']}.wav"
        if out.exists():
            continue
        wav = model.generate(r["text"], audio_prompt_path=r["prompt_wav"])
        torchaudio.save(str(out), wav.cpu(), model.sr)
        print(f"[{i + 1}/{len(rows)}] {out.name}")


if __name__ == "__main__":
    main()
