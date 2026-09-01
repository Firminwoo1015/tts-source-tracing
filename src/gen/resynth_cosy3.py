"""Analysis-synthesis interventions using Fun-CosyVoice3's own stack (tts_cosy env).

  resynth_hift3 : real -> CosyVoice3 mel frontend (24 kHz, 80 mel, hop 480) -> CausalHiFT
                  decoder (decoder path only; exact path match for cosyvoice3)
  resynth_s3vc3 : real -> speech_tokenizer_v3 tokens -> DiT flow -> CausalHiFT via
                  inference_vc with the SAME speaker's enrollment prompt (full
                  analysis-synthesis round trip of the CosyVoice3 lineage, no text/LM)
Mirrors src/gen/resynth_cosy.py (CosyVoice2).
"""

import argparse
import json
import sys
import os
from pathlib import Path

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[2]))
CKPTS = Path(os.environ.get("TTS_ANAL_CKPTS", ROOT / "ckpts"))
COSY = ROOT / "third_party" / "CosyVoice"
sys.path.insert(0, str(COSY))
sys.path.insert(0, str(COSY / "third_party" / "Matcha-TTS"))


def snapshot_dir(repo="FunAudioLLM/Fun-CosyVoice3-0.5B-2512"):
    base = CKPTS / "hub" / ("models--" + repo.replace("/", "--"))
    rev = (base / "refs" / "main").read_text().strip()
    return str(base / "snapshots" / rev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/main.jsonl"))
    ap.add_argument("--which", nargs="+", default=["hift3", "s3vc3"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import CosyVoice3

    cosy = CosyVoice3(snapshot_dir(), load_trt=False, load_vllm=False, fp16=False)
    rows = [json.loads(l) for l in open(args.manifest)]
    if args.limit:
        rows = rows[: args.limit]

    if "hift3" in args.which:
        out_dir = ROOT / "data/generated/resynth_hift3"
        out_dir.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            for r in rows:
                out = out_dir / f"{r['utt_id']}.wav"
                if out.exists():
                    continue
                mel, _ = cosy.frontend._extract_speech_feat(r["real_wav"])
                mel = mel.transpose(1, 2)  # (1, 80, T)
                res = cosy.model.hift.inference(speech_feat=mel)
                wav = res[0] if isinstance(res, tuple) else res
                torchaudio.save(str(out), wav.cpu(), cosy.sample_rate)
        print("hift3 done")

    if "s3vc3" in args.which:
        out_dir = ROOT / "data/generated/resynth_s3vc3"
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in rows:
            out = out_dir / f"{r['utt_id']}.wav"
            if out.exists():
                continue
            chunks = [o["tts_speech"] for o in cosy.inference_vc(
                r["real_wav"], r["prompt_wav"], stream=False)]
            torchaudio.save(str(out), torch.cat(chunks, dim=1), cosy.sample_rate)
        print("s3vc3 done")


if __name__ == "__main__":
    main()
