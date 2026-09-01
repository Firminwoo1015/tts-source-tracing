"""Generate speech with Fun-CosyVoice3-0.5B-2512 (zero-shot cloning; Qwen2 LM + DiT flow +
CausalHiFT, 24 kHz) for every row of a manifest.

Run inside the `tts_cosy` conda env (same requirements.txt as the pinned CosyVoice2
clone); code from third_party/CosyVoice (pinned commit 074ca6d already includes CosyVoice3).
Checkpoint: HF snapshot of FunAudioLLM/Fun-CosyVoice3-0.5B-2512 ($HF_HOME=$TTS_ANAL_CKPTS).
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


CV3_PREFIX = "You are a helpful assistant.<|endofprompt|>"


def snapshot_dir(repo="FunAudioLLM/Fun-CosyVoice3-0.5B-2512"):
    base = CKPTS / "hub" / ("models--" + repo.replace("/", "--"))
    rev = (base / "refs" / "main").read_text().strip()
    return str(base / "snapshots" / rev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/main.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/generated/cosyvoice3"))
    ap.add_argument("--limit", type=int, default=0, help="stop after N rows (smoke test)")
    args = ap.parse_args()

    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import CosyVoice3

    cosy = CosyVoice3(snapshot_dir(), load_trt=False, load_vllm=False, fp16=False)
    print("CosyVoice3 sample_rate:", cosy.sample_rate)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.manifest)]
    if args.limit:
        rows = rows[: args.limit]
    for i, r in enumerate(rows):
        out = out_dir / f"{r['utt_id']}.wav"
        if out.exists():
            continue
        # CosyVoice3 requires the released instruction prefix ending in <|endofprompt|>
        # before the prompt transcript (see third_party/CosyVoice/example.py).
        chunks = [
            o["tts_speech"]
            for o in cosy.inference_zero_shot(
                r["text"], CV3_PREFIX + r["prompt_text"], r["prompt_wav"], stream=False
            )
        ]
        wav = torch.cat(chunks, dim=1)
        torchaudio.save(str(out), wav, cosy.sample_rate)
        print(f"[{i + 1}/{len(rows)}] {out.name} {wav.shape[1] / cosy.sample_rate:.2f}s", flush=True)


if __name__ == "__main__":
    main()
