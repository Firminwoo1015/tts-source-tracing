"""Generate speech with IndexTTS-1.5 (zero-shot cloning; GPT-style AM + BigVGAN2).

Run inside `tts_index` env from the index-tts repo root so its imports resolve.
Checkpoints expected at $TTS_ANAL_CKPTS/IndexTTS-1.5.
"""

import argparse
import json
import sys
import os
from pathlib import Path

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[2]))
CKPTS = Path(os.environ.get("TTS_ANAL_CKPTS", ROOT / "ckpts"))
REPO = ROOT / "third_party" / "index-tts"
CKPT = CKPTS / "IndexTTS-1.5"
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/main.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/generated/indextts"))
    args = ap.parse_args()

    from indextts.infer import IndexTTS

    tts = IndexTTS(model_dir=str(CKPT), cfg_path=str(CKPT / "config.yaml"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.manifest)]
    for i, r in enumerate(rows):
        out = out_dir / f"{r['utt_id']}.wav"
        if out.exists():
            continue
        tts.infer(r["prompt_wav"], r["text"], str(out))
        print(f"[{i + 1}/{len(rows)}] {out.name}")


if __name__ == "__main__":
    main()
