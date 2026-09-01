"""Generate speech with F5-TTS (zero-shot cloning) for every row of a manifest."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/main.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/generated/f5tts"))
    args = ap.parse_args()

    from f5_tts.api import F5TTS

    tts = F5TTS()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.manifest)]
    for i, r in enumerate(rows):
        out = out_dir / f"{r['utt_id']}.wav"
        if out.exists():
            continue
        tts.infer(
            ref_file=r["prompt_wav"],
            ref_text=r["prompt_text"],
            gen_text=r["text"],
            file_wave=str(out),
            seed=0,
        )
        print(f"[{i + 1}/{len(rows)}] {out.name}")


if __name__ == "__main__":
    main()
