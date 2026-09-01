"""Generate speech with XTTS-v2 (zero-shot cloning) for every row of a manifest.

Run inside the `tts_xtts` conda env. Set COQUI_TOS_AGREED=1 (CPML license,
non-commercial research use) and TTS_HOME for the checkpoint cache.
"""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[2]))
CKPTS = Path(os.environ.get("TTS_ANAL_CKPTS", ROOT / "ckpts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/main.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/generated/xtts"))
    args = ap.parse_args()

    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    os.environ.setdefault("TTS_HOME", os.environ.get("TTS_ANAL_CKPTS", str(Path(__file__).resolve().parents[2] / "ckpts")))

    import torch
    from TTS.api import TTS

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.manifest)]
    for i, r in enumerate(rows):
        out = out_dir / f"{r['utt_id']}.wav"
        if out.exists():
            continue
        tts.tts_to_file(
            text=r["text"],
            speaker_wav=r["prompt_wav"],
            language="en",
            file_path=str(out),
        )
        print(f"[{i + 1}/{len(rows)}] {out.name}")


if __name__ == "__main__":
    main()
