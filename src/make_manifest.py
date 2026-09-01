"""Build a generation manifest from LibriSpeech test-clean.

For each selected speaker we pick:
  - one enrollment utterance (speaker prompt for zero-shot cloning), 4-10 s long
  - N target utterances whose transcripts every TTS system will synthesize

Output: data/manifests/<name>.jsonl with
  {utt_id, speaker, text, prompt_wav, prompt_text, real_wav}
"""

import argparse
import json
import random
import os
from pathlib import Path

import soundfile as sf

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
LIBRI = Path(os.environ.get("TTS_ANAL_DATASETS", ROOT / "datasets")) / "LibriSpeech/test-clean"


def norm_text(text: str) -> str:
    """LibriSpeech transcripts are ALL-CAPS without punctuation; modern TTS
    systems expect normal sentences."""
    t = text.strip().lower().capitalize()
    if not t.endswith((".", "!", "?")):
        t += "."
    return t


def utt_duration(flac: Path) -> float:
    info = sf.info(str(flac))
    return info.frames / info.samplerate


def load_speaker_utts(spk_dir: Path):
    """Return list of (utt_id, flac_path, text) for one speaker."""
    utts = []
    for chapter in sorted(spk_dir.iterdir()):
        trans = list(chapter.glob("*.trans.txt"))
        if not trans:
            continue
        texts = {}
        for line in trans[0].read_text().strip().splitlines():
            uid, text = line.split(" ", 1)
            texts[uid] = text
        for flac in sorted(chapter.glob("*.flac")):
            uid = flac.stem
            if uid in texts:
                utts.append((uid, flac, texts[uid]))
    return utts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pilot")
    ap.add_argument("--n-speakers", type=int, default=4)
    ap.add_argument("--n-utts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=10.0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    speakers = sorted(p for p in LIBRI.iterdir() if p.is_dir())
    rng.shuffle(speakers)

    out = ROOT / "data" / "manifests" / f"{args.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for spk_dir in speakers:
        if len({r["speaker"] for r in rows}) >= args.n_speakers:
            break
        utts = load_speaker_utts(spk_dir)
        # keep mid-length utterances only
        utts = [
            (u, f, t)
            for u, f, t in utts
            if args.min_dur <= utt_duration(f) <= args.max_dur
        ]
        if len(utts) < args.n_utts + 1:
            continue
        rng.shuffle(utts)
        prompt = utts[0]
        targets = utts[1 : 1 + args.n_utts]
        for uid, flac, text in targets:
            rows.append(
                {
                    "utt_id": uid,
                    "speaker": spk_dir.name,
                    "text": norm_text(text),
                    "prompt_wav": str(prompt[1]),
                    "prompt_text": norm_text(prompt[2]),
                    "real_wav": str(flac),
                }
            )

    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_spk = len({r["speaker"] for r in rows})
    print(f"wrote {len(rows)} utts from {n_spk} speakers -> {out}")


if __name__ == "__main__":
    main()
