"""ASR control: transcribe every clean-condition utterance with
faster-whisper large-v3 (cached) and compute word/char error statistics
against the reference text. Output: results/paper/asr_wer.csv
"""

import json
import re
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = os.environ.get("TTS_ANAL_RESULTS", "results/paper5c17")
CKPTS = Path(os.environ.get("TTS_ANAL_CKPTS", ROOT / "ckpts"))
SYSTEMS = os.environ.get("TTS_ANAL_SYSTEMS", "f5tts,xtts,cosyvoice3,chatterbox,indextts").split(",")
CONDS = ["real"] + SYSTEMS
EXCLUDE = set((ROOT / os.environ.get("TTS_ANAL_EXCLUDE", "data/manifests/exclude17.txt")).read_text().split())


def norm(t):
    return re.sub(r"[^a-z' ]", "", t.lower()).split()


def align_counts(ref, hyp):
    """Levenshtein with operation counts (sub, del, ins)."""
    m, n = len(ref), len(hyp)
    D = [[(0, 0, 0, 0)] * (n + 1) for _ in range(m + 1)]  # (cost,s,d,i)
    for i in range(1, m + 1):
        D[i][0] = (i, 0, i, 0)
    for j in range(1, n + 1):
        D[0][j] = (j, 0, 0, j)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                D[i][j] = D[i - 1][j - 1]
            else:
                sub = (D[i - 1][j - 1][0] + 1, D[i - 1][j - 1][1] + 1,
                       D[i - 1][j - 1][2], D[i - 1][j - 1][3])
                dele = (D[i - 1][j][0] + 1, D[i - 1][j][1],
                        D[i - 1][j][2] + 1, D[i - 1][j][3])
                ins = (D[i][j - 1][0] + 1, D[i][j - 1][1],
                       D[i][j - 1][2], D[i][j - 1][3] + 1)
                D[i][j] = min(sub, dele, ins)
    return D[m][n]


def cer(ref, hyp):
    r, h = list(ref), list(hyp)
    m, n = len(r), len(h)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return prev[n] / max(m, 1)


def main():
    from faster_whisper import WhisperModel

    model = WhisperModel("large-v3", device="cuda", compute_type="float16",
                         download_root=str(CKPTS / "hub"))
    rows = [json.loads(l) for l in open(ROOT / "data/manifests/main.jsonl")]
    rec = []
    for r in rows:
        if r["utt_id"] in EXCLUDE:
            continue
        for c in CONDS:
            p = (r["real_wav"] if c == "real"
                 else str(ROOT / "data/generated" / c / f"{r['utt_id']}.wav"))
            segs, _ = model.transcribe(p, language="en", beam_size=5)
            hyp = " ".join(s.text for s in segs).strip()
            ref_w, hyp_w = norm(r["text"]), norm(hyp)
            cost, s, d, i = align_counts(ref_w, hyp_w)
            nref = max(len(ref_w), 1)
            rec.append({"cond": c, "utt_id": r["utt_id"],
                        "speaker": r["speaker"],
                        "wer": cost / nref, "sub": s / nref, "del": d / nref,
                        "ins": i / nref,
                        "cer": cer(" ".join(ref_w), " ".join(hyp_w)),
                        "hyp": hyp})
    df = pd.DataFrame(rec)
    df.to_csv(ROOT / RESULTS / "asr_wer.csv", index=False)
    print(df.groupby("cond")[["wer", "cer"]].mean().round(4))
    print("WER=0 rate:", df.groupby("cond")["wer"].apply(
        lambda x: (x == 0).mean()).round(3).to_dict())


if __name__ == "__main__":
    main()
