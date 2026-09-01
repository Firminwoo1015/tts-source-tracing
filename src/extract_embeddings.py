"""Extract layer-wise SSL embeddings (mean-pooled per utterance).

For each condition (real / <system>) and each utterance in the manifest, run the
SSL model with output_hidden_states=True and store the per-layer mean-pooled
embedding: array [n_layers+1, dim].

Output: results/embeddings/<ssl_short>/<condition>.npz
  with keys: utt_ids (N,), speakers (N,), embs (N, L+1, D)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))

SSL_MODELS = {
    "wavlm": "microsoft/wavlm-large",
    "hubert": "facebook/hubert-large-ll60k",
    "xlsr": "facebook/wav2vec2-xls-r-300m",
    "w2vbert": "facebook/w2v-bert-2.0",
    # monolingual twin of xlsr (same wav2vec2-large arch, English-only LV-60k)
    # for the mono-vs-multilingual deep-layer comparison
    "w2v2lv60": "facebook/wav2vec2-large-lv60",
}


def load_audio_16k(path: str, trim_norm: bool = False) -> torch.Tensor:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(axis=1))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if trim_norm:
        # control condition: remove trivial cues (leading/trailing silence, level)
        import librosa

        trimmed, _ = librosa.effects.trim(wav.numpy(), top_db=35)
        wav = torch.from_numpy(trimmed)
        rms = wav.pow(2).mean().sqrt().clamp_min(1e-8)
        wav = wav * (0.05 / rms)
    return wav


@torch.inference_mode()
def embed(model, processor, wav: torch.Tensor, device) -> np.ndarray:
    inputs = processor(wav.numpy(), sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple of (1, T, D), length n_layers+1
    layers = torch.stack([h.squeeze(0).mean(dim=0) for h in out.hidden_states])
    return layers.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/main.jsonl"))
    ap.add_argument("--ssl", default="wavlm", choices=list(SSL_MODELS))
    ap.add_argument("--conditions", nargs="+", required=True,
                    help="'real' and/or system names matching data/generated/<name>")
    ap.add_argument("--trim-norm", action="store_true",
                    help="trim silence + RMS-normalize before embedding (control)")
    ap.add_argument("--gen-root", default=str(ROOT / "data" / "generated"),
                    help="directory holding <condition>/<utt_id>.wav")
    ap.add_argument("--force", action="store_true", help="recompute even if <tag>/<cond>.npz exists")
    ap.add_argument("--out-tag", default=None,
                    help="override output subdir name (default: ssl name)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = SSL_MODELS[args.ssl]

    from transformers import AutoFeatureExtractor, AutoModel

    processor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    rows = [json.loads(l) for l in open(args.manifest)]
    tag = args.out_tag or (args.ssl + ("-norm" if args.trim_norm else ""))
    out_root = ROOT / "results" / "embeddings" / tag
    out_root.mkdir(parents=True, exist_ok=True)

    for cond in args.conditions:
        if (out_root / f"{cond}.npz").exists() and not args.force:
            print(f"{args.ssl}/{cond}: exists, skipped (use --force to recompute)")
            continue
        utt_ids, speakers, embs = [], [], []
        missing = 0
        for r in rows:
            if cond == "real" and args.gen_root.endswith("generated"):
                path = r["real_wav"]
            else:
                path = Path(args.gen_root) / cond / f"{r['utt_id']}.wav"
                if not Path(path).exists():
                    missing += 1
                    continue
            wav = load_audio_16k(str(path), trim_norm=args.trim_norm)
            embs.append(embed(model, processor, wav, device))
            utt_ids.append(r["utt_id"])
            speakers.append(r["speaker"])
        embs = np.stack(embs)
        np.savez(
            out_root / f"{cond}.npz",
            utt_ids=np.array(utt_ids),
            speakers=np.array(speakers),
            embs=embs,
        )
        print(f"{args.ssl}/{cond}: {embs.shape} saved ({missing} missing)")


if __name__ == "__main__":
    main()
