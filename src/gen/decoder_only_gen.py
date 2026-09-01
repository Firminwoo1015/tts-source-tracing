"""Decoder-only estimand: THE controlled vocoder experiment.

One shared log-mel analysis (LJSpeech 22.05 kHz / 80-band convention from the
parallel_wavegan ecosystem) computed ONCE per real utterance, decoded by
multiple pretrained vocoders that were all trained on this same feature
convention. Any separability between the resulting conditions is caused by
the waveform decoder alone (plus its own affine feature normalization),
with the acoustic input held exactly fixed.

Run in the `tts_voc` env. Output: data/generated/voc_<name>/<utt_id>.wav
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[2]))
CKPTS = Path(os.environ.get("TTS_ANAL_CKPTS", ROOT / "ckpts"))
DL = CKPTS / "parallel_wavegan"

TAGS = {
    "voc_pwg": "ljspeech_parallel_wavegan.v1",
    "voc_melgan": "ljspeech_melgan.v3",
    "voc_mbmelgan": "ljspeech_multi_band_melgan.v2",
    "voc_hifigan": "ljspeech_hifigan.v1",
    "voc_stylemelgan": "ljspeech_style_melgan.v1",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/main.jsonl"))
    args = ap.parse_args()

    import librosa
    from parallel_wavegan.utils import download_pretrained_model, load_model
    from parallel_wavegan.bin.preprocess import logmelfilterbank

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [json.loads(l) for l in open(args.manifest)]

    models = {}
    feat_cfg = None
    for cond, tag in TAGS.items():
        try:
            ckpt = download_pretrained_model(tag, str(DL))
            cfg = yaml.safe_load(open(Path(ckpt).parent / "config.yml"))
            model = load_model(ckpt, cfg)
            model.remove_weight_norm()
            # per-model affine feature normalization (part of its own input
            # pipeline); the RAW log-mel input is identical across vocoders
            model.register_stats(str(Path(ckpt).parent / "stats.h5"))
            models[cond] = (model.eval().to(device), cfg)
            if feat_cfg is None:
                feat_cfg = cfg  # shared LJSpeech feature convention
            print("loaded", cond, tag)
        except Exception as e:
            print(f"SKIP {cond} ({tag}): {e}")

    for cond in models:
        (ROOT / "data/generated" / cond).mkdir(parents=True, exist_ok=True)

    sr_t = feat_cfg["sampling_rate"]
    with torch.inference_mode():
        for i, r in enumerate(rows):
            wav, sr = sf.read(r["real_wav"], dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)
            if sr != sr_t:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=sr_t)
            # ONE shared analysis per utterance
            mel = logmelfilterbank(
                wav, sampling_rate=sr_t,
                hop_size=feat_cfg["hop_size"],
                fft_size=feat_cfg["fft_size"],
                win_length=feat_cfg["win_length"],
                window=feat_cfg["window"],
                num_mels=feat_cfg["num_mels"],
                fmin=feat_cfg["fmin"], fmax=feat_cfg["fmax"])
            for cond, (model, cfg) in models.items():
                out = ROOT / "data/generated" / cond / f"{r['utt_id']}.wav"
                if out.exists():
                    continue
                m = torch.from_numpy(mel).float().to(device)
                y = model.inference(m, normalize_before=True).view(-1)
                sf.write(str(out), y.cpu().numpy(), sr_t)
            if (i + 1) % 50 == 0:
                print(f"[{i + 1}/{len(rows)}]")
    for cond in models:
        n = len(list((ROOT / "data/generated" / cond).glob("*.wav")))
        print(cond, n, "wavs")


if __name__ == "__main__":
    main()
