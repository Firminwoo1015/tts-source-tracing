"""Multi-seed subset generation: same 25 utterances (5 speakers x 5 texts),
extra sampling seeds per system, to test stability of attribution under
generation stochasticity. Run inside the system's own conda env.

Output: data/generated_seeds/<system>_s<seed>/<utt_id>.wav
Seed 0 corresponds to the existing main-set outputs.
"""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[2]))
CKPTS = Path(os.environ.get("TTS_ANAL_CKPTS", ROOT / "ckpts"))


def rows(manifest):
    return [json.loads(l) for l in open(manifest)]


def out_dir(system, seed):
    d = ROOT / "data/generated_seeds" / f"{system}_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_seed(seed):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True,  # + cosyvoice3 (tts_cosy), qwen3tts (tts_qwen)
                    choices=["f5tts", "xtts", "cosyvoice2", "chatterbox",
                             "indextts", "cosyvoice3", "qwen3tts"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/seedsub.jsonl"))
    args = ap.parse_args()
    rs = rows(args.manifest)
    od = out_dir(args.system, args.seed)

    if args.system == "f5tts":
        from f5_tts.api import F5TTS

        tts = F5TTS()
        for r in rs:
            out = od / f"{r['utt_id']}.wav"
            if out.exists():
                continue
            tts.infer(ref_file=r["prompt_wav"], ref_text=r["prompt_text"],
                      gen_text=r["text"], file_wave=str(out), seed=args.seed)

    elif args.system == "xtts":
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        os.environ.setdefault("TTS_HOME", os.environ.get("TTS_ANAL_CKPTS", str(Path(__file__).resolve().parents[2] / "ckpts")))
        import torch
        from TTS.api import TTS

        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        for r in rs:
            out = od / f"{r['utt_id']}.wav"
            if out.exists():
                continue
            set_seed(args.seed)
            tts.tts_to_file(text=r["text"], speaker_wav=r["prompt_wav"],
                            language="en", file_path=str(out))

    elif args.system == "cosyvoice2":
        import sys

        sys.path.insert(0, str(ROOT / "third_party/CosyVoice"))
        sys.path.insert(0, str(ROOT / "third_party/CosyVoice/third_party/Matcha-TTS"))
        import torch
        import torchaudio
        from cosyvoice.cli.cosyvoice import CosyVoice2

        cosy = CosyVoice2(str(CKPTS / "CosyVoice2-0.5B"),
                          load_jit=False, load_trt=False, fp16=False)
        for r in rs:
            out = od / f"{r['utt_id']}.wav"
            if out.exists():
                continue
            set_seed(args.seed)
            chunks = [o["tts_speech"] for o in cosy.inference_zero_shot(
                r["text"], r["prompt_text"], r["prompt_wav"], stream=False)]
            torchaudio.save(str(out), torch.cat(chunks, dim=1), cosy.sample_rate)

    elif args.system == "chatterbox":
        import perth
        perth.PerthImplicitWatermarker = perth.DummyWatermarker  # no watermark
        import torchaudio
        from chatterbox.tts import ChatterboxTTS

        model = ChatterboxTTS.from_pretrained(device="cuda")
        for r in rs:
            out = od / f"{r['utt_id']}.wav"
            if out.exists():
                continue
            set_seed(args.seed)
            wav = model.generate(r["text"], audio_prompt_path=r["prompt_wav"])
            torchaudio.save(str(out), wav.cpu(), model.sr)

    elif args.system == "indextts":
        import sys

        sys.path.insert(0, str(ROOT / "third_party/index-tts"))
        from indextts.infer import IndexTTS

        tts = IndexTTS(model_dir=str(CKPTS / "IndexTTS-1.5"),
                       cfg_path=str(CKPTS / "IndexTTS-1.5/config.yaml"))
        for r in rs:
            out = od / f"{r['utt_id']}.wav"
            if out.exists():
                continue
            set_seed(args.seed)
            tts.infer(r["prompt_wav"], r["text"], str(out))

    elif args.system == "cosyvoice3":
        import sys

        sys.path.insert(0, str(ROOT / "third_party/CosyVoice"))
        sys.path.insert(0, str(ROOT / "third_party/CosyVoice/third_party/Matcha-TTS"))
        import torch
        import torchaudio
        from cosyvoice.cli.cosyvoice import CosyVoice3
        base = CKPTS / "hub" / "models--FunAudioLLM--Fun-CosyVoice3-0.5B-2512"
        snap = base / "snapshots" / (base / "refs" / "main").read_text().strip()
        cosy = CosyVoice3(str(snap), load_trt=False, load_vllm=False, fp16=False)
        for r in rs:
            out = od / f"{r['utt_id']}.wav"
            if out.exists():
                continue
            set_seed(args.seed)
            chunks = [o["tts_speech"] for o in cosy.inference_zero_shot(
                r["text"], "You are a helpful assistant.<|endofprompt|>" + r["prompt_text"],
                r["prompt_wav"], stream=False)]
            torchaudio.save(str(out), torch.cat(chunks, dim=1), cosy.sample_rate)

    elif args.system == "qwen3tts":
        import torch
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel

        model = Qwen3TTSModel.from_pretrained(
            os.environ.get("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
            device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
        prompts = {}
        for r in rs:
            out = od / f"{r['utt_id']}.wav"
            if out.exists():
                continue
            key = (r["speaker"], r["prompt_wav"])
            if key not in prompts:
                prompts[key] = model.create_voice_clone_prompt(
                    ref_audio=r["prompt_wav"], ref_text=r["prompt_text"], x_vector_only_mode=False)
            set_seed(args.seed)
            wavs, sr = model.generate_voice_clone(
                text=r["text"], language="English", voice_clone_prompt=prompts[key])
            sf.write(str(out), wavs[0], sr)

    n = len(list(od.glob("*.wav")))
    print(f"{args.system} seed {args.seed}: {n} wavs")


if __name__ == "__main__":
    main()
