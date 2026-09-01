"""Per-file provenance records for every released waveform (E part 2).

Writes provenance_files.jsonl.gz (one JSON object per file) with: relative path, condition, kind,
utt_id/speaker, audio format (sr/channels/subtype/samples), SHA-256 of the file, and for generated
speech the checkpoint identity (HF revision or local checkpoint hash key from provenance.json),
wrapper identity (package version or pinned third-party commit), the inference settings as called
by the wrapper script (library defaults are recorded by name, see src/gen/*), the seed policy,
the enrollment clip and target text hashes, and for perturbed/seed/intervention files the source
and the perturbation/seed. Pure bookkeeping, no model inference.
    python src/make_provenance_files.py            -> provenance_files.jsonl.gz + provenance_files_summary.md
"""
import gzip, hashlib, json, os, sys
from pathlib import Path
import soundfile as sf
ROOT = Path(__file__).resolve().parents[1]
PROV = json.load(open(ROOT / "provenance.json"))
MAN = {r["utt_id"]: r for r in map(json.loads, open(ROOT / "data/manifests/main.jsonl"))}
VCTK = {r["utt_id"]: r for r in map(json.loads, open(ROOT / "data/manifests/vctk.jsonl"))} if (ROOT / "data/manifests/vctk.jsonl").exists() else {}
def sha(p, cache={}):
    p = str(p)
    if p not in cache:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
        cache[p] = h.hexdigest()
    return cache[p]
def ver(freeze, pkg):
    for l in open(ROOT / "provenance" / freeze):
        if l.lower().startswith(pkg.lower() + "=="): return l.strip()
    return None
def hfrev(repo):
    r = PROV["hf_cache_revisions"].get(repo) or PROV["hf_models_used"].get(repo.replace("/", "--"), {}).get("resolved_revision")
    return r if isinstance(r, str) else (r[0] if isinstance(r, list) and r else r)
# ---- system-level metadata (main generation run); settings quote the wrapper call in src/gen/<sys>_gen.py
SYS_META = {
 "f5tts": dict(system="F5-TTS v1", checkpoint={"repo": "SWivid/F5-TTS", "file": "F5TTS_v1_Base/model_1250000.safetensors", "revision": hfrev("SWivid/F5-TTS")},
               wrapper={"package": ver("freeze_tts_anal.txt", "f5-tts"), "script": "src/gen/f5tts_gen.py", "call": "F5TTS().infer(ref_file, ref_text, gen_text, file_wave, seed=0)"},
               settings={"seed": 0, "nfe_step": 32, "cfg_strength": 2.0, "sway_sampling_coef": -1.0, "speed": 1.0, "target_rms": 0.1, "vocoder": "vocos (charactr/vocos-mel-24khz)", "note": "library defaults of f5_tts.api.F5TTS.infer except the explicit seed"},
               seed_policy="fixed library seed 0 for every utterance (deterministic given the environment)", env="tts_anal"),
 "xtts": dict(system="XTTS-v2", checkpoint={"name": "coqui tts_models/multilingual/multi-dataset/xtts_v2", "hash_keys": [k for k in PROV["other_checkpoints"] if "xtts_v2" in k]},
              wrapper={"package": ver("freeze_tts_xtts.txt", "coqui-tts"), "script": "src/gen/xtts_gen.py", "call": "TTS(...).tts_to_file(text, speaker_wav, language='en', file_path)"},
              settings={"temperature": 0.75, "length_penalty": 1.0, "repetition_penalty": 5.0, "top_k": 50, "top_p": 0.85, "gpt_cond_len": 30, "gpt_cond_chunk_len": 4, "max_ref_len": 30, "sound_norm_refs": False, "split_sentences": True, "speed": 1.0, "note": "values of the released checkpoint config.json (tts_models--multilingual--multi-dataset--xtts_v2) and TTS.api.tts_to_file defaults"},
              seed_policy="no per-utterance seed (library default RNG state)", env="tts_xtts"),
 "cosyvoice3": dict(system="Fun-CosyVoice3-0.5B-2512", checkpoint={"repo": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "revision": hfrev("FunAudioLLM/Fun-CosyVoice3-0.5B-2512")},
                    wrapper={"third_party_commit": {"CosyVoice": PROV["third_party_commits"]["CosyVoice"]}, "script": "src/gen/cosyvoice3_gen.py", "call": "CosyVoice3(...).inference_zero_shot(text, 'You are a helpful assistant.<|endofprompt|>'+prompt_text, prompt_wav, stream=False)"},
                    settings={"sampling": "ras_sampling top_p=0.8 top_k=25 win_size=10 tau_r=0.1 (cosyvoice3.yaml)", "speed": 1.0, "stream": False, "fp16": False, "note": "repository YAML defaults; the YAML seeds the RNG to 1986 at model load, not per utterance"},
                    seed_policy="repository default (RNG seeded once at load, then sequential sampling)", env="tts_cosy"),
 "chatterbox": dict(system="Chatterbox", checkpoint={"repo": "ResembleAI/chatterbox", "revision": hfrev("ResembleAI/chatterbox")},
                    wrapper={"package": ver("freeze_tts_chatter.txt", "chatterbox-tts"), "script": "src/gen/chatterbox_gen.py", "call": "ChatterboxTTS.from_pretrained(device='cuda').generate(text, audio_prompt_path=prompt_wav)"},
                    settings={"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8, "repetition_penalty": 1.2, "min_p": 0.05, "top_p": 1.0, "watermark": "disabled (no-op watermarker)", "note": "chatterbox-tts generate() defaults"},
                    seed_policy="no per-utterance seed (library default RNG state)", env="tts_chatter"),
 "indextts": dict(system="IndexTTS-1.5", checkpoint={"repo": "IndexTeam/IndexTTS-1.5", "revision": hfrev("IndexTeam/IndexTTS-1.5"), "hash_keys": [k for k in PROV["checkpoint_hashes"] if k.startswith("IndexTTS-1.5/") and k.endswith((".pth", ".yaml"))]},
                  wrapper={"third_party_commit": {"index-tts": PROV["third_party_commits"]["index-tts"]}, "script": "src/gen/indextts_gen.py", "call": "IndexTTS(model_dir, cfg_path).infer(prompt_wav, text, out)"},
                  settings={"do_sample": True, "top_p": 0.8, "top_k": 30, "temperature": 1.0, "num_beams": 3, "repetition_penalty": 10.0, "max_mel_tokens": 600, "length_penalty": 0.0, "note": "index-tts infer() defaults"},
                  seed_policy="no per-utterance seed (library default RNG state)", env="tts_index"),
 "cosyvoice2": dict(system="CosyVoice2-0.5B (comparison configuration)", checkpoint={"repo": "FunAudioLLM/CosyVoice2-0.5B", "revision": hfrev("FunAudioLLM/CosyVoice2-0.5B")},
                    wrapper={"third_party_commit": {"CosyVoice": PROV["third_party_commits"]["CosyVoice"]}, "script": "src/gen/cosyvoice_gen.py"}, settings={"note": "repository YAML defaults"}, seed_policy="repository default (RNG seeded once at load)", env="tts_cosy"),
 "qwen3tts": dict(system="Qwen3-TTS-12Hz-1.7B-Base (exploratory extension)", checkpoint={"repo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "revision": hfrev("Qwen/Qwen3-TTS-12Hz-1.7B-Base")},
                  wrapper={"package": ver("freeze_tts_qwen.txt", "qwen-tts"), "script": "src/gen/qwen3tts_gen.py"}, settings={"note": "qwen-tts generate defaults"}, seed_policy="no per-utterance seed", env="tts_qwen"),
 "e2tts": dict(system="E2-TTS (descriptive F5-family swap)", checkpoint={"repo": "SWivid/E2-TTS", "revision": hfrev("SWivid/E2-TTS")}, wrapper={"script": "src/gen/f5_variants_gen.py"}, settings={"note": "f5_tts defaults"}, seed_policy="fixed library seed 0", env="tts_anal"),
 "f5tts_base": dict(system="F5-TTS Base (descriptive swap)", checkpoint={"repo": "SWivid/F5-TTS", "file": "F5TTS_Base/model_1200000.safetensors", "revision": hfrev("SWivid/F5-TTS")}, wrapper={"script": "src/gen/f5_variants_gen.py"}, settings={"note": "f5_tts defaults"}, seed_policy="fixed library seed 0", env="tts_anal"),
 "f5tts_bigvgan": dict(system="F5-TTS Base + BigVGAN (descriptive swap)", checkpoint={"repo": "SWivid/F5-TTS", "file": "F5TTS_Base_bigvgan/model_1250000.pt", "revision": hfrev("SWivid/F5-TTS")}, wrapper={"script": "src/gen/f5_variants_gen.py"}, settings={"note": "f5_tts defaults"}, seed_policy="fixed library seed 0", env="tts_anal"),
 "kokoro": dict(system="Kokoro (released comparison only)", checkpoint={"note": "see provenance.json"}, wrapper={"script": "src/gen/kokoro_gen.py"}, settings={}, seed_policy="deterministic", env="tts_anal"),
}
RESYNTH = {
 "resynth_vocos": ("intervention", "real -> Vocos mel analysis -> Vocos decoder (charactr/vocos-mel-24khz)", "src/gen/resynth.py"),
 "resynth_glvocos": ("intervention", "real -> Vocos mel analysis -> Griffin-Lim (no neural decoder)", "src/gen/resynth_glvocos.py"),
 "resynth_griffinlim": ("intervention", "real -> generic log-mel -> Griffin-Lim", "src/gen/resynth_extra.py"),
 "resynth_hift3": ("intervention", "real -> CosyVoice3 acoustic features -> causal HiFT", "src/gen/resynth_cosy3.py"),
 "resynth_s3vc3": ("intervention", "real -> speech_tokenizer_v3 tokens -> DiT flow -> HiFT (same-speaker prompt)", "src/gen/resynth_cosy3.py"),
 "resynth_hift": ("intervention", "real -> CosyVoice2 mel -> HiFT (comparison configuration)", "src/gen/resynth_cosy.py"),
 "resynth_s3vc": ("intervention", "real -> CosyVoice2 token round trip (comparison configuration)", "src/gen/resynth_cosy.py"),
 "resynth_bigvgan": ("intervention", "real -> BigVGAN-v2 24 kHz 100-band (nvidia/bigvgan_v2_24khz_100band_256x)", "src/gen/resynth.py"),
 "resynth_encodec": ("intervention", "real -> EnCodec 24 kHz round trip (unmatched control)", "src/gen/resynth.py"),
 "resynth_dac": ("intervention", "real -> DAC 24 kHz round trip (unmatched control)", "src/gen/resynth.py"),
 "resynth_qwencodec": ("intervention", "real -> Qwen3-TTS-Tokenizer-12Hz round trip (exploratory)", "src/gen/resynth_qwencodec.py"),
 "voc_pwg": ("decoder_only", "shared LJSpeech log-mel -> Parallel WaveGAN (parallel_wavegan ljspeech_parallel_wavegan.v1)", "src/gen/decoder_only_gen.py"),
 "voc_melgan": ("decoder_only", "shared LJSpeech log-mel -> MelGAN (ljspeech_melgan.v3)", "src/gen/decoder_only_gen.py"),
 "voc_mbmelgan": ("decoder_only", "shared LJSpeech log-mel -> Multi-band MelGAN (ljspeech_multi_band_melgan.v2)", "src/gen/decoder_only_gen.py"),
 "voc_hifigan": ("decoder_only", "shared LJSpeech log-mel -> HiFi-GAN (ljspeech_hifigan.v1)", "src/gen/decoder_only_gen.py"),
 "voc_stylemelgan": ("decoder_only", "shared LJSpeech log-mel -> StyleMelGAN (ljspeech_style_melgan.v1)", "src/gen/decoder_only_gen.py"),
}
def audio_info(p):
    i = sf.info(str(p)); return {"sr": i.samplerate, "channels": i.channels, "subtype": i.subtype, "samples": i.frames, "duration_s": round(i.frames / i.samplerate, 3)}
def manifest_fields(utt, man=MAN):
    r = man.get(utt)
    if not r: return {}
    d = {"speaker": r.get("speaker"), "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest(), "real_wav": os.path.relpath(r["real_wav"], "/home/nas5/minwoolee/datasets") if r.get("real_wav") else None}
    if r.get("real_wav") and Path(r["real_wav"]).exists(): d["real_wav_sha256"] = sha(r["real_wav"])
    if r.get("prompt_wav"):
        d["enrollment_wav"] = os.path.relpath(r["prompt_wav"], "/home/nas5/minwoolee/datasets")
        if Path(r["prompt_wav"]).exists(): d["enrollment_wav_sha256"] = sha(r["prompt_wav"])
        if r.get("prompt_text"): d["enrollment_text_sha256"] = hashlib.sha256(r["prompt_text"].encode()).hexdigest()
    return d
SYS = ["f5tts", "xtts", "cosyvoice3", "chatterbox", "indextts"]
PAPER_CONDS = set(SYS + ["resynth_vocos", "resynth_glvocos", "resynth_griffinlim", "resynth_hift3", "resynth_s3vc3",
                         "resynth_bigvgan", "resynth_encodec", "resynth_dac",
                         "voc_pwg", "voc_melgan", "voc_mbmelgan", "voc_hifigan", "voc_stylemelgan"])
PAPER_PERTS = {"common", "common_sym", "mp3_64k", "lp4k", "hp2k", "noise20", "noise20_s2", "noise20_s3",
               "phaserand", "phaserand_s2", "phaserand_s3"}
PAPER_SEEDS = {f"{s}_s{k}" for s in SYS for k in (101, 102, 103)}

def main():
    out = gzip.open(ROOT / "provenance_files.jsonl.gz", "wt"); n = 0; counts = {}
    def emit(rec):
        nonlocal n; out.write(json.dumps(rec, sort_keys=True) + "\n"); n += 1; counts[rec["condition"]] = counts.get(rec["condition"], 0) + 1
    # 1) main generated + interventions + decoder-only + variants
    for d in sorted((ROOT / "data/generated").iterdir()):
        if not d.is_dir(): continue
        cond = d.name
        if cond not in PAPER_CONDS: continue
        for w in sorted(d.glob("*.wav")):
            utt = w.stem; rec = {"path": str(w.relative_to(ROOT)), "condition": cond, "utt_id": utt, "sha256": sha(w), **audio_info(w), **manifest_fields(utt)}
            if cond in SYS_META:
                s = SYS_META[cond]; rec.update({"kind": "generated", "system": s["system"], "checkpoint": s["checkpoint"], "wrapper": s["wrapper"], "inference_settings": s["settings"], "seed_policy": s["seed_policy"], "conda_env": s["env"], "postprocessing": "none (written by the wrapper at its native rate)"})
            elif cond in RESYNTH:
                k, desc, scr = RESYNTH[cond]; rec.update({"kind": k, "path_description": desc, "script": scr, "input": "real LibriSpeech utterance (see real_wav)", "postprocessing": "none"})
            else: rec.update({"kind": "other"})
            emit(rec)
    # 2) controlled-seed subsets
    for d in sorted((ROOT / "data/generated_seeds").iterdir()):
        if not d.is_dir() or d.name not in PAPER_SEEDS: continue
        sysn, seed = d.name.rsplit("_s", 1); s = SYS_META.get(sysn, {})
        for w in sorted(d.glob("*.wav")):
            utt = w.stem
            emit({"path": str(w.relative_to(ROOT)), "condition": d.name, "kind": "generated_controlled_seed", "utt_id": utt, "sha256": sha(w), **audio_info(w), **manifest_fields(utt),
                  "system": s.get("system", sysn), "checkpoint": s.get("checkpoint"), "wrapper": {**s.get("wrapper", {}), "script": "src/gen/multiseed_gen.py"}, "inference_settings": s.get("settings"),
                  "seed_policy": f"controlled wrapper seed {seed} (random/np/torch/cuda seeded before each utterance; F5-TTS: infer(seed={seed}))", "conda_env": s.get("env"), "postprocessing": "none",
                  "note": "seeds 101-103: 20 speakers x 3 utterances (data/manifests/seedsub20.jsonl); seeds 1-2: earlier 5-speaker subset (seedsub.jsonl); same checkpoints/wrappers/environments as the main run"})
    # 3) VCTK subset
    for d in []:  # VCTK subset is not part of the paper release
        if not d.is_dir(): continue
        s = SYS_META.get(d.name, {})
        for w in sorted(d.glob("*.wav")):
            emit({"path": str(w.relative_to(ROOT)), "condition": f"vctk/{d.name}", "kind": "generated_vctk", "utt_id": w.stem, "sha256": sha(w), **audio_info(w), **manifest_fields(w.stem, VCTK),
                  "system": s.get("system", d.name), "checkpoint": s.get("checkpoint"), "wrapper": s.get("wrapper"), "inference_settings": s.get("settings"), "seed_policy": s.get("seed_policy"), "postprocessing": "none"})
    # 4) perturbed
    for d in sorted((ROOT / "data/perturbed").iterdir()):
        if not d.is_dir(): continue
        if d.name not in PAPER_PERTS: continue
        pert, _, suf = d.name.partition("_s"); pseed = int(suf) if suf.isdigit() else 0
        if not suf.isdigit(): pert = d.name
        for c in sorted(d.iterdir()):
            if not c.is_dir() or c.name not in set(SYS) | {"real"}: continue
            for w in sorted(c.glob("*.wav")):
                utt = w.stem; src = (ROOT / "data/generated" / c.name / f"{utt}.wav") if c.name != "real" else Path(MAN.get(utt, {}).get("real_wav", ""))
                emit({"path": str(w.relative_to(ROOT)), "condition": f"perturbed/{d.name}/{c.name}", "kind": "perturbed", "utt_id": utt, "sha256": sha(w), **audio_info(w), "speaker": MAN.get(utt, {}).get("speaker"),
                      "perturbation": pert, "perturbation_seed": pseed, "script": "src/gen/perturb.py", "source_condition": c.name, "source_path": str(src.relative_to(ROOT)) if str(src).startswith(str(ROOT)) else (os.path.relpath(str(src), "/home/nas5/minwoolee/datasets") if src and str(src) else None),
                      "source_sha256": sha(src) if src and Path(src).exists() else None, "postprocessing": "perturbation applied at 16 kHz after the common load/resample step (see perturb.py)"})
    out.close()
    with open(ROOT / "provenance_files_summary.md", "w") as f:
        f.write("# provenance_files.jsonl.gz\n\nPer-file provenance records (one JSON object per released waveform, `python src/make_provenance_files.py`).\n\n")
        f.write(f"Total records: {n}\n\n| condition | files |\n|---|---|\n")
        for k in sorted(counts): f.write(f"| {k} | {counts[k]} |\n")
        f.write("\nFields: path, condition, kind, utt_id, speaker, sha256, sr, channels, subtype, samples, duration_s; generated: system, checkpoint (HF revision / checkpoint hash keys into provenance.json), wrapper (package version or pinned third-party commit, script, call), inference_settings (as called; library defaults recorded by name), seed_policy, conda_env, text_sha256, enrollment_wav(+sha256), enrollment_text_sha256, real_wav(+sha256); interventions/decoder-only: path_description, script; perturbed: perturbation, perturbation_seed, source_condition, source_path, source_sha256.\n")
    print("records:", n); print(json.dumps(counts, indent=0)[:1500])
if __name__ == "__main__":
    main()
