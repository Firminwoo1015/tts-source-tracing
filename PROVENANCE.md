# Provenance (reconstructed post hoc; see Codex audit 2026-08)

- third_party commits: {"BigVGAN": "7d2b454564a6c7d014227f635b7423881f14bdac", "CosyVoice": "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc", "index-tts": "13495845e3028f0bb6ca1462ad22aa0e76349e40"}
- HF cache revisions: 48 models (provenance.json)
- checkpoint hashes: 21 files (provenance.json)
- env freezes: provenance/freeze_<env>.txt (7 envs)
- HF models used: 17 revision IDs recorded; snapshot-backed per-file hashes for 15 repositories (92 files); repos whose weights live in local directories (CosyVoice2-0.5B, IndexTTS-1.5) are hashed separately under checkpoint_hashes (21 files)
- non-HF checkpoints (XTTS/DAC/ParallelWaveGAN, AppleDouble files excluded): 20 files hashed
- BigVGAN local patch recorded as git diff (bigvgan_local_patch)
- NOTE: the main run had no uniform per-utterance seeding protocol (F5 family seed=0; CosyVoice2/3 YAMLs fix an RNG seed at load time; XTTS/Chatterbox/IndexTTS/Qwen3-TTS wrappers set no seed); multi-seed subset used seeds 1,2 with torch/np/random seeding.
- Chatterbox: PerTh watermarker replaced by DummyWatermarker at generation (no audio watermark in outputs).
