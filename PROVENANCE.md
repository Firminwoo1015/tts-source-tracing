# Provenance

Reconstructed from the files on disk after the runs (not logged at generation time). Per-file records: provenance_files.jsonl.gz.

- third_party commits: {"BigVGAN": "7d2b454564a6c7d014227f635b7423881f14bdac", "CosyVoice": "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc", "index-tts": "13495845e3028f0bb6ca1462ad22aa0e76349e40"}
- HF cache revisions: 13 models (provenance.json)
- checkpoint hashes: 6 files (provenance.json)
- env freezes: provenance/freeze_<env>.txt (6 envs)
- HF models used: 13 revision IDs recorded; snapshot-backed per-file hashes for 12 repositories (72 files); repos whose weights live in local directories (IndexTTS-1.5) are hashed separately under checkpoint_hashes (6 files)
- non-HF checkpoints (XTTS/DAC/ParallelWaveGAN, AppleDouble files excluded): 20 files hashed
- BigVGAN local patch recorded as git diff (bigvgan_local_patch)
- Seed policy of the main run: each wrapper's native policy (F5-TTS seed 0; the CosyVoice3 YAML seeds its RNG once at load; XTTS/Chatterbox/IndexTTS set no seed). The controlled-seed subsets (seeds 101-103) seed random/numpy/torch before every utterance.
- Chatterbox: PerTh watermarker replaced by DummyWatermarker at generation (no audio watermark in outputs).
