# Data licenses and attributions

## Source corpus (not redistributed as original files)
* **LibriSpeech** (Panayotov et al., ICASSP 2015), `test-clean` subset. License: CC BY 4.0.
  Used as source of real speech, enrollment prompts and target texts. `audio_perturbed.tar`
  contains channel-perturbed *derivatives* of LibriSpeech `test-clean` utterances (band-pass
  filtering, phase randomization, MP3 64 kbps, 20 dB noise, resampler chains). These derivatives
  are redistributed under CC BY 4.0 with attribution to LibriSpeech; they are modified versions
  of the originals.

## Generated speech (our derivative works) and the models behind it

Model licenses (from the official model cards / repository LICENSE files at the revisions used) and
the conditions we attach to the *outputs* we release. Model licenses govern the weights; they do not
automatically transfer to generated audio, so the output column states what we apply. Where a model
license restricts outputs (XTTS-v2 CPML) or is non-commercial (F5-TTS, EnCodec weights), the
corresponding released audio is **non-commercial research use only**; all other generated audio,
perturbations and embeddings are released for research on speech deepfake detection and source
tracing under the terms below. Voices are clones of public read-speech corpora; do not use them to
impersonate anyone. This is not legal advice; institutional review applies before publication.

| system / model | license (source) | revision / location | URL | output / usage note |
|---|---|---|---|---|
| F5-TTS v1 (SWivid/F5-TTS) | cc-by-nc-4.0 | 84e5a410d9ce | https://huggingface.co/SWivid/F5-TTS | model card: CC BY-NC 4.0 (non-commercial); outputs released here for non-commercial research only |
| XTTS-v2 (coqui tts_models/multilingual/multi-dataset/xtts_v2) | Coqui Public Model License (CPML) | coqui TTS_HOME cache (hashes in provenance.json) | https://coqui.ai/cpml | CPML restricts the model AND its outputs to non-commercial use; XTTS outputs released for non-commercial research only |
| Fun-CosyVoice3-0.5B-2512 (FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | apache-2.0 | 29e01c4e8d00 | https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 | as above |
| Chatterbox (ResembleAI/chatterbox) | mit | 5bb1f6ee58e5 | https://huggingface.co/ResembleAI/chatterbox | MIT model/code; PerTh watermarker disabled at generation (documented); outputs released for source-tracing research |
| IndexTTS-1.5 (IndexTeam/IndexTTS-1.5) | apache-2.0 (model card); code repo index-tts: bilibili Model Use License Agreement | 25851a6036df | https://huggingface.co/IndexTeam/IndexTTS-1.5 | check the bilibili Model Use License for output/usage conditions before any non-research use |
| Vocos mel-24kHz (charactr/vocos-mel-24khz; F5 decoder + resynth_vocos/glvocos) | mit | 0feb3fdd929b | https://huggingface.co/charactr/vocos-mel-24khz |  |
| BigVGAN-v2 24 kHz (nvidia/bigvgan_v2_24khz_100band_256x; resynth_bigvgan) | mit | c329ede9e9bb | https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x | code patch in third_party/patches/bigvgan.patch (MIT) |
| EnCodec 24 kHz (facebook/encodec_24khz; resynth_encodec) | no license field on the model card; upstream encodec weights are CC BY-NC 4.0 (code MIT) | c1dbe2ae3f1d | https://huggingface.co/facebook/encodec_24khz | resynth_encodec outputs: non-commercial research only |
| DAC 24 kHz (descript-audio-codec; resynth_dac) | MIT (code and weights per repository) | descript cache (hashes in provenance.json) | https://github.com/descriptinc/descript-audio-codec |  |
| parallel_wavegan LJSpeech vocoders (decoder-only experiment) | MIT (kan-bayashi/ParallelWaveGAN) | files hashed in provenance.json | https://github.com/kan-bayashi/ParallelWaveGAN |  |
| WavLM-Large (microsoft/wavlm-large) | no license field on the model card; upstream repo MIT | c1423ed94bb0 | https://huggingface.co/microsoft/wavlm-large | embeddings only |
| HuBERT-Large (facebook/hubert-large-ll60k) | apache-2.0 | ff022d095678 | https://huggingface.co/facebook/hubert-large-ll60k | embeddings only |
| XLS-R 300M (facebook/wav2vec2-xls-r-300m) | apache-2.0 | 1a640f32ac3e | https://huggingface.co/facebook/wav2vec2-xls-r-300m | embeddings only |
| wav2vec2-LV60 (facebook/wav2vec2-large-lv60) | apache-2.0 | 0cde644b64da | https://huggingface.co/facebook/wav2vec2-large-lv60 | embeddings only |
| w2v-BERT 2.0 (facebook/w2v-bert-2.0) | mit | da985ba0987f | https://huggingface.co/facebook/w2v-bert-2.0 | embeddings only |
| faster-whisper large-v3 (Systran/faster-whisper-large-v3; ASR control) | mit | edaa852ec7e1 | https://huggingface.co/Systran/faster-whisper-large-v3 | transcripts only |

HF Dataset metadata for the bundle should therefore use `license: other` with this per-file-group
table rather than a single license identifier.

## Third-party code
See `third_party/README.md` (upstream commits; BigVGAN patch) and `provenance/` for environments.
