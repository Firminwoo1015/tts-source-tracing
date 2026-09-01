# third_party/ — upstream repositories (NOT vendored in the release)

These clones are excluded by .gitignore. Re-create them at the exact commits used
(recorded here and in ../provenance.json -> third_party_commits), then apply our patch:

| repo | upstream | commit | code license (LICENSE file at that commit) |
|---|---|---|---|
| BigVGAN   | https://github.com/NVIDIA/BigVGAN.git        | 7d2b454564a6c7d014227f635b7423881f14bdac | MIT |
| CosyVoice | https://github.com/FunAudioLLM/CosyVoice.git | 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc (clone with --recursive; its Matcha-TTS submodule ships a dangling `data` symlink upstream — harmless) | Apache-2.0 |
| index-tts | https://github.com/index-tts/index-tts.git   | 13495845e3028f0bb6ca1462ad22aa0e76349e40 | bilibili Model Use License Agreement (repo LICENSE); the IndexTTS-1.5 weights' model card says apache-2.0 |

```
cd third_party
git clone https://github.com/NVIDIA/BigVGAN.git        && git -C BigVGAN   checkout 7d2b454564a6c7d014227f635b7423881f14bdac
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git && git -C CosyVoice checkout 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
git -C CosyVoice submodule update --init --recursive   # pin Matcha-TTS to the submodule pointer of that commit
git clone https://github.com/index-tts/index-tts.git   && git -C index-tts checkout 13495845e3028f0bb6ca1462ad22aa0e76349e40
git -C BigVGAN apply ../patches/bigvgan.patch   # hub-mixin loading fix used for the BigVGAN-v2 intervention
```

Only BigVGAN is patched (patches/bigvgan.patch, also stored verbatim in provenance.json
-> bigvgan_local_patch); CosyVoice and index-tts are used unmodified.
