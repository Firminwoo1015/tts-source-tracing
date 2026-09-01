# Where Modern Zero-Shot TTS Leaves Its Trace: Localizing Source-Tracing Cues in Self-Supervised Speech Representations

Minwoo Lee, Jaegul Choo (KAIST AI). Manuscript under review (submitted to ICASSP 2027); please cite the preprint once available.

A paired, speaker- and text-disjoint source-tracing benchmark of five open zero-shot TTS systems
(F5-TTS v1, XTTS-v2, Fun-CosyVoice3, Chatterbox, IndexTTS-1.5; all native 24 kHz) on LibriSpeech
`test-clean` (39 speakers × 10 targets, 17-ID duration-QC mask → 373 paired IDs, 2,238 waveforms),
probed layer-wise with five frozen SSL encoders (WavLM-Large, HuBERT-Large, XLS-R 300M,
wav2vec2-LV60, w2v-BERT 2.0). This repository holds everything the paper says is released:
the manifests, the generation and analysis code, checkpoint/wrapper/output provenance, and every
number behind the tables and figures. Audio and embeddings are in the data bundle (Section 4).

## 1. Layout

```
data/manifests/        main.jsonl (390 target utterances, 39 speakers, enrollment clip per speaker),
                       exclude17.txt (17 IDs removed by the duration rule), speaker_folds.json (outer
                       speaker-disjoint folds), seedsub20.jsonl (controlled-seed subset: 20 speakers × 3)
src/gen/               generation and intervention scripts (one conda env per TTS system, see below)
src/                   embedding extraction, ASR, analyses, figures, provenance builders
results/paper5c17/     every table/figure input of the paper (CSV), fig2_*.pdf, PAPER_NUMBERS2.json
provenance.json        checkpoint revisions and per-file SHA-256, third-party commits, licenses
provenance_files.jsonl.gz  one record per released waveform (SHA-256, format, checkpoint, wrapper
                       version/commit, inference settings, seed policy, enrollment/text hashes)
provenance/            conda env exports and pip freezes of every environment
analysis_plan.md       pre-specified protocol and interpretation rules of the added controls
                       (A1/A2/B/C/D/N), their outcomes, and the post-review amendment (G_min)
DATA_LICENSES.md, NOTICE, LICENSE   per-source terms (corpora, models, our code = MIT)
```

## 2. Claim → file map (all under `results/paper5c17/`)

| paper item | script | file(s) |
|---|---|---|
| Table 1 (speaker-disjoint macro-F1, 5 encoders + baselines), §4.1 per-class recall | `analyze_cloning.py`, `analyze_mfcc_cloning.py`, `analyze_simplecues.py`, `analyze_asr.py` | `cloning_summary_{ssl}.csv`, `cloning_ttsonly_{perclass,confusion}_wavlm.csv`, `mfcc_cloning.csv`, `simplecues.csv`, `asr_feature_baseline.csv` |
| §4.1 trim+RMS nuisance control (paired difference CIs) | `analyze_normalization_control.py` | `normalization_control.csv` |
| §4.1 controlled-seed study | `analyze_seeds2.py` | `seeds2_{transfer,geometry,stochasticity}.csv` |
| Fig. 1, Table 2 (ΔP_t, S_t, T_t, G_t, G_adj and the strongest-control gap G_min at 10,000 replicates) | `analyze_transplant2.py`, `analyze_centroid2.py`, `plot_figures2.py` | `intervention_wavlm.csv`, `centroid2_wavlm_L0.csv` (rows `*_vs_unmatched`, `*_vs_strongest`, `<control>_under_<target>`; Griffin–Lim `T_lo/T_hi`), `fig2_intervention.pdf` |
| Fig. 2 (layer-wise ΔP_t), §4.2 w2v-BERT L19 | `analyze_transplant2.py --ssl w2vbert`, `analyze_centroid2.py --ssl w2vbert --layer 19` | `intervention_w2vbert.csv`, `centroid2_w2vbert_L19.csv`, `fig2_interv_layers.pdf` |
| §4.2 decoder-only control | `analyze_deconly.py` | `deconly_wavlm.csv`, `voc_vs_reference_wavlm.csv` |
| Table 3 (early/deep bands), alternative bands | `analyze_bands.py`, `analyze_spkdisjoint_extras.py` | `bands_spkdisjoint.csv`, `bands_alt.csv` |
| §4.3 logistic / centroid / kNN deep-band probes | `analyze_deep_probe.py` | `deep_probe.csv` |
| §4.3 paired-kNN mechanism (LOO, leave-utterance/speaker-out, same-utterance fraction, ties) | `analyze_knn_mechanism.py` | `knn_mechanism.csv` |
| §4.3 Whisper WER and WER=0 subset | `run_asr.py`, `analyze_asr.py` | `asr_wer.csv`, `asr_summary.csv`, `asr_perfect_deepband.csv` |
| Table 4 (cross-condition stress test; w2v-BERT released) | `analyze_robustB3.py` | `robustB3_{wavlm,w2vbert}.csv` |
| every number quoted in the text | `paper_numbers.py` | `PAPER_NUMBERS2.json` |

Protocol in one paragraph: outer speaker-disjoint 5-fold GroupKFold with nested inner
GroupKFold(4) layer selection; cosine kNN (k=5, majority vote, ties to the first class in sorted
label order) on L2-normalised mean-pooled layer embeddings; macro-F1 with 1,000 speaker-level
bootstrap replicates (10,000 for the strongest-control rows); interventions are speaker-disjoint and
their targets are fixed from the architecture. Baselines use the same folds and kNN. Full definitions
of the intervention metrics and the stress-test operators are in the paper (§3) and in
`analysis_plan.md`.

## 3. Reproducing

**Analyses and figures from the released embeddings** (CPU, conda env `tts_anal`, ~2 h):

```bash
cp env.example.sh env.sh && source env.sh       # paths: TTS_ANAL_CKPTS, TTS_ANAL_DATASETS (LibriSpeech test-clean)
tar -xf embeddings.tar                          # from the data bundle -> results/embeddings/<tag>/<cond>.npz
bash src/run_all_analyses.sh                    # rewrites results/paper5c17/*.csv, fig2_*.pdf, PAPER_NUMBERS2.json
```

**Full pipeline** (generation → QC → interventions → perturbations → decoder-only → ASR → embeddings):
`src/make_manifest.py --name main --n-speakers 40 --n-utts 10 --seed 0` (speakers with ≥ 11 utterances
of 3–10 s qualify, seeded shuffle picks enrollment + 10 targets; 39 of 40 qualify), then `src/gen/<system>_gen.py`
in the system's env (`tts_anal` for F5-TTS, `tts_xtts`, `tts_cosy` for CosyVoice3, `tts_chatter`, `tts_index`;
`provenance/` has the exact environments), `src/gen/resynth*.py` and `src/gen/decoder_only_gen.py` (env `tts_voc`)
for the interventions, `src/gen/perturb.py` (seeds 0, 2, 3 for the stochastic perturbations),
`src/gen/multiseed_gen.py --seed {101,102,103} --manifest data/manifests/seedsub20.jsonl`,
`src/run_asr.py`, and `src/extract_embeddings.py` (tags `{ssl}`, `{ssl}_p_<pert>[_s2,_s3]`, `{ssl}_tn`
with `--trim-norm`, `{ssl}_seeds20`). Generated audio is not bit-exactly reproducible (each wrapper's native RNG
policy, documented per file in `provenance_files.jsonl.gz`), which is why the embeddings are released.
The manifests store LibriSpeech paths of the original machine; `src/relocate_manifests.py --old <prefix>` rewrites them.

## 4. Data bundle (audio + embeddings)

https://huggingface.co/datasets/firminumanu/tts-source-tracing-data (Hugging Face Dataset, gated for research use: accept the research-use terms, then `hf download firminumanu/tts-source-tracing-data --repo-type dataset --local-dir .`). The bundle covers exactly the paper's scope: the five systems, the interventions, the decoder-only vocoders, the seven perturbations (three realizations for the stochastic ones), the controlled-seed subsets, and the embeddings of those conditions. Archives and SHA-256 sums:

```
ab720c3e736880f707e5442492e5759d9499d53de914fe46285a0291590a27cd  audio_generated.tar  (2.28 GB)
0617b935972bc586ff7912ffe191991e86360beeb60a1630c8a74abeb58c1d2e  audio_perturbed.tar  (4.45 GB)
3d687c43d2f77ea5b15377a98636aa88743dd300eeac99f4702ead6ff5501c92  audio_subsets.tar  (0.31 GB)
6057f6921b9d78dc38c1c5c4635750269c0eadcb8f2bc1fea4d18688be6ae3a2  embeddings.tar  (8.26 GB)
49a25202588f6449bcd75b16dce3143cef1695c02e5b5c06c66eb7fe88d9c23d  manifests_and_results.tar  (5.3 MB)
```

`sha256sum -c SHA256SUMS`; `tar -xf <archive>.tar` from this directory. LibriSpeech/VCTK originals are not
redistributed (CC BY 4.0 derivatives are); outputs of F5-TTS, E2-TTS, XTTS-v2 and EnCodec are
non-commercial research use only. See `DATA_LICENSES.md`.

## 5. Citation

```
@unpublished{lee2026trace,
  title={Where Modern Zero-Shot {TTS} Leaves Its Trace: Localizing Source-Tracing Cues in Self-Supervised Speech Representations},
  author={Lee, Minwoo and Choo, Jaegul}, year={2026}, note={Under review}}
```
