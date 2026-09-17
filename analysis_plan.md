# Pre-specified analysis plan for the added controls

This file fixes the protocol and the interpretation rules of the control analyses that were added
after the first complete draft, so that their wording could not be adapted to their outcomes. It was
written before any of them was run; the outcomes were appended afterwards. Everything else in the paper (Table 1, the interventions, the bands) predates
this plan and is not covered by it.

## Common settings
* Data: the paired LibriSpeech corpus, five systems (F5-TTS v1, XTTS-v2, Fun-CosyVoice3, Chatterbox,
  IndexTTS-1.5) plus `real`, QC mask `data/manifests/exclude17.txt` (373 IDs, 2,238 waveforms).
* Splits: outer speaker-disjoint GroupKFold(5) as in `analyze_cloning.py`; inner GroupKFold(4) on the
  training speakers for any layer or hyper-parameter selection.
* Metric: macro-F1 (6-way and TTS-only); CIs from 1,000 speaker-level bootstrap replicates of the pooled
  outer predictions (layer selection is not re-run per replicate; stated in the paper).
* Wording: "the unadjusted 95% CI excludes zero", never "significant".

## A1. Deep-layer decodability (`analyze_deep_probe.py` → `deep_probe.csv`)
* Bands early = layers 1–5, deep = 17–21 (alternative deep bands 20–24 and 13–24 as sensitivity), nested
  layer selection within the band. Classifiers on L2-normalized mean-pooled embeddings: cosine kNN k=5,
  multinomial logistic regression (L2, lbfgs, C ∈ {0.1, 1, 10} by inner CV), cosine nearest centroid.
* Rule: all three high → "deep source information is decodable across classifier families"; kNN high but
  linear/centroid low → "present mainly in local geometry"; all low → the claim is restricted to kNN.
* **Outcome:** deep-band TTS-only macro-F1 kNN 0.69–0.81, logistic regression 0.88–0.94, nearest centroid
  0.71–0.85 for all five encoders → first rule. The early−deep gap persists under logistic regression
  (WavLM 0.96 vs 0.90), so the paper does not attribute the gap to kNN.

## A2. Paired-reference kNN mechanism (`analyze_knn_mechanism.py` → `knn_mechanism.csv`)
* WavLM and HuBERT, layers 17/19/21, TTS-only; protocols ordinary LOO, leave-utterance-out,
  leave-speaker-out; k ∈ {1,3,5,7,11}; uniform and inverse-distance voting; statistics: fraction of the
  top-k LOO neighbours sharing the query's utterance ID, uniform-vote tie rate.
* Rule: attribute the LOO collapse to paired same-utterance distractors only if that fraction exceeds 0.5
  at k=5 and leave-utterance-out restores macro-F1 across k and voting schemes.
* **Outcome:** WavLM L19 LOO 0.03 (k=1) / 0.25 (k=5); same-utterance fraction 0.98 (k=1), 0.71 (top-5);
  tie rate 0.54; leave-utterance-out 0.76–0.83, leave-speaker-out 0.63–0.74; distance weighting produced
  no ties and changed the restored values by ≤ 0.04; HuBERT alike → rule satisfied.

## B. Cross-condition stress test (`analyze_robustB3.py` → `robustB3_{wavlm,w2vbert}.csv`)
* Per perturbation, the same perturbed query files under clean→clean, perturbed→perturbed and
  clean→perturbed on the outer folds; layer chosen per fold on the clean training speakers and kept fixed;
  three realizations for the stochastic perturbations (metrics averaged, realization-specific predictions
  held fixed in the speaker-paired bootstrap of matched − mismatched).
* Rule: keep the stress test if the matched − mismatched CI excludes zero for the severe perturbations;
  keep the caveat that matched performance cannot distinguish a surviving trace from
  transformation-induced artifacts.
* **Outcome:** WavLM L4 in every fold; C→C 0.72, P→P 0.52–0.74, C→P 0.21–0.23 for high-pass, noise and
  phase randomization; all seven differences exclude zero → kept.

## C. Metric definitions (no new data)
* Signed target margin S_t = ΔP_t − max_{j≠t} ΔP_j over the TTS classes; translation T_t = −Δd_t;
  relative alignment G_t = mean_{j≠t} Δd_j − Δd_t; codec-adjusted G_adj = G_t − mean over the two
  unmatched codecs (EnCodec, DAC). Off-target controls are scored under the target of the matched probe
  they are compared with; targets are fixed from the architecture, never from the outcome.
* The strongest-control gap G_min = G_matched − max over {EnCodec, DAC, BigVGAN-v2} (maximum recomputed
  inside each of 10,000 speaker-bootstrap replicates) is a sensitivity analysis added after this plan;
  G_adj above is the pre-specified metric.

## D. Generation-seed sensitivity (`analyze_seeds2.py` → `seeds2_*.csv`)
* 20 speakers (4 per outer fold) × 3 utterances × wrapper seeds 101–103 × 5 systems, same checkpoints,
  wrappers and settings as the main run (verified beforehand). Reference = original-run embeddings of the
  outer-training speakers with the layer chosen on the original run; queries = the controlled-seed
  outputs of the fold's test speakers; TTS-only macro-F1 with speaker-paired bootstrap CIs; geometry =
  same-system cross-seed distance vs cross-system same-utterance distance.
* Rule: "robust to generation stochasticity" if original→seed macro-F1 is within 0.05 of
  original→original and the distance ratio is clearly below one; otherwise the limitation is quantified.
* **Outcome:** no two seeds gave identical waveforms. Macro-F1 changes (regenerated minus original)
  range from −0.0320 to +0.0532 (−0.03 to +0.05 at paper precision), with no decrease whose CI
  excludes zero. Distance ratios are 0.20–0.40. The largest increase slightly exceeds the stated 0.05
  bound, so the rule is not strictly satisfied for every tested seed. The paper reports the observed
  sensitivity without claiming equivalence.

## N. Nuisance control (`analyze_normalization_control.py` → `normalization_control.csv`)
* Embeddings re-extracted after silence trimming (librosa, top_db=35) and per-utterance RMS normalization
  to 0.05, WavLM and w2v-BERT, all six conditions, same protocol as Table 1.
* Rule: "cues do not depend on level/silence conventions" only if macro-F1 stays within 0.05; otherwise
  the dependence is quantified.
* **Outcome:** six-way macro-F1 fell by 0.08 (WavLM) and 0.09 (w2v-BERT), paired CIs [0.06, 0.10] and
  [0.06, 0.13]; the selected w2v-BERT layer moved from L3 to L18 in every fold → the paper reports the
  dependence rather than invariance.

## N2. Nuisance control on the RQ1 interventions (`analyze_transplant2.py --ssl wavlm_tn`, `analyze_centroid2.py --ssl wavlm_tn --layer 0`)
* Added after the plan above, in response to external review. Same intervention design as C, but every
  condition (support set containing real speech and outputs from all five TTS systems, clean real
  queries, resynthesized queries) is embedded from audio that was silence-trimmed (librosa, top_db=35) and RMS-normalized to 0.05, i.e. the tag
  `wavlm_tn`. WavLM L0, same probes, targets and bootstrap as the raw-condition run.
* Rule: report the effect as convention-independent only if every matched path keeps ΔP_t above zero;
  report any change in the control comparison separately rather than replacing the raw-condition numbers.
* **Outcome:** all three matched paths retain positive target prediction-rate shifts after trim+RMS:
  Vocos .49→.55, HiFT .43→.38, and the token round trip .49→.55. Their normalized G_min values are
  3.8 [2.2,5.4], 0.6 [0.3,0.9], and 3.7 [3.1,4.3], respectively, in units of 10⁻³ (raw: 1.2 [−0.002,2.5],
  0.9 [0.7,1.1], 1.8 [1.5,2.2]). All three normalized G_min CIs exclude zero, compared with two in the
  raw condition. This preprocessing changes the support and centroids as well as the queries, so the gap
  changes alone do not identify their cause. The other interventions also keep ΔP_t above zero
  (GL-Vocos .64→.73, GL-generic .62→.73, BigVGAN .08→.08). The paper reports both conditions.

## N3. Chatterbox matched paths (`src/gen/resynth_chatterbox.py`, `analyze_transplant2.py`, `analyze_centroid2.py`, `analyze_lineage.py`)
* Added on 2026-09-17, after the rest of the paper was final, to test a third system whose decoder takes
  representations that can be computed from real speech. XTTS-v2 and IndexTTS-1.5 decode latents of a
  text-conditioned GPT and have no such entry point. The specification below was fixed before any
  Chatterbox resynthesis existed.
* Paths, mirroring the two CosyVoice3 paths: `resynth_hiftcb` (real speech → Chatterbox S3Gen mel
  extractor, 24 kHz/80 mel/hop 480 → HiFTGenerator) and `resynth_s3vccb` (`ChatterboxVC`: S3TokenizerV2 →
  flow → HiFT, conditioned on the same speaker's enrollment prompt). Checkpoint `ResembleAI/chatterbox`
  revision 5bb1f6ee, the file that produced the Chatterbox class, watermark disabled as in
  `chatterbox_gen.py`. Hypothesized target `chatterbox`. Same kNN, centroid, bootstrap and control set
  as C and N2.
* Rules: (R1) the WavLM L0 ΔP_t CI lies above zero; (R2) the G_adj CI lies above zero, with G_min reported
  as in the paper but not required; (R3) a path is read as lineage-level rather than Chatterbox-specific
  when its ΔP toward CosyVoice3 is at least its ΔP toward Chatterbox or its S_t CI includes zero;
  secondary, the own-target versus sibling-system ΔP and G for the four CosyVoice3 and Chatterbox paths.
  Every result is reported whatever its direction.
* Integration: both analysis scripts drew all bootstrap replicates from one sequential RNG, so simply
  adding the two paths moved CI endpoints of existing rows by Monte Carlo noise, and one reported bound
  would have changed (token round trip normalized G_min lower bound 3.150 → 3.156, i.e. 3.1 → 3.2). The
  added paths and the rows under the Chatterbox target therefore draw from a separate stream
  (`RNG_ADDED`), and every previously released row of the six intervention CSVs is reproduced exactly.
  The released `centroid2_w2vbert_L19.csv` did not match the released script before this change, in CI
  endpoints only (point estimates identical, and the paper cites only a point estimate from it). It is
  now regenerated from the script.
* **Outcome:** R1 is met for both paths, ΔP_t .33 [.25,.41] (HiFT) and .32 [.24,.41] (token round trip).
  R2 is met for both, G_adj 1.8 [1.6,2.0] and 3.1 [2.6,3.6] ×10⁻³, and G_min also excludes zero,
  1.4 [1.2,1.6] and 2.7 [2.1,3.2] raw, 1.3 [1.0,1.6] and 3.9 [3.2,4.7] after trim+RMS. Under R3 the HiFT
  path keeps a positive target margin, S_t .15 [.03,.25], with a .18 shift toward CosyVoice3, while the
  token round trip does not resolve Chatterbox over CosyVoice3, S_t .10 [−.07,.27], with a .22 shift
  toward CosyVoice3. The CosyVoice3 paths shift toward Chatterbox by only .08 and .09. The own-minus-
  sibling G difference excludes zero for all four paths. The paper describes the HiFT path as
  target-preferential and the token round trip's prediction-level preference over CosyVoice3 as
  unresolved, and does not call either cue Chatterbox-specific. At WavLM L19 no Chatterbox path has a
  ΔP_t CI above zero, and at w2v-BERT L19 the token round trip keeps .43 against .11 for the HiFT path.
* Observation, not investigated: the CosyVoice3 token round trip's WavLM L0 embedding moves less under
  trim+RMS (mean absolute change 0.038) than every other condition (0.069–0.095, including real speech
  and both Chatterbox paths). It changes no reported conclusion.

## Corrections applied after the first release of these files
* `asr_wer.csv` had been produced with an earlier QC mask (`exclude7.txt`, 19 IDs) and so held 371 of
  the 373 paired IDs per condition. The two missing IDs (`3729-6852-0013`, `61-70970-0030`) were
  transcribed with the same model and scoring functions and merged in; the ASR-derived numbers were
  recomputed. The WER=0 intersection grew from 204 to 205 IDs.
* `band_pred` regenerated a `GroupKFold(5)` split on every call, so the WER=0 subset was evaluated
  under a different speaker split from the full set. It now accepts a fixed speaker→fold map and
  `analyze_asr.py` passes the released `speaker_folds.json` to both evaluations. On the full set this
  reproduces the previous split exactly, so `deepband_full` is unchanged.
* 2026-09-17, with the Chatterbox paths of section N3: the bootstrap replicates of the added paths now
  come from a separate RNG stream, so every previously released row of the intervention CSVs is
  reproduced exactly. The released `centroid2_w2vbert_L19.csv` had not matched the released script in its
  CI endpoints (point estimates identical, and the paper cites only a point estimate from that file), and
  it is now regenerated from the script. No reported value changed.
* The Outcome summaries of sections D and N2 above were corrected on 2026-09-16 against
  `results/final/seeds2_transfer.csv` and `results/final/centroid2_wavlm_tn_L0.csv`. The seed range is
  now stated as regenerated minus original, which reverses the sign of the `drop_vs_orig` column
  (original minus regenerated) reported earlier, and the two normalized G_min lower bounds are rounded
  once from the raw values 0.0022494448 and 0.0031499320 instead of twice. The D outcome no longer
  claims that the pre-specified 0.05 bound was met, because the largest single-seed increase is
  +0.0532. The N2 outcome now names the three matched paths (Vocos, HiFT, token round trip) separately
  from the other interventions. The rules, the plan dates and the analysis history above are unchanged,
  and no experiment was re-run for these edits: the paper's reported values were already correct.
