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
* **Outcome:** no two seeds gave identical waveforms; macro-F1 change −0.05 to +0.03 with no decrease
  whose CI excludes zero; distance ratio 0.20–0.40 → rule satisfied.

## N. Nuisance control (`analyze_normalization_control.py` → `normalization_control.csv`)
* Embeddings re-extracted after silence trimming (librosa, top_db=35) and per-utterance RMS normalization
  to 0.05, WavLM and w2v-BERT, all six conditions, same protocol as Table 1.
* Rule: "cues do not depend on level/silence conventions" only if macro-F1 stays within 0.05; otherwise
  the dependence is quantified.
* **Outcome:** six-way macro-F1 fell by 0.08 (WavLM) and 0.09 (w2v-BERT), paired CIs [0.06, 0.10] and
  [0.06, 0.13]; the selected w2v-BERT layer moved from L3 to L18 in every fold → the paper reports the
  dependence rather than invariance.

