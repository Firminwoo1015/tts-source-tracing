#!/bin/bash
# Reproduces every table, figure and quoted number of the paper from the released embeddings
# (results/embeddings/<tag>/<cond>.npz, see README "Data bundle"). CPU only, conda env `tts_anal`.
# Writes results/final/*.csv, fig2_*.pdf and PAPER_NUMBERS2.json (the claim-to-file map is in README).
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TTS_ANAL_SYSTEMS="f5tts,xtts,cosyvoice3,chatterbox,indextts"
export TTS_ANAL_RESULTS="results/final"
export TTS_ANAL_EXCLUDE="data/manifests/exclude17.txt"
L="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
for ssl in wavlm hubert xlsr w2v2lv60 w2vbert; do python src/analyze_cloning.py --ssl $ssl; done   # Table 1, per-class recall
python src/analyze_mfcc_cloning.py                              # Table 1 (MFCC baseline)
python src/analyze_simplecues.py                                # Table 1 (prosodic baseline)
python src/analyze_asr.py                                       # Table 1 (Whisper-error baseline), §4.3 WER=0 control
python src/analyze_bands.py                                     # Table 3
python src/analyze_spkdisjoint_extras.py                        # alternative deep bands (bands_alt.csv)
python src/analyze_transplant2.py --ssl wavlm   --layers $L     # Fig. 1(a,b), Table 2, Fig. 2 (prediction shifts)
python src/analyze_transplant2.py --ssl w2vbert --layers $L     # Fig. 2 (w2v-BERT)
python src/analyze_centroid2.py --ssl wavlm   --layer 0         # Fig. 1(c), Table 2 (T, G, G_adj, G_min at 10,000 replicates)
python src/analyze_centroid2.py --ssl w2vbert --layer 19        # §4.2 w2v-BERT L19 statement
python src/analyze_decoder_swap.py --ssl wavlm --layer 0        # §4.2 same-mel decoder swap (D_G) + cross-mel GL table
python src/analyze_lineage.py --ssl wavlm --layer 0             # §4.2 CosyVoice3/Chatterbox own-vs-related alignment
python src/analyze_deconly.py --ssl wavlm                       # §4.2 decoder-only control
python src/analyze_deep_probe.py                                # §4.3 logistic / centroid / kNN deep-band probes
python src/analyze_knn_mechanism.py                             # §4.3 paired-kNN mechanism
python src/analyze_robustB3.py                                  # Table 4 (WavLM) + w2v-BERT
python src/analyze_normalization_control.py                     # §4.1 trim+RMS control
# §4.2 nuisance control on the interventions (control N2): needs the trim+RMS embeddings,
#   python src/extract_embeddings.py --ssl wavlm --trim-norm --out-tag wavlm_tn --conditions \
#     real f5tts xtts cosyvoice3 chatterbox indextts resynth_vocos resynth_glvocos \
#     resynth_griffinlim resynth_hift3 resynth_s3vc3 resynth_hiftcb resynth_s3vccb resynth_glc3 resynth_glcb resynth_bigvgan \
#     resynth_encodec resynth_dac
if [ -f results/embeddings/wavlm_tn/resynth_vocos.npz ]; then
  python src/analyze_transplant2.py --ssl wavlm_tn --layers 0 2 4 8 19   # §4.2 normalized target shifts
  python src/analyze_centroid2.py   --ssl wavlm_tn --layer 0             # §4.2 normalized G_adj, G_min
  python src/analyze_decoder_swap.py --ssl wavlm_tn --layer 0            # decoder swap under trim+RMS
fi
python src/analyze_seeds2.py                                    # §4.1 controlled-seed study
python src/plot_figures2.py                                     # fig2_intervention, fig2_interv_layers
python src/paper_numbers.py results/final                   # PAPER_NUMBERS2.json (every quoted number)
