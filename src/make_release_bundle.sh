#!/bin/bash
# Build the data bundle released with the paper (exactly the paper's scope: the five systems, the
# interventions, the decoder-only vocoders, the paper's perturbations, the controlled-seed subsets,
# and the embeddings of those conditions). Plain tar archives + SHA256SUMS.
#   bash src/make_release_bundle.sh [OUT_DIR]     (default: <repo>/../tts_anal_release)
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; OUT="${1:-$ROOT/../tts_anal_release}"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"; cd "$ROOT"
SYS="f5tts xtts cosyvoice3 chatterbox indextts"
INTERV="resynth_vocos resynth_glvocos resynth_griffinlim resynth_hift3 resynth_s3vc3 resynth_bigvgan resynth_encodec resynth_dac"
VOC="voc_pwg voc_melgan voc_mbmelgan voc_hifigan voc_stylemelgan"
PERTS="common common_sym mp3_64k lp4k hp2k noise20 noise20_s2 noise20_s3 phaserand phaserand_s2 phaserand_s3"
L="$OUT/.lists"; mkdir -p "$L"; : > "$L/gen"; : > "$L/pert"; : > "$L/seeds"; : > "$L/emb"
for c in $SYS $INTERV $VOC; do echo "data/generated/$c" >> "$L/gen"; done
for p in $PERTS; do for c in real $SYS; do [ -d "data/perturbed/$p/$c" ] && echo "data/perturbed/$p/$c" >> "$L/pert"; done; done
for s in $SYS; do for k in 101 102 103; do echo "data/generated_seeds/${s}_s$k" >> "$L/seeds"; done; done
for ssl in wavlm hubert xlsr w2v2lv60 w2vbert; do
  for c in real $SYS $INTERV $VOC; do f="results/embeddings/$ssl/$c.npz"; [ -f "$f" ] && echo "$f" >> "$L/emb"; done
  for k in 101 102 103; do for s in $SYS; do f="results/embeddings/${ssl}_seeds20/${s}_s$k.npz"; [ -f "$f" ] && echo "$f" >> "$L/emb"; done; done
done
for ssl in wavlm w2vbert; do
  for c in real $SYS; do f="results/embeddings/${ssl}_tn/$c.npz"; [ -f "$f" ] && echo "$f" >> "$L/emb"; done
  for p in $PERTS; do for c in real $SYS; do f="results/embeddings/${ssl}_p_$p/$c.npz"; [ -f "$f" ] && echo "$f" >> "$L/emb"; done; done
done
RES=$(git ls-files results/paper5c17 2>/dev/null || ls results/paper5c17/*)
echo "[bundle] root=$ROOT out=$OUT ($(date))"
tar -cf "$OUT/manifests_and_results.tar" data/manifests/main.jsonl data/manifests/exclude17.txt data/manifests/speaker_folds.json data/manifests/seedsub20.jsonl $RES analysis_plan.md DATA_LICENSES.md LICENSE NOTICE PROVENANCE.md provenance.json $(git ls-files provenance) provenance_files.jsonl.gz provenance_files_summary.md README.md
tar -cf "$OUT/embeddings.tar"      -T "$L/emb"
tar -cf "$OUT/audio_generated.tar" -T "$L/gen"
tar -cf "$OUT/audio_perturbed.tar" -T "$L/pert"
tar -cf "$OUT/audio_subsets.tar"   -T "$L/seeds"
cd "$OUT" && sha256sum *.tar > SHA256SUMS && ls -l *.tar && cat SHA256SUMS
echo "[bundle] done ($(date))"
