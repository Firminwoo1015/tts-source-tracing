# Copy to env.sh, edit the paths, then `source env.sh` before running anything.
export TTS_ANAL_ROOT=/path/to/tts_anal          # this repository (default: inferred from script location)
export TTS_ANAL_CKPTS=/path/to/ckpts            # checkpoints + HF cache ($TTS_ANAL_CKPTS/hub); default: $TTS_ANAL_ROOT/ckpts
export TTS_ANAL_DATASETS=/path/to/datasets      # LibriSpeech/test-clean, VCTK;               default: $TTS_ANAL_ROOT/datasets
export CONDA_ROOT=/path/to/anaconda3            # default: `conda info --base`
export HF_HOME=$TTS_ANAL_CKPTS
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
