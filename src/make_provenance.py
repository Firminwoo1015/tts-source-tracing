"""Historical provenance reconstruction (Codex audit item).

Collects, from what exists on disk NOW:
  - sha256 of key model checkpoint files under $TTS_ANAL_CKPTS
  - HF cache revisions (snapshot commit hashes) for every cached model
  - git commits of third_party repos
  - per-env package freezes (written next to this report)
  - generation WAV metadata sample (sr / subtype / channels per condition)
Output: PROVENANCE.md + provenance.json at repo root.
"""

import hashlib
import json
import subprocess
import os
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
CKPTS = Path(os.environ.get("TTS_ANAL_CKPTS", ROOT / "ckpts"))
ENVS = ["tts_anal", "tts_xtts", "tts_cosy", "tts_chatter", "tts_index", "tts_voc", "tts_qwen"]


def sha256(p, cap=1 << 30):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def logical(q):
    """Machine-independent key: $TTS_ANAL_CKPTS/... or ~/... instead of absolute paths."""
    q = Path(q).resolve()
    try:
        return "$TTS_ANAL_CKPTS/" + str(q.relative_to(CKPTS.resolve()))
    except ValueError:
        pass
    try:
        return "~/" + str(q.relative_to(Path.home().resolve()))
    except ValueError:
        return str(q)


def main():
    prov = {}

    # 1) explicit checkpoint dirs
    ckpt_files = {}
    for d in ["CosyVoice2-0.5B", "IndexTTS-1.5"]:
        base = CKPTS / d
        if base.exists():
            for p in sorted(base.rglob("*")):
                if p.is_file() and p.suffix in {".pt", ".pth", ".safetensors", ".bin", ".onnx", ".yaml", ".yml", ".json"} and p.stat().st_size > 0:
                    ckpt_files[str(p.relative_to(CKPTS))] = {
                        "bytes": p.stat().st_size,
                        "sha256": sha256(p) if p.stat().st_size < 8e9 else "skipped(>8GB)"}
    prov["checkpoint_hashes"] = ckpt_files

    # 2) HF cache revisions
    hub = {}
    for m in sorted((CKPTS / "hub").glob("models--*")):
        snaps = sorted((m / "snapshots").glob("*")) if (m / "snapshots").exists() else []
        hub[m.name.replace("models--", "").replace("--", "/")] = [s.name for s in snaps]
    prov["hf_cache_revisions"] = hub

    # 3) third_party commits
    commits = {}
    for repo in sorted((ROOT / "third_party").glob("*")):
        if (repo / ".git").exists():
            try:
                c = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip()
                commits[repo.name] = c
            except Exception as e:
                commits[repo.name] = f"error: {e}"
    prov["third_party_commits"] = commits

    # 4) env freezes
    outdir = ROOT / "provenance"
    outdir.mkdir(exist_ok=True)
    for env in ENVS:
        pip = Path(os.environ.get("CONDA_ROOT", Path(sys.executable).resolve().parents[3])) / "envs" / env / "bin" / "pip"
        if pip.exists():
            r = subprocess.run([str(pip), "freeze"], capture_output=True, text=True)
            (outdir / f"freeze_{env}.txt").write_text(r.stdout)
    prov["env_freezes"] = [f"provenance/freeze_{e}.txt" for e in ENVS]

    # 4b) (v3) resolved HF revisions + per-file SHA-256 for every model used
    USED = ["SWivid--F5-TTS","SWivid--E2-TTS","charactr--vocos-mel-24khz","facebook--encodec_24khz",
            "nvidia--bigvgan_v2_24khz_100band_256x","ResembleAI--chatterbox","FunAudioLLM--CosyVoice2-0.5B",
            "microsoft--wavlm-large","facebook--hubert-large-ll60k","facebook--wav2vec2-xls-r-300m",
            "facebook--wav2vec2-large-lv60","facebook--w2v-bert-2.0","Systran--faster-whisper-large-v3",
            "IndexTeam--IndexTTS-1.5","FunAudioLLM--Fun-CosyVoice3-0.5B-2512","Qwen--Qwen3-TTS-12Hz-1.7B-Base","Qwen--Qwen3-TTS-Tokenizer-12Hz"]
    hubm = {}
    for m in USED:
        d = CKPTS / "hub" / f"models--{m}"
        if not d.exists():
            hubm[m] = "NOT IN CACHE"; continue
        ref = d / "refs" / "main"
        rev = ref.read_text().strip() if ref.exists() else None
        snaps = sorted((d / "snapshots").glob("*"))
        use = d / "snapshots" / rev if rev and (d / "snapshots" / rev).exists() else (snaps[-1] if snaps else None)
        files = {}
        if use:
            for q in sorted(use.rglob("*")):
                if q.is_file() and q.stat().st_size > 0 and q.resolve().stat().st_size < 6e9:
                    files[str(q.relative_to(use))] = {"bytes": q.resolve().stat().st_size, "sha256": sha256(q.resolve())}
        hubm[m] = {"resolved_revision": rev, "snapshot_used": use.name if use else None,
                   "all_snapshots": [x.name for x in snaps], "files": files}
    prov["hf_models_used"] = hubm
    # 4c) (v3) non-HF checkpoints: XTTS (coqui TTS_HOME), DAC (descript cache), ParallelWaveGAN (skip AppleDouble ._*)
    extra = {}
    for pat in [CKPTS / "tts", Path.home() / ".cache" / "descript" / "dac"]:
        if pat.exists():
            for q in sorted(pat.rglob("*")):
                if q.is_file() and q.suffix in {".pth", ".pt", ".json", ".yaml", ".yml", ".h5", ".pkl"} and not q.name.startswith("._"):
                    extra[logical(q)] = {"bytes": q.stat().st_size, "sha256": sha256(q)}
    for q in sorted((CKPTS / "parallel_wavegan").rglob("*")):
        if q.is_file() and q.suffix in {".pkl", ".yml", ".h5"} and not q.name.startswith("._"):
            extra[logical(q)] = {"bytes": q.stat().st_size, "sha256": sha256(q)}
    prov["other_checkpoints"] = extra
    # 4d) (v3) local BigVGAN patch
    prov["bigvgan_local_patch"] = subprocess.run(
        ["git", "-C", str(ROOT / "third_party/BigVGAN"), "diff", "--", "bigvgan.py"],
        capture_output=True, text=True).stdout

    # 4e) license identifiers (model cards / repo LICENSE files at the revisions used; see DATA_LICENSES.md)
    lic_path = ROOT / "DATA_LICENSES.md"
    try:
        import re as _re
        rows = [l for l in lic_path.read_text().split("\n") if l.startswith("| ") and not l.startswith("| system") and not l.startswith("|---")]
        prov["licenses"] = {r.split("|")[1].strip(): {"license": r.split("|")[2].strip(), "revision": r.split("|")[3].strip(), "url": r.split("|")[4].strip(), "note": r.split("|")[5].strip()} for r in rows if r.count("|") >= 6}
    except Exception as e:
        prov["licenses"] = f"error: {e}"

    # 5) WAV metadata per condition (first file)
    meta = {}
    for cond in sorted((ROOT / "data/generated").glob("*")):
        w = next(iter(sorted(cond.glob("*.wav"))), None)
        if w:
            i = sf.info(str(w))
            meta[cond.name] = {"sr": i.samplerate, "subtype": i.subtype,
                               "channels": i.channels}
    prov["wav_metadata"] = meta

    (ROOT / "provenance.json").write_text(json.dumps(prov, indent=1))

    md = ["# Provenance (reconstructed post hoc; see Codex audit 2026-08)\n"]
    md.append(f"- third_party commits: {json.dumps(commits)}")
    md.append(f"- HF cache revisions: {len(hub)} models (provenance.json)")
    md.append(f"- checkpoint hashes: {len(ckpt_files)} files (provenance.json)")
    md.append(f"- env freezes: provenance/freeze_<env>.txt ({len(ENVS)} envs)")
    n_rev = sum(isinstance(v, dict) and v["resolved_revision"] is not None for v in hubm.values())
    n_snap = sum(isinstance(v, dict) and len(v["files"]) > 0 for v in hubm.values())
    n_files = sum(len(v["files"]) for v in hubm.values() if isinstance(v, dict))
    md.append(f"- HF models used: {n_rev} revision IDs recorded; snapshot-backed per-file hashes for "
              f"{n_snap} repositories ({n_files} files); repos whose weights live in local directories "
              f"(CosyVoice2-0.5B, IndexTTS-1.5) are hashed separately under checkpoint_hashes "
              f"({len(ckpt_files)} files)")
    md.append(f"- non-HF checkpoints (XTTS/DAC/ParallelWaveGAN, AppleDouble files excluded): {len(extra)} files hashed")
    md.append("- BigVGAN local patch recorded as git diff (bigvgan_local_patch)")
    md.append("- NOTE: the main run had no uniform per-utterance seeding protocol "
              "(F5 family seed=0; CosyVoice2/3 YAMLs fix an RNG seed at load time; "
              "XTTS/Chatterbox/IndexTTS/Qwen3-TTS wrappers set no seed); multi-seed "
              "subset used seeds 1,2 with torch/np/random seeding.")
    md.append("- Chatterbox: PerTh watermarker replaced by DummyWatermarker "
              "at generation (no audio watermark in outputs).")
    (ROOT / "PROVENANCE.md").write_text("\n".join(md) + "\n")
    print("wrote PROVENANCE.md, provenance.json,",
          f"{len(ckpt_files)} hashes, {len(hub)} HF models, {len(commits)} repos")


if __name__ == "__main__":
    main()
