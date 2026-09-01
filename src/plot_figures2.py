"""Reviewer-hardened camera-ready figures. Output: results/paper/fig2_*.pdf|png"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = os.environ.get("TTS_ANAL_RESULTS", "results/paper5c17")
SYSTEMS = os.environ.get("TTS_ANAL_SYSTEMS", "f5tts,xtts,cosyvoice3,chatterbox,indextts").split(",")
RES = ROOT / "results"
OUT = ROOT / RESULTS

plt.rcParams.update({
    # ICASSP kit: >= 9 pt everywhere at final size; Type-1/42 fonts (no Type 3)
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 200, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
COL = 86 / 25.4      # spconf column width (178mm textwidth, 6mm colsep)
FULL = 178 / 25.4    # full text width for figure*
SSLS = ["wavlm", "hubert", "xlsr", "w2v2lv60", "w2vbert"]
CLAB = {"real": "real", "f5tts": "F5", "xtts": "XTTS", "cosyvoice2": "CosyV.2",
        "chatterbox": "Chatter.", "indextts": "Index", "cosyvoice3": "CosyV.3",
        "qwen3tts": "Qwen3"}
LAB = {"wavlm": "WavLM-L (en)", "hubert": "HuBERT-L (en)",
       "xlsr": "XLS-R (multi)", "w2v2lv60": "w2v2-LV60 (en)",
       "w2vbert": "w2v-BERT 2.0 (multi)"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("saved", name)


def fig_layerwise():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for ssl in SSLS:
        df = pd.read_csv(OUT / f"cloning_{len(SYSTEMS) + 1}way_layers_{ssl}.csv")
        style = "--" if ssl in ("hubert", "wavlm") else "-"
        ax.plot(df.layer, df.loo_acc, style, lw=1.3, label=LAB[ssl])
    k = len(SYSTEMS) + 1
    ax.axhline(1 / k, color="gray", ls=":", lw=0.8)
    ax.text(0.3, 1 / k + 0.02, "chance", color="gray", fontsize=9)
    ax.set_xlabel("SSL layer")
    ax.set_ylabel(f"{k}-way attribution acc.")
    ax.set_ylim(0.1, 1.02)
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    save(fig, "fig2_layerwise")


def fig_intervention():
    """Fig. 1: (a) per-class prediction-rate shift heatmap vs clean real; (b) signed target margin
    S_t with speaker-bootstrap CIs, rows shared with (a); (c) geometry scatter: translation T_t vs
    relative alignment G_t (both x1e3, WavLM L0) with leader-line labels."""
    df = pd.read_csv(OUT / "intervention_wavlm.csv")
    d0 = df[(df.layer == 0) & (df.probe != "clean_real")].set_index("probe")
    c2 = pd.read_csv(OUT / "centroid2_wavlm_L0.csv").set_index("probe")
    probes = [("resynth_vocos", "Vocos→F5", "f5tts"), ("resynth_glvocos", "GL (Vocos mel)→F5", "f5tts"),
              ("resynth_griffinlim", "GL (generic)→F5", "f5tts"),
              ("resynth_hift3", "HiFT→C3", "cosyvoice3"), ("resynth_s3vc3", "Token RT→C3", "cosyvoice3"),
              ("resynth_hift", "HiFT→C2", "cosyvoice2"), ("resynth_s3vc", "S3 RT→C2", "cosyvoice2"),
              ("resynth_qwencodec", "Qwen codec RT", "qwen3tts"),
              ("resynth_bigvgan", "BigVGAN→Index", "indextts"),
              ("resynth_encodec", "EnCodec (ctrl)", None), ("resynth_dac", "DAC (ctrl)", None)]
    probes = [(p, l, t) for p, l, t in probes if p in d0.index and (t is None or t in SYSTEMS)]
    conds = ["real"] + SYSTEMS
    CL2 = dict(CLAB); CL2.update({"cosyvoice3": "C3", "chatterbox": "Chat.", "indextts": "Index"})
    clab = [CL2.get(c, c) for c in conds]
    M = np.array([[d0.loc[p, f"dP_{c}"] for c in conds] for p, _, _ in probes])
    n = len(probes)
    fig = plt.figure(figsize=(FULL, 2.05 + 0.05 * max(0, n - 8)), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[3.3, 1.3, 3.0])
    ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharey=ax); ax3 = fig.add_subplot(gs[2])
    # ---- (a) heatmap
    ax.imshow(M, cmap="RdBu_r", vmin=-0.9, vmax=0.9, aspect="auto")
    for i, (p, _, t) in enumerate(probes):
        if t:
            j = conds.index(t)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="k", lw=1.6))
        for j in range(len(conds)):
            if abs(M[i, j]) >= 0.10:
                ax.text(j, i, f"{M[i, j]:.2f}".replace("-", "$-$"), ha="center", va="center", fontsize=7.5,
                        color="white" if abs(M[i, j]) > 0.45 else "black")
    ax.set_xticks(range(len(conds)), clab, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(n), [l for _, l, _ in probes], fontsize=9)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_title(r"(a) $\Delta P$(class)", fontsize=9.5)
    # ---- (b) signed target margin, rows shared with (a)
    for i in range(n):
        if i % 2 == 1:
            ax2.axhspan(i - 0.5, i + 0.5, color="#f2f2f2", zorder=0)
    for i, (p, _, t) in enumerate(probes):
        r = d0.loc[p]
        if t and "S_target" in r and not pd.isna(r["S_target"]):
            ax2.barh(i, r["S_target"], xerr=[[r["S_target"] - r["S_ci_lo"]], [r["S_ci_hi"] - r["S_target"]]],
                     capsize=2, error_kw={"lw": 0.8}, color="#4878b0" if r["S_target"] > 0 else "#c44e52",
                     height=0.72, zorder=3)
        else:  # off-target controls: S under the F5 target and under the CosyVoice3 target
            vals = [(t2, r.get(f"S_{t2}", np.nan)) for t2 in ["f5tts", "cosyvoice3"] if t2 in SYSTEMS]
            for k, (t2, v) in enumerate(vals):
                if not pd.isna(v):
                    ax2.barh(i - 0.19 + 0.38 * k, v, height=0.34, color=["#7a7a7a", "#bdbdbd"][k], zorder=3,
                             label=f"control under {CL2.get(t2, t2)}" if i == n - 1 else None)
    ax2.axvline(0, color="k", lw=0.7)
    ax2.tick_params(axis="y", labelleft=False, left=False)
    ax2.tick_params(axis="x", labelsize=9)
    ax2.set_xlabel(r"$S_t$", fontsize=9, labelpad=1)
    ax2.set_title(r"(b) margin $S_t$", fontsize=9.5)
    # ---- (c) geometry scatter with leader-line labels
    def lab_of(l): return l.split("→")[0].replace("GL (Vocos mel)", "GL-Vocos").replace("GL (generic)", "GL-generic")
    pts = []  # (x, y, label, kind)
    for p, l, t in probes:
        if t and p in c2.index and "T" in c2.columns and not pd.isna(c2.loc[p, "T"]):
            pts.append((1e3 * c2.loc[p, "T"], 1e3 * c2.loc[p, "G"], lab_of(l), "gl" if "GL" in l else "matched"))
    CTRL_LAB = {"resynth_encodec": "Enc", "resynth_dac": "DAC", "resynth_bigvgan": "BigV"}
    for p in ["resynth_encodec", "resynth_dac", "resynth_bigvgan"]:
        for t2 in [x for x in ["f5tts", "cosyvoice3"] if x in SYSTEMS]:
            if p == "resynth_bigvgan" and t2 != "cosyvoice3":
                continue
            key = f"{p}_under_{t2}"
            if key in c2.index:
                pts.append((1e3 * c2.loc[key, "T"], 1e3 * c2.loc[key, "G"],
                            CTRL_LAB[p] + "/" + ("F5" if t2 == "f5tts" else "C3"), "ctrl"))
    style = {"matched": dict(marker="o", s=46, color="#4878b0", zorder=4),
             "gl": dict(marker="D", s=34, color="#4878b0", zorder=4),
             "ctrl": dict(marker="s", s=40, color="#8c8c8c", zorder=3)}
    # label anchor positions in data units (x1e3); None -> default offset
    POS = {"Vocos": (12.8, 9.3), "GL-generic": (3.2, 14.7), "GL-Vocos": (3.2, 12.0),
           "HiFT": (5.2, 6.6), "Token RT": (11.2, 5.9), "BigVGAN": (9.6, -2.2),
           "Enc/F5": (5.0, 9.9), "DAC/F5": (12.8, 3.5), "BigV/C3": (12.8, 1.4),
           "DAC/C3": (3.6, 4.6), "Enc/C3": (0.8, -1.8)}
    for x, y, lab, kind in pts:
        ax3.scatter(x, y, **style[kind])
        tx, ty = POS.get(lab, (x + 0.6, y))
        col = "#555555" if kind == "ctrl" else "black"
        ax3.annotate(lab, (x, y), xytext=(tx, ty), textcoords="data", fontsize=8.5, color=col,
                     ha="left" if tx > x else "right", va="center",
                     arrowprops=dict(arrowstyle="-", lw=0.5, color="#9a9a9a", shrinkA=0, shrinkB=3),
                     bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85), zorder=5)
    ax3.axhline(0, color="k", lw=0.6); ax3.axvline(0, color="k", lw=0.6)
    ax3.set_xlim(-2.2, 17.8); ax3.set_ylim(-3.0, 15.9)
    ax3.set_xlabel(r"translation $T_t$ ($\times10^{-3}$)", fontsize=9)
    ax3.set_ylabel(r"relative alignment $G_t$ ($\times10^{-3}$)", fontsize=9)
    ax3.set_title("(c) centroid geometry", fontsize=9.5); ax3.tick_params(labelsize=9)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03)
    save(fig, "fig2_intervention")


def fig_robustAB():
    A = pd.read_csv(OUT / "robustA_cloning_wavlm.csv")
    B = pd.read_csv(OUT / "robustB_wavlm.csv")
    order = ["clean", "common", "mp3_64k", "lp4k", "hp2k", "phaserand", "noise20"]
    labels = ["clean", "comm.", "MP3", "LP4k", "HP2k", "phase", "noise"]
    a = [A[(A.pert == p) & (A.layer == 0)].acc.iloc[0] for p in order]
    bsub = B[B.layer == 4].set_index("pert").reindex(order)
    b = bsub.acc.values
    berr = [b - bsub.ci_lo.values, bsub.ci_hi.values - b]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(COL, 1.85))
    ax.bar(x - 0.19, a, 0.38, label="A: matched (LOO, L0)")
    ax.bar(x + 0.19, b, 0.38, yerr=berr, capsize=2, error_kw={"lw": 0.8},
           label="B: clean→pert. (spk-disj., L4)")
    k = len(SYSTEMS) + 1
    ax.axhline(1 / k, color="gray", ls=":", lw=0.8)
    ax.set_xticks(x, labels, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel(f"{k}-way acc. (WavLM)")
    ax.set_ylim(0, 1.42)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(fontsize=9, loc="upper center", frameon=False)
    fig.tight_layout()
    save(fig, "fig2_robustAB")


def fig_retention():
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    early, deep = [], []
    for ssl in SSLS:
        df = pd.read_csv(OUT / f"cloning_ttsonly_layers_{ssl}.csv")
        early.append(df.iloc[1:6].loo_acc.mean())
        deep.append(df.iloc[17:22].loo_acc.mean())
    x = np.arange(len(SSLS))
    ax.bar(x - 0.19, early, 0.38, label="early layers (1–5)")
    ax.bar(x + 0.19, deep, 0.38, label="deep layers (17–21)")
    ax.set_xticks(x, [LAB[s].replace(" ", "\n", 1) for s in SSLS], fontsize=9)
    ax.set_ylabel(f"TTS-only {len(SYSTEMS)}-way acc.")
    ax.axhline(1 / len(SYSTEMS), color="gray", ls=":", lw=0.8)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9, ncol=2, loc="upper center", frameon=False)
    fig.tight_layout()
    save(fig, "fig2_retention")


def fig_interv_layers():
    """Layer-wise Delta P to the hypothesized target for the architecture-matched probes
    (unmatched codecs are not plotted: their margin is a different quantity)."""
    matched = [("resynth_vocos", "Vocos→F5", "-"), ("resynth_glvocos", "GL→F5", "--")]
    if "cosyvoice3" in SYSTEMS:
        matched += [("resynth_hift3", "HiFT→C3", "-"), ("resynth_s3vc3", "Token RT→C3", "-")]
    if "cosyvoice2" in SYSTEMS:
        matched += [("resynth_hift", "HiFT→CosyV.2", "-"), ("resynth_s3vc", "S3 RT→CosyV.2", "-")]
    if "qwen3tts" in SYSTEMS:
        matched += [("resynth_qwencodec", "Qwen codec→Qwen3", "-.")]
    leg_rows = (len(matched) + 3) // 4
    leg_h = 0.16 * leg_rows + 0.06; fig_h = 1.55 + leg_h
    MK = {"resynth_vocos": "o", "resynth_glvocos": "x", "resynth_hift3": "^", "resynth_s3vc3": "s",
          "resynth_hift": "v", "resynth_s3vc": "D", "resynth_qwencodec": "P"}
    fig, axes = plt.subplots(1, 2, figsize=(COL, fig_h), sharey=True)
    for ax, ssl, name in [(axes[0], "wavlm", "WavLM"), (axes[1], "w2vbert", "w2v-BERT 2.0")]:
        df = pd.read_csv(OUT / f"intervention_{ssl}.csv")
        for probe, lab, ls in matched:
            d = df[df.probe == probe].sort_values("layer")
            ax.plot(d.layer, d.delta_target, ls, lw=1.5, marker=MK.get(probe, "o"), ms=3.5, markevery=4,
                    label=lab if ssl == "wavlm" else None)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("layer")
    axes[0].set_ylabel(r"$\Delta P$ to target")
    fig.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False,
               handlelength=1.4, columnspacing=0.6, handletextpad=0.3, borderaxespad=0)
    fig.tight_layout(rect=[0, 0, 1, 1 - leg_h / fig_h])
    save(fig, "fig2_interv_layers")


if __name__ == "__main__":
    fig_layerwise()
    fig_intervention()
    fig_robustAB()
    fig_retention()
    fig_interv_layers()
