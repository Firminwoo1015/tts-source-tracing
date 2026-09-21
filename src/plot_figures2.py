"""Reviewer-hardened camera-ready figures. Output: results/final/fig2_*.pdf|png"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = os.environ.get("TTS_ANAL_RESULTS", "results/final")
SYSTEMS = os.environ.get("TTS_ANAL_SYSTEMS", "f5tts,xtts,cosyvoice3,chatterbox,indextts").split(",")
RES = ROOT / "results"
OUT = ROOT / RESULTS

plt.rcParams.update({
    # ICASSP kit: >= 9 pt everywhere at final size; Type-1/42 fonts (no Type 3)
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 200, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    "text.color": "#243342", "axes.labelcolor": "#243342", "axes.edgecolor": "#71808B",
    "xtick.color": "#243342", "ytick.color": "#243342", "axes.linewidth": .55,
    "savefig.facecolor": "white",
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

# Fig. 1 / Fig. 2 shared encoding: one color and marker per intervention probe,
# gray hollow markers for the off-target controls (circle = F5 target, square = CosyVoice3 target,
# triangle = Chatterbox target). Shape encodes the path type within a system (HiFT ^, token round trip s).
INK, GRAY, LIGHT = "#243342", "#71808B", "#E7ECF0"
PALETTE = {"resynth_vocos": "#247BA8", "resynth_glvocos": "#C77C17", "resynth_griffinlim": "#C77C17",
           "resynth_hift3": "#258A73", "resynth_s3vc3": "#C15365", "resynth_bigvgan": "#8061A8",
           "resynth_hiftcb": "#5E7F1F", "resynth_s3vccb": "#9C5B2E",
           "resynth_glc3": "#C77C17", "resynth_glcb": "#C77C17"}   # all Griffin-Lim controls share one color
MARKERS = {"resynth_vocos": "o", "resynth_glvocos": "D", "resynth_griffinlim": "D",
           "resynth_hift3": "^", "resynth_s3vc3": "s", "resynth_bigvgan": "v",
           "resynth_hiftcb": "^", "resynth_s3vccb": "s", "resynth_glc3": "D", "resynth_glcb": "D"}


def style_axis(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#B7C2CA")
    ax.tick_params(axis="both", length=2.5, width=.55, pad=3, colors=INK)
    ax.set_axisbelow(True)


def assert_inside(fig, name):
    """Every visible text must lie inside the fixed media box (the PDF is placed at 1:1 scale)."""
    from matplotlib.text import Text
    fig.canvas.draw(); r = fig.canvas.get_renderer(); out = []
    for a in fig.findobj(match=Text):
        if not a.get_visible() or not a.get_text().strip():
            continue
        if a.axes is not None and a not in a.axes.texts and a not in [a.axes.title, a.axes.xaxis.label, a.axes.yaxis.label]:
            continue
        bb = a.get_window_extent(r)
        if bb.x0 < -.75 or bb.y0 < -.75 or bb.x1 > fig.bbox.width + .75 or bb.y1 > fig.bbox.height + .75:
            out.append(a.get_text())
    assert not out, f"text outside {name} canvas: {out}"


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
    """Fig. 1: per-class prediction-rate shift vs clean real at WavLM L0, one row per intervention,
    single column, two panels. (a) the five matched paths grouped by target system and the three
    off-target neural controls; (b) the same-mel Griffin-Lim reconstructions that replace the three
    native decoders. Margins and centroid geometry are in Table 2, the decoder-swap table, and
    intervention_wavlm.csv / centroid2_wavlm_L0.csv."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory
    df = pd.read_csv(OUT / "intervention_wavlm.csv")
    d0 = df[(df.layer == 0) & (df.probe != "clean_real")].set_index("probe")
    panels = [("(a) Matched paths and neural controls",
               [[("resynth_vocos", "Vocos (F5)", "f5tts")],
                [("resynth_hift3", "HiFT (C3)", "cosyvoice3"), ("resynth_s3vc3", "Token RT (C3)", "cosyvoice3")],
                [("resynth_hiftcb", "HiFT (Chat.)", "chatterbox"), ("resynth_s3vccb", "Token RT (Chat.)", "chatterbox")],
                [("resynth_bigvgan", "BigVGAN (Index)", "indextts"), ("resynth_encodec", "EnCodec", None),
                 ("resynth_dac", "DAC", None)]]),
              ("(b) Griffin–Lim from the same mel",
               [[("resynth_glvocos", "GL (F5 mel)", "f5tts"), ("resynth_glc3", "GL (C3 mel)", "cosyvoice3"),
                 ("resynth_glcb", "GL (Chat. mel)", "chatterbox")]])]
    classes = ["real"] + SYSTEMS
    CL2 = dict(CLAB); CL2.update({"cosyvoice3": "C3", "chatterbox": "Chat.", "indextts": "Idx."})   # 24 pt cells
    cmap = LinearSegmentedColormap.from_list("shifts", ["#267CA1", "#FAFBFC", "#D38A7E"]); norm = Normalize(-.7, .7)
    gap, pgap = .38, 1.5                     # between system blocks; between panels (holds the (b) title)
    total = sum(len(b) for _, bl in panels for b in bl) \
        + gap * sum(len(bl) - 1 for _, bl in panels) + pgap * (len(panels) - 1)
    W, H = COL * 72, 207.0
    top = 30                                 # (a) title line + class labels above the grid
    fig = plt.figure(figsize=(COL, H / 72))
    ax = fig.add_axes([101 / W, 39 / H, (W - 103) / W, (H - 39 - top) / H])
    ax.set_xlim(-.5, len(classes) - .5); ax.set_ylim(total - .5, -.5)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=3, colors=INK)
    ttl = blended_transform_factory(fig.transFigure, ax.transData)
    y, yt, yl, yc = 0.0, [], [], []
    for k, (title, blocks) in enumerate(panels):
        if k == 0:
            ax.text(2 / W, -.5, title, transform=ttl, ha="left", va="bottom", fontsize=9, color=INK,
                    fontweight="bold", clip_on=False)
            ax.texts[-1].set_position((2 / W, -.5 - 16 / (H - 39 - top) * total))   # above the class labels
        else:
            ax.text(2 / W, (last + .5 + y - .5) / 2, title, transform=ttl, ha="left", va="center",
                    fontsize=9, color=INK, fontweight="bold", clip_on=False)   # centered in the panel gap
        for block in blocks:
            for p, label, t in block:
                for jx, cl in enumerate(classes):
                    v = d0.loc[p, f"dP_{cl}"]
                    ax.add_patch(Rectangle((jx - .5, y - .5), 1, 1, facecolor=cmap(norm(v)), edgecolor="none", zorder=2))
                    # threshold, the boxed target cell, and the two CosyVoice3 -> Chatterbox cross-shifts
                    # (.08/.09) that the text compares with the .18/.22 in the other direction
                    if abs(v) >= .10 or cl == t or (cl == "chatterbox" and p in ("resynth_hift3", "resynth_s3vc3")):
                        ax.text(jx, y, f"{v:.2f}".replace("0.", "."), ha="center", va="center",
                                color="white" if jx == 0 else "black", fontsize=9, zorder=4)
                if t:
                    ax.add_patch(Rectangle((classes.index(t) - .5, y - .5), 1, 1, fill=False, lw=1.1, edgecolor=INK, zorder=5))
                yt.append(y); yl.append(label); yc.append(PALETTE.get(p, INK)); last = y; y += 1
            y += gap
        y += pgap - gap
    ax.set_yticks(yt, yl)
    for tick, c in zip(ax.get_yticklabels(), yc): tick.set_color(c)
    ax.set_xticks(np.arange(len(classes)), [CL2.get(cl, cl) for cl in classes])
    ax.xaxis.set_ticks_position("top"); ax.tick_params(axis="x", pad=2)
    import matplotlib.cm as cm
    cax = fig.add_axes([(101 + 18) / W, 25 / H, (W - 103 - 36) / W, 4.5 / H])
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=[-.7, 0, .7])
    cb.ax.set_xticklabels(["-.7", "0", ".7"], fontsize=9); cb.ax.tick_params(length=2, width=.55, pad=1, colors=INK)
    cb.outline.set_linewidth(.55); cb.outline.set_edgecolor(GRAY)
    cb.set_label(r"$\Delta P$(class)", fontsize=9, color=INK, labelpad=1)
    assert_inside(fig, "fig2_intervention")
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
    """Fig. 2: layer-wise Delta P to the hypothesized target for the architecture-matched probes and
    the mel-only Griffin-Lim probe (same colors and markers as Fig. 1).

    Point estimates only, by choice: this figure carries the depth trend, and at L19 several
    curves sit near zero where small error bars add more clutter than information. The L19 CI
    judgement stays in the text and the per-layer CIs are in the released results."""
    matched = [("resynth_vocos", "Vocos (F5)")]      # matched paths only; the GL controls are in Fig. 1
    if "cosyvoice3" in SYSTEMS:
        matched += [("resynth_hift3", "HiFT (C3)"), ("resynth_s3vc3", "Token RT (C3)")]
    if "chatterbox" in SYSTEMS:
        matched += [("resynth_hiftcb", "HiFT (Chat.)"), ("resynth_s3vccb", "Token RT (Chat.)")]
    # two legend columns, filled column-first: Vocos over the two HiFT paths, GL over the two round trips
    legend_order = ["resynth_vocos", "resynth_hift3", "resynth_hiftcb", "resynth_s3vc3", "resynth_s3vccb"]
    height_in = 2.10                       # three legend rows above two 64 pt panels
    fig = plt.figure(figsize=(COL, height_in))
    W, H = COL * 72, height_in * 72
    axes = [fig.add_axes([x / W, 31 / H, 82 / W, 64 / H]) for x in [42, 155]]
    handles = []
    for ax, ssl, title in zip(axes, ["wavlm", "w2vbert"], ["WavLM", "w2v-BERT 2.0"]):
        df = pd.read_csv(OUT / f"intervention_{ssl}.csv")
        style_axis(ax)
        ax.set_xlim(-.6, 24.6); ax.set_ylim(-.13, 1.05)
        # ticks at the layers the text interprets. A tick at 2 is not available: at this
        # width layers 0 and 2 are 6.5 pt apart and each label is 6 pt wide, so they touch.
        # The early peak is legible from the curves themselves, so it carries no marker.
        ax.set_xticks([0, 12, 19, 24], ["0", "12", "19", "24"])
        ax.set_yticks([0, .5, 1])
        ax.set_title(title, pad=5, fontsize=9, color=INK)
        ax.axvline(19, color="#9DA9B2", lw=.6, ls=":", zorder=1)
        ax.axhline(0, color="#9DA9B2", lw=.7); ax.grid(axis="y", color=LIGHT, lw=.6)
        ax.set_xlabel("Layer", labelpad=3)
        for p, label in matched:
            sub = df[df.probe.eq(p)].sort_values("layer")
            line, = ax.plot(sub.layer, sub.delta_target, color=PALETTE[p], marker=MARKERS[p], markevery=4,
                            ms=3, lw=1.35, ls="--" if p == "resynth_glvocos" else "-", label=label)
            if ssl == "wavlm":
                handles.append((p, line))
    axes[0].set_ylabel(r"Target shift $\Delta P_t$", labelpad=5)
    axes[1].tick_params(labelleft=False)
    handles = [h for q in legend_order for p, h in handles if p == q]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.56, 1), ncol=2, frameon=False,
               handlelength=2.1, handletextpad=.35, columnspacing=.8, borderaxespad=0, labelspacing=.3)
    assert_inside(fig, "fig2_interv_layers")
    save(fig, "fig2_interv_layers")


if __name__ == "__main__":
    fig_layerwise()
    fig_intervention()
    fig_robustAB()
    fig_retention()
    fig_interv_layers()
