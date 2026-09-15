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
# gray hollow markers for the off-target controls (circle = F5 target, square = CosyVoice3 target).
INK, GRAY, LIGHT = "#243342", "#71808B", "#E7ECF0"
PALETTE = {"resynth_vocos": "#247BA8", "resynth_glvocos": "#C77C17", "resynth_griffinlim": "#C77C17",
           "resynth_hift3": "#258A73", "resynth_s3vc3": "#C15365", "resynth_bigvgan": "#8061A8"}
MARKERS = {"resynth_vocos": "o", "resynth_glvocos": "D", "resynth_griffinlim": "D",
           "resynth_hift3": "^", "resynth_s3vc3": "s", "resynth_bigvgan": "v"}


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


def fig_intervention(height_inches=2.53):
    """Fig. 1: four panels on shared probe rows at WavLM L0. (a) per-class prediction-rate shift
    heatmap vs clean real; (b) signed target margin S_t; (c) target-distance reduction T_t,
    titled "Distance reduction" for width; (d) relative
    alignment G_t, each with 95% speaker-bootstrap CIs. Off-target controls (EnCodec, DAC, BigVGAN)
    are scored under the F5-TTS (hollow circle) and CosyVoice3 (hollow square) targets."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
    from matplotlib.patches import Rectangle
    df = pd.read_csv(OUT / "intervention_wavlm.csv")
    d0 = df[(df.layer == 0) & (df.probe != "clean_real")].set_index("probe")
    geom = pd.read_csv(OUT / "centroid2_wavlm_L0.csv").set_index("probe")
    rows = [("resynth_vocos", "Vocos → F5", "f5tts"), ("resynth_glvocos", "GL (Vocos) → F5", "f5tts"),
            ("resynth_griffinlim", "GL (generic) → F5", "f5tts"),
            ("resynth_hift3", "HiFT → C3", "cosyvoice3"), ("resynth_s3vc3", "Token RT → C3", "cosyvoice3"),
            ("resynth_bigvgan", "BigVGAN → Index", "indextts"),
            ("resynth_encodec", "EnCodec", None), ("resynth_dac", "DAC", None)]
    rows = [(p, l, t) for p, l, t in rows if p in d0.index and (t is None or t in SYSTEMS)]
    classes = ["real"] + SYSTEMS
    CL2 = dict(CLAB); CL2.update({"cosyvoice3": "C3", "chatterbox": "Chat.", "indextts": "Index"})
    ctrl_targets = [t for t in ["f5tts", "cosyvoice3"] if t in SYSTEMS]
    n = len(rows)
    fig = plt.figure(figsize=(FULL, height_inches))
    W, H = FULL * 72, height_inches * 72
    bottom, height = 61, H - 90.4          # room below for the colorbar, above for a 2-line title
    # x0 and width in points. The heatmap sits 14 pt further left than before, using the
    # slack the row labels left at the canvas edge, so the three metric panels can spread
    # far enough apart for one-line titles. At 9 pt the titles overhang their 66-68 pt
    # panels, so the centres, not the panels, set the spacing.
    specs = [(97, 139), (253, 66), (344, 68), (436, 66)]
    axes = [fig.add_axes([x / W, bottom / H, w / W, height / H]) for x, w in specs]
    a, b, c, d = axes
    # (b) is shortened so the two-line (c) title clears it: at 9 pt the panels are only
    # 64-67 pt wide, so a one-line "Target-distance reduction" would overlap its neighbours
    for ax, title in zip(axes, [r"(a) $\Delta P$(class)", "(b) Margin",
                                "(c) Distance reduction", "(d) Alignment"]):
        ax.set_ylim(n - .5, -.5); ax.set_yticks([]); style_axis(ax)
        ax.set_title(title, pad=4, fontsize=9, color=INK)
        for y in range(0, n, 2):
            ax.axhspan(y - .5, y + .5, color="#F4F7F9", zorder=0)
        for y in [2.5, 4.5, 5.5]:                                   # F5 / C3 / Index / control groups
            if y < n - .5:
                ax.axhline(y, color="#D8E0E5", lw=.6, zorder=1)
    # ---- (a) heatmap
    vals = np.array([[d0.loc[p, f"dP_{cl}"] for cl in classes] for p, _, _ in rows])
    cmap = LinearSegmentedColormap.from_list("shifts", ["#267CA1", "#FAFBFC", "#D38A7E"])
    norm = Normalize(-.7, .7)
    im = a.imshow(vals, cmap=cmap, norm=norm, aspect="auto", interpolation="none", zorder=2)
    # the heatmap encodes a prediction-rate shift centred on zero, not a posterior probability.
    # The bar gets its own axes so the four panels keep the shared row height they are aligned on.
    cax = fig.add_axes([(specs[0][0] + 24) / W, 26 / H, (specs[0][1] - 24) / W, 4.5 / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-.7, 0, .7])
    cb.ax.set_xticklabels(["-.7", "0", ".7"], fontsize=9)
    cb.ax.tick_params(length=2, width=.55, pad=1, colors=INK)
    cb.outline.set_linewidth(.55); cb.outline.set_edgecolor(GRAY)
    a.set_xticks(np.arange(len(classes)), [CL2.get(cl, cl) for cl in classes], rotation=30, ha="right",
                 rotation_mode="anchor")
    a.set_yticks(np.arange(n), [l for _, l, _ in rows])
    a.tick_params(axis="y", length=0, pad=7)
    for i, (p, _, t) in enumerate(rows):
        a.get_yticklabels()[i].set_color(PALETTE.get(p, INK))
        tj = classes.index(t) if t else -1
        for j, v in enumerate(vals[i]):
            # print cells above the threshold, and always the boxed target cell so a
            # small hypothesized-target shift (BigVGAN -> IndexTTS) is still readable
            if abs(v) >= .10 or j == tj:
                # compact cell text: no leading zero, plain hyphen as the minus sign;
                # the real column is the only dark fill, so it alone carries white text
                a.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center", va="center",
                       color="white" if j == 0 else "black", fontsize=9, zorder=4)
        if t:
            a.add_patch(Rectangle((classes.index(t) - .5, i - .5), 1, 1, fill=False, lw=1.1,
                                  edgecolor=INK, zorder=5))
    # ---- (b)-(d) aligned interval plots
    def errorpoint(ax, value, lo, hi, y, color, marker, hollow):
        """Symbols are drawn over the interval so hollow control shapes stay recognizable.
        No visual minimum is imposed: an interval narrower than the marker stays hidden,
        which the caption states. Verified: every hidden interval sits clear of zero."""
        assert np.isfinite([value, lo, hi]).all() and lo <= value <= hi
        ax.errorbar(value, y, xerr=[[value - lo], [hi - value]], fmt=marker, color=color, ecolor=color,
                    elinewidth=.9, capsize=1.7, capthick=.7, ms=4.0, mew=.7,
                    mfc="white" if hollow else color, zorder=4)

    for ax in [b, c, d]:
        ax.axvline(0, color="#7D8991", lw=.75, zorder=2)
        ax.grid(axis="x", color="#E5EBEF", lw=.5)
    b.set_xlim(-.52, .80); b.set_xticks([-.5, 0, .5], ["-.5", "0", ".5"])
    c.set_xlim(-3.6, 14.5); c.set_xticks([0, 10])
    d.set_xlim(-2.2, 15.5); d.set_xticks([0, 10])
    b.set_xlabel(r"$S_t$", labelpad=4); c.set_xlabel(r"$T_t$ ($\times 10^{-3}$)", labelpad=4)
    d.set_xlabel(r"$G_t$ ($\times 10^{-3}$)", labelpad=4)
    ctrl_marker = {"f5tts": "o", "cosyvoice3": "s"}
    for i, (p, _, t) in enumerate(rows):
        evals = ([(t, 0., False)] if t else []) + (
            [(t2, off, True) for t2, off in zip(ctrl_targets, [-.25, .25])]
            if p in ["resynth_encodec", "resynth_dac", "resynth_bigvgan"] else [])
        for target, off, control in evals:
            marker = ctrl_marker[target] if control else MARKERS[p]
            color = GRAY if control else PALETTE[p]
            r = d0.loc[p]
            s_keys = [f"S_{target}", f"S_{target}_lo", f"S_{target}_hi"] if control else ["S_target", "S_ci_lo", "S_ci_hi"]
            errorpoint(b, *[r[k] for k in s_keys], i + off, color, marker, control)
            g = geom.loc[f"{p}_under_{target}" if control else p]
            for ax, key in [(c, "T"), (d, "G")]:
                errorpoint(ax, *[g[k] * 1e3 for k in [key, key + "_lo", key + "_hi"]], i + off, color, marker, control)
    # ---- key: the five matched shapes, then the two control shapes
    sw = DrawingArea(54, 11, 0, 0)
    for x, p in zip([5, 16, 27, 38, 49], ["resynth_vocos", "resynth_glvocos", "resynth_hift3", "resynth_s3vc3", "resynth_bigvgan"]):
        sw.add_artist(Line2D([x], [5.5], color=PALETTE[p], marker=MARKERS[p], ls="none", markersize=4.0, markeredgewidth=.7))
    groups = [HPacker(children=[sw, TextArea("Hypothesized targets", textprops={"size": 9, "color": INK})], align="center", pad=0, sep=4)]
    for marker, label in [("o", "Control / F5"), ("s", "Control / C3")]:
        s1 = DrawingArea(10, 11, 0, 0)
        s1.add_artist(Line2D([5], [5.5], color=GRAY, marker=marker, mfc="white", ls="none", markersize=4.0, markeredgewidth=.7))
        groups.append(HPacker(children=[s1, TextArea(label, textprops={"size": 9, "color": INK})], align="center", pad=0, sep=4))
    fig.add_artist(AnchoredOffsetbox(loc="lower center", child=HPacker(children=groups, align="center", pad=0, sep=16),
                                     frameon=False, bbox_to_anchor=(.57, .004), bbox_transform=fig.transFigure, borderpad=0, pad=0))
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
    matched = [("resynth_vocos", "Vocos → F5"), ("resynth_glvocos", "GL (Vocos) → F5")]
    if "cosyvoice3" in SYSTEMS:
        matched += [("resynth_hift3", "HiFT → C3"), ("resynth_s3vc3", "Token RT → C3")]
    fig = plt.figure(figsize=(COL, 2.12))
    W, H = COL * 72, 2.12 * 72
    axes = [fig.add_axes([x / W, 31 / H, 82 / W, 78 / H]) for x in [42, 155]]
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
                handles.append(line)
    axes[0].set_ylabel(r"Target shift $\Delta P_t$", labelpad=5)
    axes[1].tick_params(labelleft=False)
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
