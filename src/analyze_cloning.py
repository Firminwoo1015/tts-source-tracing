"""Cloning-only primary analysis (reviewer-hardened).

Conditions: real + 5 zero-shot cloning systems (Kokoro excluded — speaker
confound; it stays as an appendix sanity check).

Per SSL encoder:
  layerwise_<ssl>.csv    6-way and 5-way (TTS-only) LOO kNN acc per layer
  spkdisjoint rows       nested layer selection: outer GroupKFold(5) over
                         speakers, inner GroupKFold(4) on train speakers picks
                         the layer; macro-F1 with SPEAKER-clustered bootstrap CI
  confusion CSV/PNG      from outer-fold predictions (6-way and TTS-only)
  unsupervised rows      KMeans ARI/NMI, 5-NN neighborhood purity, and
                         between-system / between-speaker variance shares at L0
Protocol constants: k=5 cosine kNN, L2-normalized mean-pooled
embeddings. Utterance IDs are speaker-specific in LibriSpeech, so
speaker-disjoint folds are also text-disjoint.
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, confusion_matrix, f1_score,
                             normalized_mutual_info_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold, LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

ROOT = Path(os.environ.get("TTS_ANAL_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = os.environ.get("TTS_ANAL_RESULTS", "results/paper5c17")
SYSTEMS = os.environ.get("TTS_ANAL_SYSTEMS", "f5tts,xtts,cosyvoice3,chatterbox,indextts").split(",")
CLONING = SYSTEMS
EXCLUDE = set((ROOT / os.environ.get("TTS_ANAL_EXCLUDE", "data/manifests/exclude17.txt")).read_text().split())
OUT = ROOT / RESULTS
RNG = np.random.default_rng(0)


def load_all(ssl, conds):
    emb_dir = ROOT / "results" / "embeddings" / ssl
    X, cond, spk, uid = [], [], [], []
    for c in conds:
        d = np.load(emb_dir / f"{c}.npz")
        keep = ~np.isin(d["utt_ids"], list(EXCLUDE))
        X.append(d["embs"][keep])
        cond += [c] * keep.sum()
        spk += list(d["speakers"][keep])
        uid += list(d["utt_ids"][keep])
    return np.concatenate(X), np.array(cond), np.array(spk), np.array(uid)


def knn():
    return KNeighborsClassifier(n_neighbors=5, metric="cosine")


def nested_spk_disjoint(embs, y, spk):
    """Outer GroupKFold(5); inner GroupKFold(4) on train speakers selects the
    layer; returns pooled outer predictions and per-fold chosen layers."""
    outer = GroupKFold(n_splits=5)
    preds = np.empty(len(y), dtype=object)
    chosen = []
    for tr, te in outer.split(embs[:, 0, :], y, groups=spk):
        best_layer, best_f1 = 0, -1
        inner = GroupKFold(n_splits=4)
        for layer in range(embs.shape[1]):
            Xl = normalize(embs[tr][:, layer, :])
            f1s = []
            for itr, ite in inner.split(Xl, y[tr], groups=spk[tr]):
                p = knn().fit(Xl[itr], y[tr][itr]).predict(Xl[ite])
                f1s.append(f1_score(y[tr][ite], p, average="macro"))
            m = np.mean(f1s)
            if m > best_f1:
                best_f1, best_layer = m, layer
        chosen.append(best_layer)
        Xtr = normalize(embs[tr][:, best_layer, :])
        Xte = normalize(embs[te][:, best_layer, :])
        preds[te] = knn().fit(Xtr, y[tr]).predict(Xte)
    return preds.astype(str), chosen


def speaker_bootstrap_f1(y, pred, spk, n=1000):
    speakers = np.unique(spk)
    scores = []
    for _ in range(n):
        s = RNG.choice(speakers, len(speakers), replace=True)
        idx = np.concatenate([np.where(spk == sp)[0] for sp in s])
        scores.append(f1_score(y[idx], pred[idx], average="macro"))
    return np.percentile(scores, [2.5, 97.5])


def variance_decomposition(X, cond, spk):
    """Share of total variance explained by system means vs speaker means."""
    X = normalize(X)
    mu = X.mean(0)
    total = ((X - mu) ** 2).sum()
    between_sys = sum(((X[cond == c].mean(0) - mu) ** 2).sum() * (cond == c).sum()
                      for c in np.unique(cond))
    between_spk = sum(((X[spk == s].mean(0) - mu) ** 2).sum() * (spk == s).sum()
                      for s in np.unique(spk))
    return between_sys / total, between_spk / total


def confusion_fig(y, pred, labels, title, fname):
    cm = confusion_matrix(y, pred, labels=labels, normalize="true")
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(OUT / f"{fname}.csv")
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=7)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUT / f"{fname}.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssl", default="wavlm")
    args = ap.parse_args()
    ssl = args.ssl
    OUT.mkdir(parents=True, exist_ok=True)

    summary = []
    for tag, conds in [(f"{len(CLONING) + 1}way", ["real"] + CLONING), ("ttsonly", CLONING)]:
        embs, cond, spk, uid = load_all(ssl, conds)

        rows = [{"layer": l,
                 "loo_acc": cross_val_score(knn(), normalize(embs[:, l, :]),
                                            cond, cv=LeaveOneOut(),
                                            n_jobs=8).mean()}
                for l in range(embs.shape[1])]
        pd.DataFrame(rows).to_csv(OUT / f"cloning_{tag}_layers_{ssl}.csv",
                                  index=False)

        pred, chosen = nested_spk_disjoint(embs, cond, spk)
        f1 = f1_score(cond, pred, average="macro")
        lo, hi = speaker_bootstrap_f1(cond, pred, spk)
        prfs = precision_recall_fscore_support(cond, pred, labels=conds)
        pd.DataFrame({"class": conds, "precision": prfs[0], "recall": prfs[1],
                      "f1": prfs[2]}).to_csv(
            OUT / f"cloning_{tag}_perclass_{ssl}.csv", index=False)
        confusion_fig(cond, pred, conds,
                      f"{ssl} {tag} speaker-disjoint (nested layer sel.)",
                      f"cloning_{tag}_confusion_{ssl}")

        vd_sys, vd_spk = variance_decomposition(embs[:, 0, :], cond, spk)
        km = KMeans(n_clusters=len(conds), n_init=20, random_state=0).fit(
            normalize(embs[:, 0, :]))
        ari = adjusted_rand_score(cond, km.labels_)
        nmi = normalized_mutual_info_score(cond, km.labels_)
        nn = knn().fit(normalize(embs[:, 0, :]), cond)
        dist, ind = nn.kneighbors(normalize(embs[:, 0, :]), n_neighbors=6)
        purity = (cond[ind[:, 1:]] == cond[:, None]).mean()

        summary.append({"ssl": ssl, "setting": tag,
                        "spkdisjoint_macro_f1": f1, "ci_lo": lo, "ci_hi": hi,
                        "chosen_layers": str(chosen),
                        "kmeans_ari_L0": ari, "kmeans_nmi_L0": nmi,
                        "nn_purity_L0": purity,
                        "var_between_system_L0": vd_sys,
                        "var_between_speaker_L0": vd_spk})
        print(summary[-1])

    df = pd.DataFrame(summary)
    out = OUT / f"cloning_summary_{ssl}.csv"
    df.to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
