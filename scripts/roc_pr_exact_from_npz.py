#!/usr/bin/env python3
"""Exact float32 pixel AUROC/AP — one mergesort pass per class (no 512-bin)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CLASS_NAME = {1: "lumen", 2: "fibrous_cap", 3: "lipid", 4: "vv"}
FOREGROUND = (1, 2, 3, 4)


def load_prob(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as z:
        pr = z["probabilities"]
    if pr.ndim == 4 and pr.shape[1] == 1:
        pr = pr[:, 0]
    return pr.astype(np.float32, copy=False)


def n_pixels(pred_dir: Path, gt_dir: Path) -> int:
    total = 0
    for npz in pred_dir.glob("*.npz"):
        total += np.array(Image.open(gt_dir / f"{npz.stem}.png")).size
    return total


def fill(pred_dir: Path, gt_dir: Path, cls: int, n: int):
    scores = np.empty(n, dtype=np.float32)
    y = np.empty(n, dtype=np.uint8)
    i = 0
    files = sorted(pred_dir.glob("*.npz"))
    for k, npz in enumerate(files):
        pr = load_prob(npz)
        gt = np.asarray(Image.open(gt_dir / f"{npz.stem}.png"))
        m = gt.size
        scores[i : i + m] = pr[cls].ravel()
        y[i : i + m] = (gt.ravel() == cls)
        i += m
        if (k + 1) % 400 == 0:
            print(f"  filled {k+1}/{len(files)}", flush=True)
    assert i == n
    return scores, y


def metrics_from_scores(y: np.ndarray, scores: np.ndarray, max_plot: int = 2500):
    """AUROC/AP with distinct score thresholds (ties merged).

    Equal probabilities are one threshold: TP/FP from that group are added
    together before emitting a ROC/PR point. This removes dependence on
    within-tie pixel order (mergesort / array fill order).
    """
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both positive and negative pixels")

    # ascending unique thresholds; process high→low
    order = np.argsort(scores, kind="mergesort")
    scores_s = scores[order]
    y_s = y[order]

    # group boundaries over equal scores
    # find starts of runs of equal values in ascending array, then walk descending
    change = np.flatnonzero(scores_s[1:] != scores_s[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(scores_s)]])

    fpr_pts = [0.0]
    tpr_pts = [0.0]
    prec_pts = []
    rec_pts = []
    tp = 0.0
    fp = 0.0
    # high score first: reverse groups
    for g0, g1 in zip(starts[::-1], ends[::-1]):
        grp = y_s[g0:g1]
        tp += float(grp.sum())
        fp += float(grp.size - grp.sum())
        tpr = tp / n_pos
        fpr = fp / n_neg
        prec = tp / (tp + fp)
        rec = tpr
        fpr_pts.append(fpr)
        tpr_pts.append(tpr)
        prec_pts.append(prec)
        rec_pts.append(rec)

    fpr_a = np.asarray(fpr_pts, dtype=np.float64)
    tpr_a = np.asarray(tpr_pts, dtype=np.float64)
    # AUROC: trapezoid on threshold ROC (tied mass moves as one step → correct mid-rank)
    auroc = float(np.trapezoid(tpr_a, fpr_a))

    # AP (sklearn): sum (R_n - R_{n-1}) P_n over increasing recall after each threshold
    rec_a = np.asarray(rec_pts, dtype=np.float64)
    prec_a = np.asarray(prec_pts, dtype=np.float64)
    # prepend recall=0 for the sum; sklearn uses precision_recall_curve then this sum
    r = np.concatenate([[0.0], rec_a])
    p = np.concatenate([[prec_a[0]], prec_a])  # unused; use stepwise on rec_a
    ap = 0.0
    prev_r = 0.0
    for r_i, p_i in zip(rec_a, prec_a):
        if r_i > prev_r:
            ap += (r_i - prev_r) * p_i
            prev_r = r_i
    ap = float(ap)

    # downsample for plotting (one point per distinct score; may still be large)
    n_thr = len(rec_a)
    if n_thr > max_plot:
        idx = np.linspace(0, n_thr - 1, num=max_plot, dtype=np.int64)
        idx = np.unique(np.concatenate([[0], idx, [n_thr - 1]]))
    else:
        idx = np.arange(n_thr)
    # ROC includes origin
    roc_idx = np.concatenate([[0], idx + 1])

    return {
        "auroc": auroc,
        "average_precision": ap,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": n_pos / y.size,
        "n_distinct_thresholds": int(len(starts)),
        "curve": {
            "fpr": fpr_a[roc_idx],
            "tpr": tpr_a[roc_idx],
            "precision": prec_a[idx],
            "recall": rec_a[idx],
        },
    }


def eval_arm(pred_dir: Path, gt_dir: Path, tag: str) -> dict:
    print(f"[{tag}] counting", flush=True)
    n = n_pixels(pred_dir, gt_dir)
    print(f"[{tag}] n={n}", flush=True)
    out = {"tag": tag, "n_pixels": n, "dtype": "float32", "classes": {}, "_curves": {}}
    for cls in FOREGROUND:
        name = CLASS_NAME[cls]
        print(f"[{tag}] {name} load+sort", flush=True)
        scores, y = fill(pred_dir, gt_dir, cls, n)
        m = metrics_from_scores(y, scores)
        del scores, y
        out["classes"][name] = {
            "auroc": m["auroc"],
            "average_precision": m["average_precision"],
            "n_pos_pixels": m["n_pos"],
            "n_neg_pixels": m["n_neg"],
            "prevalence": m["prevalence"],
        }
        out["_curves"][name] = m["curve"]
        print(
            f"[{tag}] {name} AUROC={m['auroc']:.6f} AP={m['average_precision']:.6g} prev={m['prevalence']:.6g}",
            flush=True,
        )
    return out


def plot_pair(a, b, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [CLASS_NAME[c] for c in FOREGROUND]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, name in zip(axes.ravel(), names):
        for lab, rep, color in (("A", a, "#1f77b4"), ("B", b, "#d62728")):
            cur = rep["_curves"][name]
            m = rep["classes"][name]
            ax.plot(cur["fpr"], cur["tpr"], color=color, lw=1.5, label=f"{lab} AUROC={m['auroc']:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(name)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle(f"Pixel ROC (exact float32) — {tag}")
    fig.tight_layout()
    roc_p = out_dir / f"{tag}_roc_exact.png"
    fig.savefig(roc_p, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, name in zip(axes.ravel(), names):
        prev = a["classes"][name]["prevalence"]
        ax.axhline(prev, color="0.5", ls=":", lw=1, label=f"prevalence={prev:.2e}")
        for lab, rep, color in (("A", a, "#1f77b4"), ("B", b, "#d62728")):
            cur = rep["_curves"][name]
            m = rep["classes"][name]
            ax.plot(
                cur["recall"],
                cur["precision"],
                color=color,
                lw=1.5,
                label=f"{lab} AP={m['average_precision']:.4g}",
            )
        ax.set_title(name)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle(f"Pixel PR (exact AP ≠ trapezoidal AUPRC) — {tag}")
    fig.tight_layout()
    pr_p = out_dir / f"{tag}_pr_exact.png"
    fig.savefig(pr_p, dpi=160)
    plt.close(fig)
    return roc_p, pr_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--pred-a", type=Path, required=True)
    ap.add_argument("--pred-b", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    ra = eval_arm(args.pred_a, args.gt, "A")
    rb = eval_arm(args.pred_b, args.gt, "B")
    roc_p, pr_p = plot_pair(ra, rb, args.out_dir, args.tag)
    summary = {
        "tag": args.tag,
        "kind": "pixel_softmax_exact_float32_tie_merged",
        "supersedes_512bin": True,
        "supersedes_per_pixel_order_sensitive": True,
        "withdraw_vv_auroc_approx_random_claim": True,
        "metric_definitions": {
            "AUROC": "trapezoid on ROC after grouping equal float32 scores into one threshold (tie-merged; order-independent)",
            "AP": "sum (Δrecall)*precision over distinct score thresholds (sklearn average_precision; not trapezoidal AUPRC)",
            "not": "trapezoidal AUPRC; not 512-bin histogram; not within-tie pixel order",
            "prevalence": "positive_pixels/all_pixels",
        },
        "arm_a": {k: v for k, v in ra.items() if k != "_curves"},
        "arm_b": {k: v for k, v in rb.items() if k != "_curves"},
        "figures": {"roc": str(roc_p), "pr": str(pr_p)},
    }
    out = args.out_dir / f"{args.tag}_roc_pr_exact.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(out, flush=True)


if __name__ == "__main__":
    main()
