#!/usr/bin/env python3
"""Study-level scoring for the revision A/B. Rules are locks/eval_rules.json.

Does not choose checkpoints or stop training. Refuses checkpoint_best.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

LOCK_PATH = Path("/root/IVOCT_RadialMamba/revision_ab/locks/eval_rules.json")
CLASSES = (1, 2, 3, 4)
CLASS_NAME = {1: "lumen", 2: "fibrous_cap", 3: "lipid", 4: "vv"}


def load_lock() -> dict:
    lock = json.loads(LOCK_PATH.read_text())
    if lock["checkpoint"] != "checkpoint_final.pth":
        raise SystemExit("eval lock checkpoint drifted")
    if lock["aggregation"] != "study":
        raise SystemExit("eval lock aggregation drifted")
    return lock


def study_of(stem: str) -> str:
    return stem.split("_", 1)[0]


def frame_counts(gt: np.ndarray, pr: np.ndarray, cls: int) -> tuple[int, int, int]:
    g = gt == cls
    p = pr == cls
    tp = int(np.logical_and(g, p).sum())
    fp = int(np.logical_and(~g, p).sum())
    fn = int(np.logical_and(g, ~p).sum())
    return tp, fp, fn


def dice_from_counts(tp: int, fp: int, fn: int):
    if tp + fp + fn == 0:
        return None
    return 2 * tp / (2 * tp + fp + fn)


def recall_from_counts(tp: int, fn: int):
    if tp + fn == 0:
        return None
    return tp / (tp + fn)


def check_probabilities(npz_path: Path, lock: dict) -> dict:
    if not npz_path.is_file():
        return {"ok": False, "reason": "missing npz"}
    with np.load(npz_path) as z:
        if "probabilities" not in z.files:
            return {"ok": False, "reason": f"keys={z.files}"}
        prob = z["probabilities"]
    if prob.ndim != 3 or prob.shape[0] < 2:
        return {"ok": False, "reason": f"shape={getattr(prob, 'shape', None)}"}
    total = prob.sum(axis=0)
    err = float(np.max(np.abs(total - 1.0)))
    ok = err < 1e-2
    return {"ok": ok, "max_abs_sum_error": err, "shape": list(prob.shape), "rule": lock["roc_pr"]}


def score_pair(gt_dir: Path, pred_dir: Path) -> dict:
    files = sorted(p for p in pred_dir.glob("*.png"))
    if not files:
        raise SystemExit(f"no png predictions in {pred_dir}")
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASSES}
    studies = {}
    vv_pos_frames = {"tp": 0, "fp": 0, "fn": 0, "n": 0}
    vv_neg = {"n": 0, "frames_with_fp": 0, "fp_pixels": 0}
    missing_gt = []
    for pred_path in files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.is_file():
            missing_gt.append(pred_path.name)
            continue
        gt = np.array(Image.open(gt_path))
        pr = np.array(Image.open(pred_path))
        if gt.shape != pr.shape:
            raise SystemExit(f"shape mismatch {pred_path.name}: {gt.shape} vs {pr.shape}")
        sid = study_of(pred_path.stem)
        bucket = studies.setdefault(sid, {c: {"tp": 0, "fp": 0, "fn": 0, "pos_frames": 0, "neg_frames": 0, "neg_fp_frames": 0} for c in CLASSES})
        for c in CLASSES:
            tp, fp, fn = frame_counts(gt, pr, c)
            for dest in (per_class[c], bucket[c]):
                dest["tp"] += tp
                dest["fp"] += fp
                dest["fn"] += fn
            if tp + fn > 0:
                bucket[c]["pos_frames"] += 1
            else:
                bucket[c]["neg_frames"] += 1
                if fp > 0:
                    bucket[c]["neg_fp_frames"] += 1
        tp, fp, fn = frame_counts(gt, pr, 4)
        if tp + fn > 0:
            vv_pos_frames["n"] += 1
            vv_pos_frames["tp"] += tp
            vv_pos_frames["fp"] += fp
            vv_pos_frames["fn"] += fn
        else:
            vv_neg["n"] += 1
            vv_neg["fp_pixels"] += fp
            if fp > 0:
                vv_neg["frames_with_fp"] += 1
    if missing_gt:
        raise SystemExit(f"missing GT for {len(missing_gt)} files, e.g. {missing_gt[:3]}")

    def pack(counts: dict) -> dict:
        return {
            "dice": dice_from_counts(counts["tp"], counts["fp"], counts["fn"]),
            "recall": recall_from_counts(counts["tp"], counts["fn"]),
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
        }

    study_rows = []
    for sid, by_c in sorted(studies.items()):
        row = {"study": sid}
        for c in CLASSES:
            row[CLASS_NAME[c]] = pack(by_c[c])
            row[CLASS_NAME[c]]["pos_frames"] = by_c[c]["pos_frames"]
            row[CLASS_NAME[c]]["neg_frames"] = by_c[c]["neg_frames"]
            row[CLASS_NAME[c]]["neg_fp_frames"] = by_c[c]["neg_fp_frames"]
        study_rows.append(row)

    main = {}
    for c in CLASSES:
        vals = [r[CLASS_NAME[c]]["dice"] for r in study_rows if r[CLASS_NAME[c]]["dice"] is not None]
        main[CLASS_NAME[c]] = {
            "study_mean_dice": float(np.mean(vals)) if vals else None,
            "n_studies_in_mean": len(vals),
            "n_studies_excluded_empty": len(study_rows) - len(vals),
        }
    pos_studies = sum(1 for r in study_rows if r["vv"]["tp"] + r["vv"]["fn"] > 0)
    vv = {
        "n_studies": len(study_rows),
        "n_positive_studies": pos_studies,
        "n_positive_frames": vv_pos_frames["n"],
        "n_negative_frames": vv_neg["n"],
        "dice_on_positive_frames_pooled": dice_from_counts(vv_pos_frames["tp"], vv_pos_frames["fp"], vv_pos_frames["fn"]),
        "recall_on_positive_frames": recall_from_counts(vv_pos_frames["tp"], vv_pos_frames["fn"]),
        "n_negative_frames_with_any_fp": vv_neg["frames_with_fp"],
        "fp_pixels_on_negative_frames": vv_neg["fp_pixels"],
    }
    return {"n_frames": len(files), "main_study_mean": main, "vv": vv, "studies": study_rows}


def paired_bootstrap(a: dict, b: dict, lock: dict) -> dict:
    spec = lock["bootstrap"]
    by_a = {r["study"]: r for r in a["studies"]}
    by_b = {r["study"]: r for r in b["studies"]}
    ids = sorted(set(by_a) & set(by_b))
    if not ids:
        raise SystemExit("no shared studies")
    rng = np.random.default_rng(spec["seed"])
    out = {"n_studies": len(ids), "n": spec["n"], "seed": spec["seed"], "contrasts": {}}
    for c in CLASSES:
        name = CLASS_NAME[c]
        diffs = np.empty(spec["n"], dtype=np.float64)
        for i in range(spec["n"]):
            draw = rng.choice(len(ids), size=len(ids), replace=True)
            da, db = [], []
            for j in draw:
                sid = ids[j]
                va = by_a[sid][name]["dice"]
                vb = by_b[sid][name]["dice"]
                if va is None or vb is None:
                    continue
                da.append(va)
                db.append(vb)
            diffs[i] = (float(np.mean(da) - np.mean(db)) if da else np.nan)
        valid = diffs[np.isfinite(diffs)]
        out["contrasts"][name] = {
            "mean_a_minus_b": float(np.mean(valid)) if len(valid) else None,
            "ci95": [float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))] if len(valid) else None,
        }
    return out


def assert_manifest(pred_dir: Path, lock: dict) -> dict:
    manifest = pred_dir.parent / "RUN_MANIFEST.json"
    if not manifest.is_file():
        raise SystemExit(f"missing {manifest}")
    meta = json.loads(manifest.read_text())
    if meta.get("checkpoint") != lock["checkpoint"]:
        raise SystemExit(f"refusing checkpoint {meta.get('checkpoint')}")
    return meta


def self_test() -> None:
    lock = load_lock()
    assert dice_from_counts(0, 0, 0) is None
    assert dice_from_counts(0, 5, 0) == 0
    assert recall_from_counts(0, 0) is None
    assert abs(dice_from_counts(2, 0, 2) - (4 / 6)) < 1e-9
    gt = np.zeros((4, 4), dtype=np.uint8)
    pr = np.zeros((4, 4), dtype=np.uint8)
    gt[0, 0] = 4
    pr[0, 0] = 4
    pr[1, 1] = 4
    tp, fp, fn = frame_counts(gt, pr, 4)
    assert (tp, fp, fn) == (1, 1, 0)
    assert lock["bootstrap"]["unit"] == "study"
    assert "A2-B2" in lock["bootstrap"]["contrasts"]
    print("self_test_ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--gt", type=Path)
    parser.add_argument("--pred", type=Path)
    parser.add_argument("--pred-b", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    lock = load_lock()
    meta = assert_manifest(args.pred, lock)
    scored = score_pair(args.gt, args.pred)
    npz = next(args.pred.glob("*.npz"), None)
    roc = check_probabilities(npz, lock) if npz else {"ok": False, "reason": "no npz"}
    report = {"manifest": meta, "lock": str(LOCK_PATH), "arm": scored, "roc_pr": roc}
    if args.pred_b:
        meta_b = assert_manifest(args.pred_b, lock)
        scored_b = score_pair(args.gt, args.pred_b)
        report["arm_b"] = scored_b
        report["manifest_b"] = meta_b
        report["bootstrap_a_minus_b"] = paired_bootstrap(scored, scored_b, lock)
    if not roc["ok"]:
        report["roc_pr"]["action"] = "do_not_draw"
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(args.out)
    else:
        print(json.dumps({k: report[k] for k in report if k not in ("arm", "arm_b")}, indent=2))


if __name__ == "__main__":
    main()
