#!/usr/bin/env python3
"""Study-level inner-val scoring. No imagesTs, no RUN_MANIFEST, no test selection."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_locked import paired_bootstrap, score_pair  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--pred-b", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    scored = score_pair(args.gt, args.pred)
    report = {"tag": args.tag, "arm": scored, "split": "inner-val", "used_for": "development_only"}
    if args.pred_b:
        scored_b = score_pair(args.gt, args.pred_b)
        report["arm_b"] = scored_b
        from eval_locked import load_lock

        report["bootstrap_a_minus_b"] = paired_bootstrap(scored, scored_b, load_lock())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
