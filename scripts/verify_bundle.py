#!/usr/bin/env python3
"""Verify the exported files, disjoint study split, and recorded Dice arithmetic.

No images, training, or checkpoint inference are needed. This utility does
not regenerate the recorded bootstrap replicates or probability scores.
"""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]

def read_json(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))

def require(condition, message):
    if not condition:
        raise SystemExit('FAIL: ' + message)

def close(a, b, label):
    require(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), label)

def dice(tp, fp, fn):
    total = 2 * tp + fp + fn
    return None if total == 0 else 2 * tp / total

def main():
    split_bytes = (ROOT / 'splits/splits_final.json').read_bytes()
    manifest = read_json('splits/study_split_manifest.json')
    require(hashlib.sha256(split_bytes).hexdigest() == manifest['split_file_sha256'], 'split file SHA256')
    folds = json.loads(split_bytes)
    require(len(folds) == 1, 'exactly one study-disjoint fold')
    tr, va = folds[0]['train'], folds[0]['val']
    tr_ids = {x.split('_', 1)[0] for x in tr}
    va_ids = {x.split('_', 1)[0] for x in va}
    ts_ids = set(manifest['imagesTs']['studies'])
    require(not (tr_ids & va_ids or tr_ids & ts_ids or va_ids & ts_ids), 'pairwise study disjointness')
    require(len(tr) == 15556 and len(va) == 3938, 'train/validation frame counts')
    require((len(tr_ids), len(va_ids), len(ts_ids)) == (74, 19, 10), 'study counts')
    require(tr_ids == set(manifest['fold0']['train_studies']), 'training manifest')
    require(va_ids == set(manifest['fold0']['val_studies']), 'validation manifest')
    fingerprint = hashlib.sha256(','.join(sorted(va_ids)).encode()).hexdigest()
    require(fingerprint == manifest['val_study_fingerprint']['sha256'], 'validation study fingerprint')
    require(sum(manifest['imagesTs']['frames_per_study'].values()) == 1899, 'test frame count')
    for rel, expected in [('train_studies.txt', tr_ids), ('val_studies.txt', va_ids), ('imagesTs_studies.txt', ts_ids)]:
        found = set((ROOT / 'splits' / rel).read_text().split())
        require(found == expected, rel)
    print('Split verified: 74 / 19 / 10 studies, 15,556 / 3,938 / 1,899 frames.')

    with (ROOT / 'results/per_study_results.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    with (ROOT / 'results/primary_results.csv').open(encoding='utf-8-sig', newline='') as f:
        summary = list(csv.DictReader(f))
    require(len(rows) == 40, '10 studies × 4 classes in per-study CSV')
    for s in summary:
        cls = s['class']
        subset = [r for r in rows if r['class'] == cls]
        require({r['study'] for r in subset} == ts_ids, cls + ' study membership')
        av, bv, paired = [], [], []
        for r in subset:
            vals = []
            for arm in ('A', 'B'):
                d = dice(*(int(r[arm + '_' + k]) for k in ('TP','FP','FN')))
                raw = r[arm + '_dice']
                if d is None:
                    require(raw.strip().lower() in ('', 'none', 'nan', 'null'), cls + ' empty Dice')
                else:
                    close(d, float(raw), cls + ' ' + arm + ' per-study Dice')
                vals.append(d)
            a, b = vals
            if a is not None: av.append(a)
            if b is not None: bv.append(b)
            if a is not None and b is not None: paired.append(a-b)
        require(len(av) == int(s['A_valid_n']) and len(bv) == int(s['B_valid_n']), cls + ' valid counts')
        require(len(paired) == int(s['paired_common_n']), cls + ' paired count')
        close(sum(av)/len(av), float(s['A_mean']), cls + ' A mean')
        close(sum(bv)/len(bv), float(s['B_mean']), cls + ' B mean')
        close(sum(paired)/len(paired), float(s['observed_paired_A_minus_B']), cls + ' observed paired difference')
        require(float(s['CI_low']) <= float(s['CI_high']), cls + ' recorded interval order')
        print(cls + ': observed means and paired difference verified.')

    hashes = read_json('reproducibility/files_sha256.json')
    for rel, expected in hashes.items():
        p = ROOT / rel
        require(p.is_file(), 'missing export file ' + rel)
        require(hashlib.sha256(p.read_bytes()).hexdigest() == expected, 'export SHA256 ' + rel)
    print('Export file hashes verified.')
    print('PASS. Bootstrap intervals and probability metrics remain recorded results, not recomputed here.')

if __name__ == '__main__':
    main()
