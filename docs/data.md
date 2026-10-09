# Data and study split

Source: [Danilov IV-OCT dataset, Zenodo record 14478210](https://zenodo.org/records/14478210). [Original dataset code](https://github.com/ViacheslavDanilov/oct_segmentation).

## Working dataset

| Partition | Frames | Study IDs | Role |
| --- | ---: | ---: | --- |
| Training | 15,556 | 74 | Weight optimization |
| Inner-validation | 3,938 | 19 | Development and checkpoint selection |
| Same-source holdout (`imagesTs`) | 1,899 | 10 | Frozen-model evaluation |
| Total | 21,393 | 103 | Working release used in this evaluation |

The three sets have no overlapping study IDs. The split uses source study identifiers. No separate clinical-record patient key was available for an additional linkage audit.

The 10 holdout IDs are `008, 009, 017, 028, 029, 057, 072, 077, 086, 087`. They were held out during local dataset construction. This partition is a same-source holdout, not an external cohort.

The source record's headline frame count and the working release count differ. The verified working count for these experiments is 21,393. The earlier 3,899-frame internal fold was a subdivision of the 19,494-frame development pool and did not provide a study-disjoint evaluation. It is not the holdout partition described here.

## Files and fingerprints

- [`../splits/splits_final.json`](../splits/splits_final.json): the nnU-Net fold 0 training and validation frame IDs.
- [`../splits/study_split_manifest.json`](../splits/study_split_manifest.json): study sets, frame counts, and test-study frame counts.
- [`../splits/train_studies.txt`](../splits/train_studies.txt), [`../splits/val_studies.txt`](../splits/val_studies.txt), and [`../splits/imagesTs_studies.txt`](../splits/imagesTs_studies.txt): plain study lists.

| Object | SHA256 |
| --- | --- |
| Full `splits_final.json` file | `9bd8f160fc9528a453f55dec6610c00a984343b79e327aaf20f0e0b558824b68` |
| Sorted 19-study inner-validation list | `40670c16c24b61d20cf4be5e6577a6992c7983d5116d0869685d9e5740fb71e6` |

The second digest hashes `','.join(sorted(study_ids)).encode('utf-8')` with no trailing newline. The two digests refer to different objects. Run `python3 scripts/verify_bundle.py` from the repository root to verify both.

## Preparation scope

Download and use the data through the source distribution. Preserve the label mapping: background 0, lumen 1, fibrous cap 2, lipid core 3, VV 4. The original image/annotation conversion script and dataset metadata are not present in this small export. Reconstructing a compatible dataset requires those archived preparation files, as listed in the [run guide](training_and_inference.md).

Study prefixes alone describe partition membership. They do not prove clinical identity beyond the source identifier scheme.
