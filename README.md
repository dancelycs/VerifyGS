# VerifyGS

<p align="center">
  <a href="https://dancelycs.github.io/VerifyGS/"><strong>View the complete result gallery</strong></a>
</p>

Official implementation and evaluation records for **VerifyGS: Conformal and Gaussian-Consistent Verification of 3D Gaussian Map Registration**.

VerifyGS evaluates a source-to-target similarity candidate from two Gaussian maps. It compares bidirectional center support with a deterministic, map-pair-conditioned random-similarity background, calibrates the resulting score with a failure-class split-conformal rank, and applies a transported covariance--color consistency veto. Images, camera poses, correspondences, and generator-specific confidence values are not required.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[evaluation,test]"
pytest -q
python reproduce.py
```

`reproduce.py` reads the released per-candidate records and recreates the paper tables and quantitative figures under `reproduced/`.

## Released evaluation records

The repository contains the complete evaluated outputs rather than selected examples. Each main benchmark directory provides observable evidence and the final verification score; `data/candidates/` contains the corresponding public registration transformations.

| Benchmark | Complete records | Candidate transformations |
|---|---|---|
| ScanNet-GSReg | [`evidence.json`](data/scannet/evidence.json), [`null.json`](data/scannet/null.json) | [`data/candidates/scannet/`](data/candidates/scannet/) |
| ETH3D | [`evidence.json`](data/eth3d/evidence.json), [`null.json`](data/eth3d/null.json) | [`data/candidates/eth3d/`](data/candidates/eth3d/) |
| BEVGSReg | [`evidence.json`](data/bevgsreg/evidence.json), [`null.json`](data/bevgsreg/null.json) | [`data/candidates/bevgsreg/`](data/candidates/bevgsreg/) |

The [`data/objaverse/`](data/objaverse/) directory is retained separately as an additional object-centric stress test. It is not used to calibrate the verifier or to support the main cross-dataset claims.

The locked scene-disjoint split and baseline thresholds are stored in [`data/protocol.json`](data/protocol.json). The final conformal and Gaussian-gate results, including repeated scene-disjoint partitions, are stored in [`data/gaussian_conformal_gate.json`](data/gaussian_conformal_gate.json), while [`data/final_calibration.json`](data/final_calibration.json) contains the fixed failure scores and Gaussian affinity floor used by the command-line verifier. Ground-truth errors and success labels define calibration populations and evaluation outcomes only; the verifier never reads them when scoring deployment candidates.

## Score one candidate

The included example creates a synthetic pair and scores its correct transform:

```bash
python examples/make_synthetic_pair.py
verifygs --pair examples/synthetic_pair.npz \
  --candidate examples/candidate.json --null-count 64 \
  --calibration data/final_calibration.json
```

The transformation convention is

```text
x_target = scale * (x_source @ rotation.T) + translation
```

A pair file must contain `source_means` and `target_means` with shape `[N,3]`. Covariance arrays `[N,3,3]` and RGB arrays `[N,3]` are required for the final Gaussian-consistency decision. Without calibration scores, the command reports observable evidence but leaves the conformal decision unset.

To recompute scores for a prepared benchmark:

```bash
python scripts/recompute_scores.py \
  --data /path/to/pair_npz_directory \
  --candidates data/candidates/scannet \
  --output scannet_scores.json
```

This command loads [`data/final_calibration.json`](data/final_calibration.json) by default and therefore reports both evidence and final decisions. Pass `--no-calibration` to recompute observable evidence alone.

Raw benchmark data are not redistributed and remain subject to their original licenses.

## Results and visualizations

The [online gallery](https://dancelycs.github.io/VerifyGS/) displays aggregate metrics, full-dataset method outcomes, risk--coverage analyses, qualitative comparisons, and rendered 3DGS scenes. The same page can be opened locally at [`results_gallery/index.html`](results_gallery/index.html).

## License

The implementation is released under the MIT License. Dataset assets and third-party registration methods remain governed by their original licenses.
