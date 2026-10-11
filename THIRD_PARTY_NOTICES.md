# Third-party materials

The root MIT license applies to Eigen-PID's own code. It does not grant a
license to third-party source or datasets.

## Bundled Gaussian reference implementations

- **GPID / Tilde-PID**, Praveen Venkatesh and contributors:
  https://github.com/praveenv253/gpid, revision
  `6a2b15e637681f329590755464ee01c11dd66d27`.
  The upstream package metadata declares MIT. The bundled files are under
  `experiments/external/gpid/`; original metadata and attribution are retained.
- **Flow-PID / Thin-PID**, Warren Zhao and contributors:
  https://github.com/warrenzha/flow-pid, revision
  `4a30b1a3ff3eb9d181fc49dff023930a617a86b3`.
  The upstream repository did not specify a software license at the time of
  access. These files are under `experiments/external/flow-pid/` and are not
  covered by Eigen-PID's MIT license. This repository does not assert that
  redistribution permission has been obtained.

The local initialization and regularization-evaluation modifications are
listed in [the baseline README](experiments/external/README.md). Neither
baseline is included in the Eigen-PID wheel or source distribution.

## fMRI data and optional reference benchmarks

NSD/Algonauts responses and surface meshes are not redistributed. Obtain data
through the upstream access process and follow its terms. Saved analysis
results and the reproduction workflow are documented in
[the fMRI README](fmri/README.md), including the separate upstream revisions
used for optional reference benchmarks.
