# FermiML-Framework-Regression

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Description

Benchmarks matchgate free-fermionic Gaussian-process and support-vector regression
models against classical baselines (GP and SVR) on real and synthetic regression
datasets. The quantum kernels are based on the matchgate covariance kernel of

> Jaeger, Braccia, Bermejo, Algaba, Garcia-Martin, and Cerezo,
> *Provable and scalable quantum Gaussian processes for quantum learning*,
> arXiv:2605.00099 (2026).

### Models

| Name | Class | Description |
|------|-------|-------------|
| `ClassicalGP` | `ClassicalGPRegressor` | RBF + WhiteKernel Gaussian process (sklearn) |
| `ClassicalSVR` | `ClassicalSVR` | RBF support-vector regressor (sklearn) |
| `VanillaSVR` | `VanillaNIFKernelRegressor` | Fidelity kernel SVR with trainable linear ansatz |
| `AngleNIFGP` | `AngleNIFGPRegressor` | Covariance-kernel GP with fixed angle-embedding ansatz |
| `VanillaNIFGP` | `VanillaNIFGPRegressor` | Covariance-kernel GP with trainable linear ansatz |
| `AngleMpfPlus` | `AngleMpfPlusGPRegressor` | Multi-sector covariance-kernel GP, fixed angle ansatz on |+⟩ |
| `VanillaMpfPlus` | `VanillaMpfPlusGPRegressor` | Multi-sector covariance-kernel GP, trainable linear ansatz on |+⟩ |

All seven standardize their targets and clip their predictions to the range of targets seen in
training. For the **inputs**, all seven take an `x_scaling` chosen from `none`, `minmax01`,
`minmax11`, `standard` and `quantile`, swept identically for every model. Until 0.0.6 each model
hardcoded its own transform and only the covariance GPs swept anything, so adding one to the
matchgate kernels alone would have tuned one side of the benchmark against a pinned baseline.

The transform matters because the matchgate embedding is periodic: a feature carried in raw
units aliases, and California Housing's `Population` column reaches 8997 — about 1432 wraps —
which is enough on its own to take `VanillaNIFGP` there from `+0.69` to `−0.23`. Bounding it is
necessary but not sufficient, which is why `quantile` is in the set: min-max preserves the shape
of a heavy-tailed column, leaving 99% of that dataset's `AveOccup` inside `[0, 0.0038]` — within
one period, but occupying 0.4% of it. Four of its eight columns are squashed that way. No
transform is left on by default because magnetisation's inputs are already angles in `[0, π]`,
where rescaling breaks the exact embedding match the dataset exists to calibrate; validation
picks per cell.

The clip exists because the covariance GPs extrapolate badly on California Housing, reaching
`−84` for a target the data caps at `5.0`. It is applied to every model, not only the ones that
need it: replayed over the 5040 completed runs it moved `ClassicalGP` by `+0.0008` and
`VanillaNIFGP` from `0.5536` to `0.7024`, improved 4336 runs and made none worse, and was an
exact no-op on Friedman and magnetisation.

### Datasets

| Name | Class | Description |
|------|-------|-------------|
| `friedman` | `FriedmanDataset` | Friedman #1 synthetic (sklearn) |
| `california-housing` | `CaliforniaHousingDataset` | California Housing subsampled (sklearn) |
| `magnetisation` | `MagnetisationDataset` | Total magnetisation of a matchgate-embedded state |

`magnetisation` is the quantum-native reference: its target is a sum of two-point
Majorana observables, so it lies *exactly* in the order-`m=2` covariance kernel's
hypothesis class. It calibrates what the kernel does when its structural
assumption holds perfectly, and any advantage claimed on the other two datasets
should be read against it.

---

## Installation

With `uv` installed:

```bash
uv venv .venv
uv sync --dev
```

With `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install uv
python -m uv sync --dev
```

---

## Quick start — notebook benchmark

Open `notebooks/benchmark_sklearn_datasets.ipynb` for an interactive CV benchmark
of the models on a single dataset. Adjust the dataset loader in the first
parameter cell and run all cells.

---

## Reproducing the full benchmark

### 1. Get the experiment config

The config lives at `data/configs/config.yaml` and is tracked in the repository.
A `git pull` is sufficient after cloning.

### 2. Build the Apptainer image

Build and convert the Docker image to a `.sif` file, then place it at:

```
data/dockerfiles/fermiml-framework-regression-0.0.4.sif
```

The launchers choose the image from `uname -m` on the node they run on, so an
aarch64 cluster needs its own build alongside the first:

```
data/dockerfiles/fermiml-framework-regression-0.0.4-arm64.sif
```

Check with `uname -m` on the target cluster before building it — the Alliance
clusters this workflow runs on are x86_64, so normally only the first is needed.

### 3. Set up the login-node environment

```bash
bash workflow/setup.sh
```

Installs Snakemake and the SLURM executor plugin into `.venv/` and patches the
plugin. Login node only — compute nodes have no outbound internet. Idempotent,
and `submit.sh` calls it for you.

### 4. Populate the dataset cache

```bash
mkdir -p datasets
apptainer exec -B data:/app/data,datasets:/app/datasets \
    data/dockerfiles/fermiml-framework-regression-0.0.4.sif \
    sh -c "cd /app && prepare_datasets --exp_config_file data/configs/config.yaml"
```

California Housing downloads through scikit-learn, and compute nodes have no
outbound internet, so the cache has to be filled first into the directory the
Snakefile binds as `/app/datasets`.

This runs **inside the image**: `workflow/setup.sh` installs only Snakemake and
the SLURM executor plugin, so scikit-learn and this package are not importable
from a login node. Add `--all_configs` to construct every dataset cell in the
grid rather than one per dataset, which also pre-flights the dataset axis.

### 5. Dry-run (preview jobs and partition routing without submitting)

```bash
bash workflow/dry_run.sh
```

Read the `[partition]` lines: they report which cluster was detected and how the
sbatch `-p` flag resolves on it.

### 6. Submit to the cluster

```bash
bash workflow/submit.sh try     # 3-job smoke test — do this first on a new cluster
bash workflow/submit.sh         # the full sweep
```

Use `submit.sh`, **not** `sbatch workflow/run.sh`: the account and partitions
differ per cluster and an `#SBATCH` line cannot be computed, so `submit.sh`
resolves them from the active cluster profile and passes them to `sbatch`.

Snakemake submits one independent SLURM job per unfinished experiment config.
Jobs whose `state.finished` file already exists are automatically skipped —
re-running the command safely resumes where it left off, and a hash whose SLURM
job is still live from an earlier master is never double-submitted.

**Common flags** (append to `bash workflow/submit.sh`, they reach Snakemake):

| Flag | Effect |
|------|--------|
| `--forceall` | Re-run all jobs, even finished ones |
| `--jobs N` | Override the cap on concurrent SLURM submissions |
| `--config slurm_partitions=false` | Stop naming partitions; let the executor plugin decide |
| `--config skip_running=false` | Disable the live-job guard |

### 7. Monitor progress

```bash
python workflow/job_status.py              # how much of the sweep is actually done
bash workflow/jobs_status.sh               # squeue counts + head, refreshed every 30 s
squeue -u $USER -h -r -O state | uniq -c   # counts by state
```

The two answer different questions. `jobs_status.sh` watches the *queue*, so it goes quiet
the moment the last job ends and says nothing about what the jobs produced.
`job_status.py` reads the *results tree*: it counts the expected jobs against
`state.finished`, breaks them down per model and per dataset, and sorts what is missing into
why — an infeasible `sector_m`, a non-PSD kernel, an OOM kill, a walltime cut. It also finds
jobs that hold complete results but lost the marker Snakemake keys on, which would otherwise
be silently recomputed; `--repair` writes those markers back (`--dry_run` to preview).

Everything an experiment produces lives under `data/{experiment_name}/`, so
rsyncing that one folder back brings the logs with the results:

| Path | Contents |
|------|----------|
| `data/{experiment_name}/results/{hash}/` | per-job results, metrics and `snakemake.log` |
| `data/{experiment_name}/logs/run_<jobid>.log` | the master (necromancer) log |
| `data/{experiment_name}/logs/try_3_<jobid>.log` | the same, for the smoke test |

---

## Running on several clusters

Runs on **iq**, **fir**, **narval**, **nibi** and **rorqual**. The cluster is detected
(`$FMR_CLUSTER`, then `$CC_CLUSTER`, then `sinfo`, then the hostname), never
silently defaulted, and each one's account, RAM-per-core ratio, core cap and
partition policy live in its own profile. To add a cluster, copy the closest
neighbour's profile and adjust it — provided it schedules partial nodes.

**trillium is not supported**: it allocates whole 192-core nodes, forbids `--mem`,
and mounts `/project` read-only on compute nodes, so one job per config hash does
not apply there. `workflow/profiles/trillium/config.yaml` records the detail.

```bash
FMR_CLUSTER=rorqual bash workflow/submit.sh    # force the cluster
bash workflow/probe_partitions.sh              # check a cluster before launching (read-only)
```

Run the probe on any new cluster. Its last two steps are the ones that catch what
`scontrol` and `sinfo` cannot: an account check that mirrors snakemake's own
(which rejects accounts you are not *associated* with, a stricter rule than
SLURM's), and an `sbatch --test-only` that validates a real job without queueing
it, catching a forbidden `--mem` or whole-node-only scheduling.

See [`workflow/README.md`](workflow/README.md) for the full procedure, the
partition-routing regimes, and the SLURM plugin patch.

---

## Customising resources or the experiment

| File | Purpose |
|------|---------|
| `workflow/config.yaml` | Active SIF version and config file path |
| `workflow/profiles/<cluster>/config.yaml` | Per-cluster account, RAM/core, core cap, job cap |
| `workflow/Snakefile` | Rule definitions, partition routing, core sizing, priorities |
| `data/configs/config.yaml` | Dataset/model/HP grid for the benchmark |

To switch experiments, update `config_file` in `workflow/config.yaml`.

---

## Analysing results

Open `notebooks/benchmark_sklearn_datasets.ipynb` for a quick CV benchmark
or adapt the analysis notebook in `notebooks/` for the full cluster results.

---

## References

- arXiv:2605.00099 — matchgate covariance kernel
- [Better than classical? The subtle art of benchmarking QML models](https://arxiv.org/abs/2403.07059) — benchmarking methodology

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## AI usage disclosure

The author acknowledges the use of AI-assisted tools during the implementation
of this project to assist debugging and improve code quality. AI tools were also
used to generate unit tests. The author has reviewed, edited, and validated all
AI-assisted content and assumes full responsibility for the final version of the
code.

---

## Citation

```
```
