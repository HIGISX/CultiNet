# CultiNet

A compact release of the CultiNet project, including the data and code for Euclidean EH-MCLP baseline comparisons, objective-weight sensitivity and ablation experiments, and the original neural-model source.

**Pretrained weights are not included.** The six-method benchmark runs Gurobi, OR-Tools, PSO, GA, SA, and NSGA-II. The neural-model source is provided separately in [`legacy_model/`](legacy_model/README.md).

## Repository layout

```text
CultiNet/
├── algo_benchmark/          # Experiment runners, configuration and shared objective
├── Algorithm/              # Existing GA, PSO, SA and NSGA-II implementations
├── data/                   # Minimal inputs for the synthetic and Xi'an cases
├── legacy_model/           # Original neural-model and training source
├── reference/              # Existing experiment results and selected locations
├── requirements.txt        # Dependencies for the four stochastic baselines
├── requirements-solvers.txt
└── requirements-model.txt
```

The Euclidean experiments use relative paths and do not require the original machine's data directories, large road-network OD files, GIS software, or cached matrices. See [`data/README.md`](data/README.md) for the input fields and preprocessing rules.

## Installation

The existing experiments used **Python 3.12.10**. Clone the repository and create a virtual environment:

```bash
git clone https://github.com/HIGISX/CultiNet.git
cd CultiNet
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Install the dependencies for all six baselines:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-solvers.txt
```

For PSO, GA, SA, and NSGA-II only, install `requirements.txt` instead. Gurobi requires a license that supports the problem size; installing `gurobipy` does not configure that license. OR-Tools does not require a Gurobi license.

The optional neural-model dependencies are listed in `requirements-model.txt`. They are separate from the EH-MCLP benchmark dependencies.

## Reproducing the experiments

Run all commands from the repository root. Generated results are written to `results/`, which is excluded from Git. Use a separate output directory for each experiment.

### Baseline comparison

```bash
python -u -X utf8 algo_benchmark/run_comparison.py --config algo_benchmark/compare_config_euclidean.json --output results/comparison
```

The default configuration runs both datasets and all six methods with **M = 30, 50, 70**. Each stochastic method runs 20 times per case, using seeds **101–120**. Each exact solver runs once per case.

Run a single dataset:

```bash
python -u -X utf8 algo_benchmark/run_comparison.py --datasets synthetic --output results/synthetic
python -u -X utf8 algo_benchmark/run_comparison.py --datasets real --output results/xian
```

Run without Gurobi:

```bash
python -u -X utf8 algo_benchmark/run_comparison.py --methods OR-Tools PSO GA SA NSGA-II --output results/no_gurobi
```

To run only the four stochastic methods, use `--methods PSO GA SA NSGA-II`. Without an exact optimality certificate, the reference is the best known solution rather than a proven optimum.

To resume an interrupted comparison, repeat the original command with `--resume`. A different configuration, input dataset, or implementation requires a new output directory.

### Objective-weight sensitivity: Xi'an, M = 30

```bash
python -u -X utf8 algo_benchmark/run_weight_sensitivity.py --config algo_benchmark/compare_config_euclidean.json --dataset real --m 30 --output results/weight_sensitivity
```

This experiment solves eight weight scenarios with Gurobi: the baseline, equal weights, and six pairwise transfers of 0.10.

### Objective-weight ablation: Xi'an, M = 30

```bash
python -u -X utf8 algo_benchmark/run_weight_ablation.py --config algo_benchmark/compare_config_euclidean.json --m 30 --output results/weight_ablation
```

The five scenarios are the full objective, equal weights, and removal of F1, F2, or F3. After removing a component, the remaining weights are renormalized proportionally. Among primary-optimal solutions, a second solve maximizes the omitted component to avoid exaggerating its decline through arbitrary tie-breaking.

Both weight experiments require an output directory that does not already exist.

### Outputs and reference results

The comparison produces `table_synthetic.csv`, `table_real.csv`, `runs.csv`, `summary.json`, `checkpoint.json`, and experiment metadata. Weight experiments produce `results.csv`, `solutions.json`, and `protocol.json`; ablation also exports `selected_sites.csv` and `table.md`.

[`reference/`](reference/README.md) contains existing summary tables and individual selections. Runtime depends on the hardware, software versions, and solver environment. Alternative optimal selections may differ while achieving the same objective.

## Configuration and model definition

The default configuration is [`algo_benchmark/compare_config_euclidean.json`](algo_benchmark/compare_config_euclidean.json).

| Setting | Default / meaning |
|---|---|
| `M_list` | `[30, 50, 70]`: number of selected facilities |
| `n_runs` / `seed_start` | `20` / `101` |
| `weights` | `[0.35, 0.40, 0.25]` |
| `service_min` | 15 minutes, converted to a Euclidean distance threshold |
| `priority_share` | Approximately 30% of the population with the lowest cultural opportunity |
| `site_capacity` | At most one selected candidate per `site_id` |
| `exact.time_limit_s` / `threads` | 600 seconds / one thread |
| `synthetic.heritage_min` | 10 minutes at 75 m/min |
| `real.heritage_min` | 30 minutes at 80 m/min |

| Dataset | Demand points | Candidate facilities | Heritage sites | Service radius |
|---|---:|---:|---:|---:|
| Synthetic | 1,000 | 200 | 60 | 1,125 m |
| Xi'an | 7,685 | 587 | 297 | 1,200 m |

Xi'an uses projected coordinates in metres: CGCS2000 / 3-degree Gauss-Kruger CM 108E (**EPSG:4545**). The synthetic coordinates are expressed in a planar metric system.

All six baselines maximize the same normalized objective:

```text
F  = 0.35 F1 + 0.40 F2 + 0.25 F3
F1 = covered population / total population
F2 = covered priority-group population / total priority-group population
F3 = linked heritage weight / total heritage weight
```

Coverage and linkage use set unions, without double counting. Each solution selects exactly M eligible candidates and satisfies the site-capacity constraint. F2 measures priority-group coverage. Across weight scenarios, compare F1/F2/F3 or evaluate selections using common baseline weights; scenario-specific weighted objectives use different scoring scales.

Synthetic heritage linkage additionally preserves the existing programme-compatibility and shortest-path river-crossing flags. Xi'an preserves the thematic compatibility rule. Its heritage grade weights are 4, 3, 2, and 1 for national, provincial, municipal, and county protection levels, respectively. These are model settings.

Edit the JSON configuration or override supported settings on the command line:

```bash
python -u -X utf8 algo_benchmark/run_comparison.py --m 30 --runs 5 --seed-start 101 --radius-m 1200 --output results/custom
```

## Neural-model source and reproduction scope

[`legacy_model/`](legacy_model/README.md) preserves the original `run.py`, `train.py`, `options.py`, `nets/`, `problems/`, and `utils/`. Its README describes the input interface and historical training-data requirements. Pretrained weights and the original training datasets are not included.

The six-method EH-MCLP runner does not perform neural-model inference. The archived neural source uses a different coverage/distance objective and has not been converted to the three-component EH-MCLP objective. The CultiNet locations in `reference/user_M30_location_sets.json` were supplied directly as point sequences for tracing the location figures; they are not newly generated predictions. The weight experiments concern objective preferences rather than neural architecture ablations.

This release packages existing source, data, and results. Training, experiments, and tests were not rerun during release preparation.

## Updating the GitHub repository

From a clone of this repository:

```bash
git add .
git commit -m "Update CultiNet"
git push origin main
```

Generated outputs and model weights are excluded by `.gitignore`. The repository's existing [MIT license](LICENSE) is preserved; data permissions and attribution requirements remain subject to their original sources.
