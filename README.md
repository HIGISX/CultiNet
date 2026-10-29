# CultiNet

Code and benchmark data for CultiNet, a cultural-facility location optimization project. The repository includes model source, synthetic and Xi'an datasets, algorithm comparisons, and objective-weight sensitivity and ablation experiments.

## Repository layout

```text
CultiNet/
├── algo_benchmark/          # Experiment runners, configuration and shared objective
├── Algorithm/              # GA, PSO, SA and NSGA-II implementations
├── data/                   # Minimal inputs for the synthetic and Xi'an cases
├── model/                  # Neural-model architecture and training source
├── reference/              # Experiment results and selected locations
├── requirements.txt        # Dependencies for the four stochastic baselines
├── requirements-solvers.txt
└── requirements-model.txt
```

The experiments use the supplied data through relative paths. See [`data/README.md`](data/README.md) for the input fields, coordinates, and preprocessing rules.

## Installation

Use **Python 3.12**. Clone the repository and create a virtual environment:

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

For the neural-model source, install its dependencies with:

```bash
python -m pip install -r requirements-model.txt
```

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

[`reference/`](reference/README.md) contains summary tables and individual selections. Runtime depends on the hardware, software versions, and solver environment.

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
| Synthetic | 10,000 | 2,000 | 600 | 1,125 m |
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

## Model source

[`model/`](model/README.md) contains `run.py`, `train.py`, `options.py`, and the `nets/`, `problems/`, and `utils/` modules. See its README for the model input format and data paths. Architecture and decoding are implemented in `nets/CacheFormer.py` and `nets/FacilitiesRepresentationEncoder.py`; training options are defined in `options.py`.

## Updating the GitHub repository

From a clone of this repository:

```bash
git add .
git commit -m "Update CultiNet"
git push origin main
```

## License

See [LICENSE](LICENSE).
