# Minimal Euclidean Data

This directory contains the coordinates, population, IDs, site constraints, and classifications required for the current fully Euclidean experiments. At runtime, the loader reconstructs the population coverage matrix A, heritage linkage matrix B, and equity-priority group U directly from these inputs. It requires no original network OD tables, GIS software, external drive paths, or matrix cache.

| Dataset | Demand points D | Candidates N | Heritage sites H | Total population | Default coverage radius | Heritage linkage radius |
|---|---:|---:|---:|---:|---:|---:|
| `synthetic/` | 1,000 | 200 | 60 | 1,000,000 | 1,125 m | 750 m |
| `xian/` | 7,685 | 587 | 297 | 6,437,381 | 1,200 m | 2,400 m |

Both datasets use a default coverage time of 15 minutes, converted to distance at 75 m/min for Synthetic and 80 m/min for Xi'an. Xi'an coordinates use the projected metric CRS EPSG:4545. Synthetic coordinates were converted from their original normalized values by multiplying by 15,000 m. All distances are calculated within each dataset's shared metric plane.

## CSV Fields

- `demand_points.csv`: `demand_id, x_m, y_m, population`.
- `candidate_points.csv`: `candidate_id, x_m, y_m, site_id, eligible`, plus `program` for Synthetic or `poi_class` for Xi'an.
- `heritage_points.csv`: `heritage_id, x_m, y_m, heritage_weight, heritage_class`.
- All three Xi'an point tables also retain `source_id` to trace records to the original demand `GRID_ID` or location `site_id`. The D/C/H IDs used for optimization match the original benchmark crosswalk.
- Synthetic `linkage_constraints.csv`: every H-C ID pair, with `program_compatible` and `path_crosses_river`. These preserve the original non-distance constraints; neither field represents Euclidean distance.

Xi'an population data retain the originally supplied 1 km demand grid, rather than the historical derived 300 m demand surface. All Xi'an candidates are eligible. Synthetic data retain the original `eligible` flags and shared-site IDs.

## Model Definitions

`A[i,j] = (euclidean_distance(i,j) / walking_speed <= service_min)`.

For Synthetic, B requires distance within the linkage threshold, program compatibility, and an original river-crossing flag of 0. For Xi'an, B retains the original thematic rule: ancient sites, ancient tombs, and grotto temples and stone carvings require a candidate in an interpretive venue category. Heritage weights are 4 for national-level protection, 3 for provincial-level protection, 2 for municipal-level protection, and 1 for county-level protection. All Synthetic heritage weights are 1.

The opportunity index is `O_i = sum_h w_h * exp(-alpha * euclidean_walking_minutes(i,h))`. Synthetic uses `alpha=0.05`; Xi'an uses `alpha=ln(2)/median(all positive Euclidean D-H walking times in minutes)`. Demand points are stably sorted by increasing O, and U is the prefix whose cumulative population is closest to 30% of the total. U remains fixed across algorithms, facility counts, and objective-weight scenarios.

The main experiment uses `F = 0.35 F1 + 0.40 F2 + 0.25 F3`. The three components are normalized by total population, population in U, and total heritage weight, respectively. Exactly M eligible candidates are selected, with at most one candidate per `site_id` by default. Weight sensitivity and ablation experiments change only the objective weights.

## Provenance and Reproduction Status

`provenance.json` preserves historical SHA-256 hashes of the original point tables, crosswalks, CRS file, and Synthetic compatibility file, together with the A/B nonzero counts and U hash recorded in the original formal Euclidean experiment. These are records from earlier experiments and do not establish that this release package has been revalidated.

Packaging extracted the necessary columns, aligned IDs, and replaced absolute paths with relative paths. At the user's request, no new experiments, tests, or release-package validation were run. At runtime, `dataset_metadata.json` records SHA-256 hashes of the current input files and A/B/U arrays for comparison when needed.

The historical real-world network heritage OD tables were found to be inconsistent with the point coordinates or IDs. This package therefore provides only the fully Euclidean workflow. It cannot reconstruct the original road network or reproduce the historical network results. Data provenance and redistribution permissions remain subject to the original sources; this directory makes no additional licensing grant.

Run from the repository root:

```powershell
# Run the four metaheuristics; core dependencies are numpy, pandas, and matplotlib
python -u -X utf8 algo_benchmark/run_comparison.py --methods PSO GA SA NSGA-II --output results/heuristics

# Run all six baselines; also install Gurobi / OR-Tools
# Gurobi requires a license that supports the instance size
python -u -X utf8 algo_benchmark/run_comparison.py --output results/benchmark

# Weight sensitivity and weight ablation with conservative handling of tied optima
# Both require Gurobi
python -u -X utf8 algo_benchmark/run_weight_sensitivity.py --output results/weight_sensitivity
python -u -X utf8 algo_benchmark/run_weight_ablation.py --output results/weight_ablation
```

Use a fresh output directory for each experiment. To resume an interrupted main benchmark in the same directory, add `--resume`; it rejects results whose configuration, source data, or code have changed. Output directories for the weight experiments must not already exist.
