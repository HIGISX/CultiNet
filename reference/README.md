# Reference Results

These files contain benchmark summaries, individual run records, and location sequences.

- `table_synthetic.csv` and `table_real.csv`: summary tables for six baseline methods using Euclidean distances and `M = 30, 50, 70`.
- `benchmark_runs.json`: 492 individual run records, including selected candidates, for the earlier Euclidean benchmark. Its synthetic records use the previous 1,000-demand-point dataset.
- `weight_sensitivity_results.csv` and `weight_sensitivity_solutions.json`: metrics and selected candidates for eight weight-sensitivity scenarios.
- `weight_ablation_results.csv` and `weight_ablation_solutions.json`: metrics and selected candidates for five objective-ablation scenarios.
- `user_M30_location_sets.json`: location sequences used for the Xi'an `M = 30` comparison figures. These candidate numbers are **one-based**.

In individual solver records, `selection` contains **zero-based** indices into the candidate array, following the ascending `candidate_id` order used by the data loader. Use `selected_ids`, where available, to match selections to candidate IDs in the data tables. Configuration, runtime, and optimality-certificate fields are preserved as recorded.
