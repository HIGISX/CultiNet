"""Five pre-specified Gurobi weight ablations on the real Xi'an instance.

Run from the project root with Python 3.12. Only objective weights change.
For an ablation, a second solve maximises the omitted component subject to
preserving the primary optimum. This avoids overstating damage due to ties.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from comparison_algorithms import gurobi_solve
from comparison_data import load_dataset

HERE = Path(__file__).resolve().parent
SCALE = 1e6
PRIMARY_ALLOWANCE = 1e-11


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_hashes(problem):
    result = {k: hashlib.sha256(getattr(problem, k).tobytes()).hexdigest()
              for k in ("A", "B", "p", "w", "U", "eligible")}
    result["constraints_and_ids"] = hashlib.sha256(json.dumps(
        [problem.K, problem.site_capacity, problem.site_id, problem.cand_id],
        ensure_ascii=False).encode("utf-8")).hexdigest()
    return result


def components(problem, selection):
    covered = problem.A[:, selection].any(axis=1)
    linked = problem.B[:, selection].any(axis=1)
    quantities = np.array([problem.p[covered].sum(), problem.p[covered & problem.U].sum(),
                           problem.w[linked].sum()], dtype=float)
    return quantities / [problem.P, problem.PU, problem.W], quantities, int(linked.sum())


def best_omitted_component(problem, primary_result, omitted, params, seed, log_path):
    """Conservative tie resolution; no omitted component has a primary weight.

    Exact OR indicators are used because zero coefficients do not force the
    usual upper-bound-only coverage indicators to equal actual coverage.
    """
    started = time.perf_counter()
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model("weight_ablation_tie_resolution", env=env) as model:
            model.Params.TimeLimit = params["time_limit_s"]
            model.Params.Threads = params["threads"]
            model.Params.Seed = seed
            model.Params.MIPGap = 0
            model.Params.MIPGapAbs = 0
            model.Params.FeasibilityTol = 1e-9
            model.Params.IntFeasTol = 1e-9
            model.Params.LogFile = str(log_path)
            model.Params.LogToConsole = 0
            model.Params.OutputFlag = 1
            x = model.addVars(problem.nJ, vtype=GRB.BINARY, name="x")
            y = model.addVars(problem.nI, vtype=GRB.BINARY, name="covered")
            z = model.addVars(problem.nH, vtype=GRB.BINARY, name="linked")
            model.addConstr(x.sum() == problem.K)
            for j in range(problem.nJ):
                x[j].UB = int(problem.eligible[j])
                x[j].Start = int(j in primary_result["selection"])
            if problem.site_capacity is not None:
                for indices in problem.groups.values():
                    model.addConstr(gp.quicksum(x[j] for j in indices) <= problem.site_capacity)
            for matrix, indicators in ((problem.A, y), (problem.B, z)):
                for i, row in enumerate(matrix):
                    indices = np.flatnonzero(row).tolist()
                    if indices:
                        model.addGenConstrOr(indicators[i], [x[j] for j in indices])
                    else:
                        indicators[i].UB = 0
            expressions = [gp.quicksum(float(problem.p[i] / problem.P * SCALE) * y[i]
                                       for i in range(problem.nI)),
                           gp.quicksum(float(problem.p[i] / problem.PU * SCALE) * y[i]
                                       for i in np.flatnonzero(problem.U).tolist()),
                           gp.quicksum(float(problem.w[h] / problem.W * SCALE) * z[h]
                                       for h in range(problem.nH))]
            primary = gp.quicksum(w * expression for w, expression in zip(problem.lam, expressions))
            model.addConstr(primary >= (primary_result["F"] - PRIMARY_ALLOWANCE) * SCALE)
            model.setObjective(expressions[omitted], GRB.MAXIMIZE)
            model.optimize()
            if model.Status != GRB.OPTIMAL:
                raise RuntimeError(f"Conservative tie resolution not optimal: {model.Status}")
            selection = [j for j in range(problem.nJ) if x[j].X > .5]
            metrics, _, _ = components(problem, selection)
            if (not problem.is_feasible(selection)
                    or np.dot(problem.lam, metrics) < primary_result["F"] - 1e-10
                    or abs(metrics[omitted] - model.ObjVal / SCALE) > 1e-10):
                raise RuntimeError("Secondary solve failed the independent coverage check")
            return selection, dict(status="OPTIMAL", runtime_s=time.perf_counter() - started,
                                   omitted_component=omitted + 1, optimum=float(model.ObjVal / SCALE),
                                   upper_bound=float(model.ObjBound / SCALE), gap=float(model.MIPGap))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "compare_config_euclidean.json")
    parser.add_argument("--m", type=int, default=30)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if config.get("distance_metric") != "euclidean" or config.get("priority_distance_metric") != "euclidean":
        raise ValueError("Use the established fully Euclidean comparison protocol")
    baseline_weights = np.array(config["weights"], dtype=float)
    if len(baseline_weights) != 3 or min(baseline_weights) <= 0 or not np.isclose(baseline_weights.sum(), 1):
        raise ValueError("Baseline must contain three positive normalised weights")
    config["M_list"] = [args.m]
    cases = [("baseline", "完整目标（基准）", baseline_weights.tolist(), None),
             ("equal", "等权", [1 / 3] * 3, None)]
    for omitted, label in enumerate(("去掉人口覆盖项", "去掉优先人群项", "去掉遗产关联项")):
        weights = baseline_weights.copy()
        weights[omitted] = 0
        weights /= weights.sum()
        cases.append((f"without_F{omitted + 1}", label, weights.tolist(), omitted))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    # Write the scenario design BEFORE seeing any new optimisation results.
    protocol = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), dataset="real",
                    scenario_count=5, M=args.m, seed=args.seed,
                    cases=[dict(case=k, scenario=n, weights=w, omitted_component=o)
                           for k, n, w, o in cases],
                    tie_rule="Maximise the omitted component among primary-optimal solutions; "
                             "primary absolute allowance 1e-11; independently checked within 1e-10.",
                    fixed="data, priority group, eligibility, cardinality, site constraints, distances, thresholds",
                    interpretation="Preference ablation, not identification of a unique optimal weight vector. "
                                   "F2 measures priority-group coverage, not general inequality.",
                    config_path=str(config_path), effective_config=config, gurobi_version=gp.gurobi.version(),
                    python=sys.version, platform=platform.platform(),
                    gurobi_parameters=dict(MIPGap=0, MIPGapAbs=0, Threads=config["exact"]["threads"],
                                           TimeLimit=config["exact"]["time_limit_s"], Seed=args.seed),
                    code_sha256={str(HERE / name): digest(HERE / name) for name in
                                 ("run_weight_ablation.py", "comparison_algorithms.py", "comparison_data.py",
                                  "eh_problem.py")})
    write_json(output / "protocol.json", protocol)
    problem, metadata = load_dataset("real", config, config_path.parent, HERE / "cache")
    fixed = input_hashes(problem)
    protocol.update(dataset_metadata=metadata, fixed_inputs_sha256=fixed)
    write_json(output / "protocol.json", protocol)
    np.savez_compressed(output / "instance.npz", A=problem.A, B=problem.B, p=problem.p,
                        w=problem.w, U=problem.U, eligible=problem.eligible,
                        site_id=np.array(problem.site_id, dtype=str), cand_id=np.array(problem.cand_id, dtype=str),
                        K=problem.K, site_capacity=problem.site_capacity, lam=problem.lam)
    print(f"Real Xi'an: {problem.nI} demand / {problem.nJ} candidates / {problem.nH} heritage; M={problem.K}", flush=True)
    rows, solutions = [], []
    for key, label, weights, omitted in cases:
        varied = replace(problem, lam=tuple(weights))
        if input_hashes(varied) != fixed:
            raise RuntimeError("An input other than the weights changed")
        primary = gurobi_solve(varied, config["exact"], seed=args.seed)
        if primary["status"] != "OPTIMAL":
            raise RuntimeError(f"Primary solve not optimal: {key}, {primary['status']}")
        primary_metrics, _, _ = components(varied, primary["selection"])
        if (not np.allclose(primary_metrics, [primary["F1"], primary["F2"], primary["F3"]], atol=1e-12, rtol=0)
                or abs(np.dot(weights, primary_metrics) - primary["F"]) > 1e-12):
            raise RuntimeError("Primary solve failed independent evaluation")
        selection, secondary = primary["selection"], None
        if omitted is not None:
            selection, secondary = best_omitted_component(varied, primary, omitted, config["exact"],
                                                         args.seed, output / "logs" / f"{key}_tie.log")
        metrics, quantities, linked_count = components(problem, selection)
        baseline_selection = set(solutions[0]["selection"]) if solutions else set(selection)
        row = dict(case=key, scenario=label, lambda1=weights[0], lambda2=weights[1], lambda3=weights[2],
                   F1=metrics[0], F2=metrics[1], F3=metrics[2],
                   covered_population=quantities[0], covered_priority_population=quantities[1],
                   linked_heritage_weight=quantities[2], linked_heritage_count=linked_count,
                   replaced_sites=len(set(selection) - baseline_selection),
                   common_baseline_score=float(np.dot(baseline_weights, metrics)),
                   scenario_objective=float(np.dot(weights, metrics)),
                   primary_optimum=primary["F"], primary_upper_bound=primary["upper_bound"],
                   primary_gap=primary["solver_gap"], primary_status=primary["status"],
                   primary_runtime_s=primary["runtime_s"],
                   secondary_status=secondary["status"] if secondary else "NOT_NEEDED",
                   secondary_runtime_s=secondary["runtime_s"] if secondary else 0)
        rows.append(row)
        solutions.append(dict(case=key, selection=selection,
                              selected_ids=[problem.cand_id[j] for j in selection],
                              primary=primary, secondary=secondary))
        pd.DataFrame(rows).to_csv(output / "results.csv", index=False, encoding="utf-8-sig")
        write_json(output / "solutions.json", solutions)
        print(f"{key}: OPTIMAL; F1={metrics[0]*100:.4f}%, F2={metrics[1]*100:.4f}%, F3={metrics[2]*100:.4f}%", flush=True)
    metrics = np.array([[r[f"F{i}"] for i in (1, 2, 3)] for r in rows])
    for row, (_, _, weights, _) in zip(rows, cases):
        if np.max(metrics @ weights) > row["primary_upper_bound"] + 1e-9:
            raise RuntimeError("Another scenario beats a primary optimality certificate")
    checks = dict(all_five_primary_optimal=True, all_three_conservative_secondary_optimal=True,
                  fixed_inputs_verified=True, all_selections_feasible=all(problem.is_feasible(s["selection"]) for s in solutions),
                  direct_coverage_recomputation_passed=True, cross_weight_certificate_check_passed=True,
                  max_primary_preservation_error=float(max(abs(r["scenario_objective"] - r["primary_optimum"]) for r in rows)))
    historical_path = HERE / "results/weight_sensitivity_euclidean_real_M30_20261018/results.csv"
    if historical_path.exists() and args.m == 30:
        historical = pd.read_csv(historical_path).iloc[0]
        checks["baseline_matches_previous_experiment"] = bool(np.allclose(
            metrics[0], historical[["F1", "F2", "F3"]].to_numpy(float), atol=1e-10, rtol=0))
        if not checks["baseline_matches_previous_experiment"]:
            raise RuntimeError("Baseline differs from the earlier real-data experiment")
    checks["instance_sha256"] = digest(output / "instance.npz")
    write_json(output / "verification.json", checks)
    # Independent selection export for inspection and future maps.
    pd.DataFrame([dict(case=s["case"], selection_order=k + 1, candidate_index=j,
                       candidate_id=problem.cand_id[j], site_id=problem.site_id[j])
                  for s in solutions for k, j in enumerate(s["selection"])]).to_csv(
        output / "selected_sites.csv", index=False, encoding="utf-8-sig")
    table = ["| 情景 | 权重（人口、优先人群、遗产） | 人口覆盖率 F1（%） | 优先人群覆盖率 F2（%） | 遗产加权关联率 F3（%） |",
             "|---|---|---:|---:|---:|"]
    for row in rows:
        weights_text = "1/3, 1/3, 1/3" if row["case"] == "equal" else ", ".join(f"{row[f'lambda{i}']:.4f}" for i in (1, 2, 3))
        table.append(f"| {row['scenario']} | {weights_text} | {row['F1']*100:.2f} | {row['F2']*100:.2f} | {row['F3']*100:.2f} |")
    (output / "table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    english_names = ["Full objective", "Equal weights", "Without population coverage", "Without priority coverage", "Without heritage linkage"]
    tex = [r"\begin{tabular}{lrrrrrr}", r"\hline",
           r"Scenario & $\lambda_1$ & $\lambda_2$ & $\lambda_3$ & $F_1$ (\%) & $F_2$ (\%) & $F_3$ (\%) \\", r"\hline"]
    for name, row in zip(english_names, rows):
        tex.append(name + " & " + " & ".join(f"{row[f'lambda{i}']:.4f}" for i in (1, 2, 3))
                   + " & " + " & ".join(f"{row[f'F{i}']*100:.2f}" for i in (1, 2, 3)) + r" \\")
    tex += [r"\hline", r"\end{tabular}"]
    (output / "table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved to {output}", flush=True)


if __name__ == "__main__":
    main()
