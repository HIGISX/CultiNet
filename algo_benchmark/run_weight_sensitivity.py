"""Minimal, pre-specified objective-weight sensitivity on the Euclidean model.

Run from the project root:
  python -u -X utf8 algo_benchmark/run_weight_sensitivity.py

Only objective weights change. The instance, priority population, distance
thresholds, cardinality and feasibility constraints stay fixed across cases.
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

import gurobipy as gp
import numpy as np
import pandas as pd

from comparison_algorithms import gurobi_solve
from comparison_data import load_dataset

HERE = Path(__file__).resolve().parent
CASES = [
    ("B", "基准", (0.35, 0.40, 0.25)),
    ("E", "等权", (1 / 3, 1 / 3, 1 / 3)),
    ("S1", "效率↑／公平↓", (0.45, 0.30, 0.25)),
    ("S2", "公平↑／效率↓", (0.25, 0.50, 0.25)),
    ("S3", "效率↑／遗产↓", (0.45, 0.40, 0.15)),
    ("S4", "遗产↑／效率↓", (0.25, 0.40, 0.35)),
    ("S5", "公平↑／遗产↓", (0.35, 0.50, 0.15)),
    ("S6", "遗产↑／公平↓", (0.35, 0.30, 0.35)),
]


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "compare_config_euclidean.json")
    parser.add_argument("--dataset", choices=["real", "synthetic"], default="real")
    parser.add_argument("--m", type=int, default=30)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", type=Path,
                        default=HERE / "results/weight_sensitivity_euclidean_real_M30")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if (config.get("distance_metric") != "euclidean"
            or config.get("priority_distance_metric") != "euclidean"):
        raise ValueError("This experiment requires the fully Euclidean baseline")
    config["M_list"] = [args.m]
    config["weights"] = list(CASES[0][2])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    problem, metadata = load_dataset(args.dataset, config, config_path.parent, HERE / "cache")
    print(problem.describe(), flush=True)
    matrix_hashes = {name: hashlib.sha256(getattr(problem, name).tobytes()).hexdigest()
                     for name in ["A", "B", "U", "p", "w", "eligible"]}
    source_files = [Path(__file__).resolve()] + [HERE / name for name in
        ["comparison_algorithms.py", "comparison_data.py", "eh_problem.py"]]
    protocol = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        design="baseline, equal weights, and six pairwise transfers of 0.10",
        dataset=args.dataset, M=args.m, cases=[dict(case=k, label=n, weights=w) for k, n, w in CASES],
        primary_metrics={"F1": "population coverage / total population",
                         "F2": "priority-population coverage / fixed priority population",
                         "F3": "linked heritage weight / total heritage weight"},
        exact=config["exact"], seed=args.seed, mip_gap=0, mip_gap_abs=0,
        gurobi_version=gp.gurobi.version(), python=sys.version, platform=platform.platform(),
        baseline_config_path=str(config_path), effective_config=config, dataset_metadata=metadata,
        fixed_arrays_sha256=matrix_hashes,
        code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        interpretation="Weights encode preferences; compare the three components across cases. "
                       "Scenario-specific weighted objective values are not a common performance scale.",
    )
    save_json(output / "protocol.json", protocol)
    rows, solutions = [], []
    for key, label, weights in CASES:
        varied = replace(problem, lam=weights)
        if any(hashlib.sha256(getattr(varied, name).tobytes()).hexdigest() != value
               for name, value in matrix_hashes.items()):
            raise RuntimeError("A supposedly fixed input changed")
        result = gurobi_solve(varied, config["exact"], seed=args.seed)
        if result["F"] is None:
            raise RuntimeError(f"No feasible solution: {key}")
        selection = result["selection"]
        covered = problem.A[:, selection].any(axis=1)
        linked = problem.B[:, selection].any(axis=1)
        quantities = [float(problem.p[covered].sum()),
                      float(problem.p[covered & problem.U].sum()),
                      float(problem.w[linked].sum())]
        direct = np.array(quantities) / [problem.P, problem.PU, problem.W]
        if (not varied.is_feasible(selection)
                or not np.allclose(direct, [result["F1"], result["F2"], result["F3"]], atol=1e-12, rtol=0)
                or abs(np.dot(weights, direct) - result["F"]) > 1e-12):
            raise RuntimeError(f"Independent feasibility/objective check failed: {key}")
        baseline_selection = set(solutions[0]["selection"]) if solutions else set(selection)
        row = dict(case=key, scenario=label, lambda1=weights[0], lambda2=weights[1], lambda3=weights[2],
                   F1=result["F1"], F2=result["F2"], F3=result["F3"],
                   covered_population=quantities[0], covered_priority_population=quantities[1],
                   linked_heritage_weight=quantities[2], linked_heritage_count=int(linked.sum()),
                   replaced_sites=len(set(selection) - baseline_selection),
                   **{k: result[k] for k in ["F", "status", "runtime_s", "upper_bound", "solver_gap"]})
        rows.append(row)
        solutions.append(dict(case=key, weights=weights, **result))
        save_json(output / "solutions.json", solutions)
        pd.DataFrame(rows).to_csv(output / "results.csv", index=False, encoding="utf-8-sig")
        print(f"{key} {weights}: {result['status']} "
              f"F1={direct[0]:.8f}, F2={direct[1]:.8f}, F3={direct[2]:.8f}", flush=True)

    components = np.array([[r["F1"], r["F2"], r["F3"]] for r in rows])
    checks = {"direct_component_and_feasibility_checks_passed": True,
              "fixed_inputs_verified": True,
              "all_optimal": all(r["status"] == "OPTIMAL" for r in rows)}
    for index, row in enumerate(rows):
        own_value = float(np.dot(CASES[index][2], components[index]))
        best_other = float(np.max(components @ CASES[index][2]))
        if row["status"] == "OPTIMAL" and best_other > own_value + 1e-9:
            raise RuntimeError(f"Another feasible selection beats the certified optimum: {row['case']}")
    checks["cross_weight_feasible_solution_check_passed"] = True
    historical = config_path.parent / config["output_dir"] / "checkpoint.json"
    if historical.exists():
        state = json.loads(historical.read_text(encoding="utf-8"))
        old = next((r for r in state["runs"] if r["dataset"] == args.dataset
                    and r["M"] == args.m and r["method"] == "Gurobi"), None)
        if old:
            same = abs(old["F"] - rows[0]["F"]) < 1e-9
            checks.update(historical_baseline_path=str(historical),
                          historical_baseline_F=old["F"], baseline_matches_historical_F=same)
            if not same:
                raise RuntimeError("The baseline does not reproduce the existing Euclidean experiment")
    save_json(output / "verification.json", checks)
    table = ["| 情景 | 权重（效率，公平，遗产） | 总人口覆盖率 F1（%） | 优先人群覆盖率 F2（%） | 遗产加权关联率 F3（%） |",
             "|---|---|---:|---:|---:|"]
    for row in rows:
        weight_text = "1/3，1/3，1/3" if row["case"] == "E" else "，".join(
            f"{row[k]:.2f}" for k in ["lambda1", "lambda2", "lambda3"])
        table.append(f"| {row['scenario']} | {weight_text} | {row['F1'] * 100:.2f} | "
                     f"{row['F2'] * 100:.2f} | {row['F3'] * 100:.2f} |")
    base, equal = rows[:2]
    delta = {name: 100 * (base[name] - equal[name]) for name in ["F1", "F2", "F3"]}
    notes = [
        "表：目标函数权重敏感性分析（欧氏距离，西安，M=30）" if args.dataset == "real" and args.m == 30
        else f"表：目标函数权重敏感性分析（欧氏距离，{args.dataset}，M={args.m}）",
        "",
        "设置：固定同一数据实例、优先组、设施数和约束，仅改变目标权重。基准与等权各一组，"
        "再对三对目标分别进行正、反向 0.10 权重转移，共 8 次 Gurobi 求解。",
        "",
        *table,
        "",
        "三个指标分母分别为总人口、固定优先人群人口与遗产权重总量；均越高越好。"
        "优先人群根据欧氏距离计算的优化前文化机会确定，约占总人口 30%。",
        f"覆盖半径为 {metadata['equivalent_walk_distance_m']:.0f} 米，遗产关联半径为 "
        f"{metadata['heritage_min'] * metadata['walk_speed_m_per_min']:.0f} 米；"
        f"保留原相容性和场地约束。Gurobi Threads={config['exact']['threads']}，"
        f"Seed={args.seed}，MIPGap=0，MIPGapAbs=0。",
        "全部达到 OPTIMAL。" if checks["all_optimal"] else "部分情景未达到 OPTIMAL，详见 results.csv。",
        "",
        f"基准相对等权的变化（百分点）：F1 {delta['F1']:+.4f}，"
        f"F2 {delta['F2']:+.4f}，F3 {delta['F3']:+.4f}。",
        "",
        "分析重点：权重变化如何推动总量覆盖、公平优先与遗产协同之间的可解释配置，"
        "并说明（0.35，0.40，0.25）所表达的适度公平优先取向。"
        "实验用于支持权重的取舍依据，不将某一组权重称为数据识别出的唯一最优权重。",
        "",
        "复现：python -u -X utf8 algo_benchmark/run_weight_sensitivity.py --output <新的结果目录>",
    ]
    (output / "table_and_design.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
