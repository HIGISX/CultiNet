"""Portable fully Euclidean EH-MCLP benchmark with frozen experiment parameters.

Examples (from project root):
  python algo_benchmark/run_comparison.py
  python algo_benchmark/run_comparison.py --m 30 50 70 --service-min 15 --runs 10
  python algo_benchmark/run_comparison.py --radius-m 1200 --heritage-min 30
  python algo_benchmark/run_comparison.py --resume
  python algo_benchmark/run_comparison.py --dry-run
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from comparison_algorithms import PARAMETER_SOURCES, gurobi_solve, ortools_solve, run_heuristic
from comparison_data import load_dataset, resolve

HERE = Path(__file__).resolve().parent
EXACT = {"Gurobi": gurobi_solve, "OR-Tools": ortools_solve}


def validate_config(cfg):
    if cfg.get("distance_metric") != "euclidean":
        raise ValueError("The minimal package supports Euclidean distance only")
    if cfg.get("priority_distance_metric") != "euclidean":
        raise ValueError("The priority group must be recomputed from Euclidean distances")
    Ms = cfg["M_list"]
    if not Ms or any(isinstance(M, bool) or not isinstance(M, int) or M < 1 for M in Ms) or len(set(Ms)) != len(Ms):
        raise ValueError("M_list must contain unique positive integers")
    if not isinstance(cfg["n_runs"], int) or cfg["n_runs"] < 1:
        raise ValueError("n_runs must be a positive integer")
    known = set(EXACT) | set(PARAMETER_SOURCES)
    if (not cfg["algorithms"] or len(set(cfg["algorithms"])) != len(cfg["algorithms"])
            or not set(cfg["algorithms"]) <= known):
        raise ValueError(f"algorithms must be a unique nonempty subset of {sorted(known)}")
    for key in ["time_limit_s", "threads", "integer_scale", "certificate_abs_tolerance"]:
        val = cfg["exact"][key]
        if not np.isfinite(val) or val <= 0:
            raise ValueError(f"exact.{key} must be finite and positive")
    if cfg["exact"]["integer_scale"] > 10**15:
        raise ValueError("integer_scale must be <=1e15 to avoid integer overflow")
    ps = cfg["heuristics"]
    for method, names in {"PSO": ["num_particles", "max_iter"], "GA": ["iterations", "generation_size", "reproduction_size"],
                          "SA": ["max_iter"], "NSGA-II": ["population_size", "generations"]}.items():
        for name in names:
            if not isinstance(ps[method][name], int) or ps[method][name] < 1:
                raise ValueError(f"{method}.{name} must be a positive integer")
    if not 2 <= ps["GA"]["reproduction_size"] < ps["GA"]["generation_size"]:
        raise ValueError("GA requires 2 <= reproduction_size < generation_size")
    if ps["NSGA-II"]["population_size"] < 2:
        raise ValueError("NSGA-II population_size must be >=2")
    for name in ["GA", "NSGA-II"]:
        if not 0 <= ps[name]["mutation_prob"] <= 1:
            raise ValueError(f"{name}.mutation_prob must lie in [0,1]")
    sa = ps["SA"]
    if not 0 < sa["stopping_T"] < sa["T0"] or not 0 < sa["alpha"] < 1:
        raise ValueError("SA requires 0 < stopping_T < T0 and 0 < alpha < 1")


def reference_for(runs, tolerance):
    feasible = [r for r in runs if r.get("F") is not None]
    if not feasible:
        return dict(kind="unavailable", value=None, upper_bound=None, absolute_certificate_gap=None)
    best = max(r["F"] for r in feasible)
    bounds = [r["upper_bound"] for r in runs if r.get("upper_bound") is not None]
    upper = min(bounds) if bounds else None
    if upper is not None and upper < best - tolerance:
        raise RuntimeError(f"A feasible F={best:.12g} exceeds a solver upper bound {upper:.12g}. Check model consistency.")
    certificate_gap = max(0.0, upper - best) if upper is not None else None
    return dict(kind="certified_optimum" if upper is not None and certificate_gap <= tolerance else "best_known",
                value=best, upper_bound=upper, absolute_certificate_gap=certificate_gap)


def summarize(runs, config, metadata):
    rows, references = [], {}
    for key in metadata:
        references[key] = {}
        for M in config["M_list"]:
            group = [r for r in runs if r["dataset"] == key and r["M"] == M]
            ref = reference_for(group, config["exact"]["certificate_abs_tolerance"])
            references[key][str(M)] = ref
            for method in config["algorithms"]:
                found = [r for r in group if r["method"] == method]
                valid = [r for r in found if r.get("F") is not None]
                vals = [r["F"] for r in valid]
                avg = float(np.mean(vals)) if vals else None
                delta = max(0, ref["value"] - avg) if avg is not None else None
                gap = ((delta / ref["value"] * 100 if ref["value"] else 0.0)
                       if avg is not None and ref["kind"] == "certified_optimum" else None)
                best_known_gap = ((delta / ref["value"] * 100 if ref["value"] else 0.0)
                                  if avg is not None else None)
                expected = 1 if method in EXACT else config["n_runs"]
                rows.append(dict(dataset=key, M=M, method=method, avg_obj=avg,
                                 time_s=float(np.mean([r["runtime_s"] for r in valid])) if valid else None,
                                 gap_pct=gap, best_known_gap_pct=best_known_gap,
                                 std_obj=float(np.std(vals, ddof=1)) if len(vals) > 1 else (0.0 if vals else None),
                                 best_obj=max(vals) if vals else None, n_runs=len(valid), expected_runs=expected,
                                 complete=len(valid) == expected, status=", ".join(sorted(set(r["status"] for r in found))),
                                 reference_kind=ref["kind"], reference_obj=ref["value"],
                                 upper_bound=ref["upper_bound"], certificate_gap_abs=ref["absolute_certificate_gap"]))
    return rows, references


def save_json(path, content):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def render_tables(output, rows, config, metadata, references):
    output = Path(output)
    pd.DataFrame(rows).to_csv(output / "summary_long.csv", index=False, encoding="utf-8-sig")
    docs, tables_html, texes = [], [], []
    for key, meta in metadata.items():
        title = "合成数据集" if key == "synthetic" else "真实数据集（西安原始 1 km 需求网格）"
        D, N, H = meta["n_demand"], meta["n_candidate"], meta["n_heritage"]
        info = f"D={D:,}, N={N:,}, H={H}; T={meta['service_min']:g} min, L={meta['heritage_min']:g} min"
        if meta.get("distance_metric") == "euclidean":
            title += "（欧氏距离）"
            info += f"; 欧氏覆盖半径={meta['equivalent_walk_distance_m']:g} m"
        wide, body, mdlines, texrows = [], [], [], []
        headings = ["Method"] + [f"M={M} {s}" for M in config["M_list"] for s in ["Avg-Obj", "Time (s)", "Gap (%)"]]
        mdlines += ["| " + " | ".join(headings) + " |", "|" + "---|" * len(headings)]
        for method in config["algorithms"]:
            values, cells, formatted, tx = [method], [f"<th>{html.escape(method)}</th>"], [method], [method]
            for M in config["M_list"]:
                row = next(r for r in rows if (r["dataset"], r["M"], r["method"]) == (key, M, method))
                block = [r for r in rows if r["dataset"] == key and r["M"] == M and r["avg_obj"] is not None]
                for metric, digits, suffix in [("avg_obj", 5, ""), ("time_s", 3, ""), ("gap_pct", 2, "%")]:
                    val = row[metric] if row["complete"] else None
                    values.append(val)
                    rendered = f"{val:.{digits}f}{suffix}" if val is not None else "n.a."
                    bests = [r[metric] for r in block if r[metric] is not None and r["complete"]]
                    optimum = (max(bests) if metric == "avg_obj" else min(bests)) if bests else None
                    is_best = val is not None and optimum is not None and abs(val - optimum) < 0.5 * 10**(-digits)
                    formatted.append(f"**{rendered}**" if is_best and metric != "time_s" else rendered)
                    cells.append(f"<td{' class=best' if is_best else ''}>{rendered}</td>")
                    escaped = rendered.replace("%", r"\%")
                    tx.append(r"\textbf{" + escaped + "}" if is_best and metric != "time_s" else escaped)
            wide.append(values)
            body.append("<tr>" + "".join(cells) + "</tr>")
            mdlines.append("| " + " | ".join(formatted) + " |")
            texrows.append(" & ".join(tx) + r" \\")
        pd.DataFrame(wide, columns=headings).to_csv(output / f"table_{key}.csv", index=False, encoding="utf-8-sig")
        refs = "; ".join(f"M={M}: {references[key][str(M)]['kind']} F={references[key][str(M)]['value']}"
                         for M in config["M_list"])
        note = (f"随机算法各 {config['n_runs']} 次独立运行；精确求解器各 1 次。Time 为单次平均求解时间（秒，含建模或初始化，"
                "不含公共数据读取与模块导入）。Gap=(F*−Avg-Obj)/F*×100%；只有数值容差内已获最优证书才显示 Gap。"
                "n.a. 表示缺少已证明最优的基准或运行不完整。")
        if config.get("parameter_profile") == "tuned_native_parameters":
            note += (f" 本表使用预实验后冻结的参数，验证种子为 {config['seed_start']}–{config['seed_start'] + config['n_runs'] - 1}；"
                     "预实验结果不纳入均值。算法仍使用原搜索机制。")
        if meta.get("distance_metric") == "euclidean":
            note += (" 本轮更换距离口径，直接沿用路网实验已冻结的算法参数；没有针对欧氏结果再次选参。"
                     + ("优先人群固定为原路网基准组。" if meta.get("priority_group_unchanged") else "优先人群根据欧氏文化机会重新计算。"))
        docs += [f"## {title}", "", info, "", *mdlines, "", note, "", refs, ""]
        group_heads = "".join(f'<th colspan="3">D={D}, N={N}, M={M}</th>' for M in config["M_list"])
        sub_heads = "".join("<th>Avg-Obj</th><th>Time (s)</th><th>Gap</th>" for _ in config["M_list"])
        tables_html.append(f"<section><h2>{title}</h2><p>{html.escape(info)}</p><table>"
                           f'<thead><tr><th rowspan="2">Method</th>{group_heads}</tr><tr>{sub_heads}</tr></thead>'
                           + "<tbody>" + "".join(body) + f"</tbody></table><p class=note>{note}</p></section>")
        texes += [r"\begin{table}[htbp]", r"\centering", f"\\caption{{EH-MCLP: {key}, D={D}, N={N}, H={H}, T={meta['service_min']:g} min}}",
                  r"\begin{tabular}{l" + "rrr" * len(config["M_list"]) + "}", r"\toprule",
                  "Method & " + " & ".join(r"\multicolumn{3}{c}{" + f"$M={M}$" + "}" for M in config["M_list"]) + r" \\",
                  " & " + " & ".join(["Avg-Obj & Time (s) & Gap"] * len(config["M_list"])) + r" \\", r"\midrule",
                  *texrows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    l1, l2, l3 = config["weights"]
    formula = f"F={l1:g}×覆盖人口比例+{l2:g}×优先人群覆盖人口比例+{l3:g}×关联遗产权重比例。"
    intro = ("# EH-MCLP 两套数据的算法对比\n\n" + formula +
             "三个分项分别归一化后加权，F∈[0,1]，不按每次实验的最好/最差结果做 min-max 归一化。\n\n")
    (output / "两张对比表.md").write_text(intro + "\n".join(docs), encoding="utf-8")
    (output / "tables.tex").write_text("% Requires \\usepackage{booktabs}\n" + "\n".join(texes), encoding="utf-8")
    (output / "两张对比表.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>EH-MCLP 两张对比表</title>'
        '<style>body{font-family:"Times New Roman","Microsoft YaHei",serif;max-width:1500px;margin:36px auto;padding:0 24px;color:#172030}'
        'h1{font-size:25px}h2{font-size:20px}section{margin:36px 0 56px}table{width:100%;border-collapse:collapse;border-top:2px solid #172030;'
        'border-bottom:2px solid #172030;text-align:right}thead{border-bottom:1px solid #172030}th,td{padding:13px 16px;white-space:nowrap}'
        'thead th{text-align:center;font-weight:normal}tbody th{text-align:left;font-weight:normal}.best{font-weight:bold}.note{font-size:13px;line-height:1.8}'
        f'</style><h1>EH-MCLP 算法对比</h1><p>F = {l1:g}F1 + {l2:g}F2 + {l3:g}F3，三个分项均为归一化比例。</p>'
        + "".join(tables_html) + "</html>", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "compare_config_euclidean.json")
    parser.add_argument("--m", nargs="+", type=int)
    radius = parser.add_mutually_exclusive_group()
    radius.add_argument("--service-min", type=float)
    radius.add_argument("--radius-m", type=float)
    parser.add_argument("--heritage-min", type=float)
    parser.add_argument("--distance-metric", choices=["euclidean"])
    parser.add_argument("--priority-distance-metric", choices=["euclidean"])
    parser.add_argument("--runs", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--datasets", nargs="+", choices=["synthetic", "real"], default=["synthetic", "real"])
    parser.add_argument("--methods", nargs="+", choices=["Gurobi", "OR-Tools", "PSO", "GA", "SA", "NSGA-II"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    base = config_path.parent
    cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))
    for name in ["distance_metric", "priority_distance_metric"]:
        if getattr(args, name) is not None:
            cfg[name] = getattr(args, name)
    for arg, key in [(args.m, "M_list"), (args.runs, "n_runs"), (args.seed_start, "seed_start"), (args.methods, "algorithms")]:
        if arg is not None:
            cfg[key] = arg
    if args.service_min is not None:
        cfg.update(service_min=args.service_min, service_radius_m=None)
    if args.radius_m is not None:
        cfg["service_radius_m"] = args.radius_m
    if args.time_limit is not None:
        cfg["exact"]["time_limit_s"] = args.time_limit
    if args.heritage_min is not None:
        for key in args.datasets:
            cfg[key]["heritage_min"] = args.heritage_min
    validate_config(cfg)
    output = args.output.resolve() if args.output else resolve(cfg["output_dir"], base)
    datasets, metadata = {}, {}
    for key in args.datasets:
        datasets[key], metadata[key] = load_dataset(key, cfg, base, HERE / "cache")
        print(datasets[key].describe(), flush=True)
    if args.dry_run:
        print(json.dumps({"config": cfg, "datasets": metadata}, ensure_ascii=False, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    source_files = [HERE / f for f in ["run_comparison.py", "comparison_algorithms.py", "comparison_data.py", "eh_problem.py"]]
    source_files += [HERE.parent / "Algorithm" / f for f in ["GA.py", "PSO_distance.py", "SA_distence.py", "INSGA.py"]]
    code_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    identity = {"config": cfg, "metadata": metadata, "code_sha256": code_hashes}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    statefile = output / "checkpoint.json"
    runs = []
    if statefile.exists():
        if not args.resume:
            raise FileExistsError(f"Results already exist: {output}. Use --resume or a new --output.")
        old = json.loads(statefile.read_text(encoding="utf-8"))
        if old["signature"] != signature:
            raise ValueError("Cannot resume: configuration, source data or algorithm code changed; use a new --output")
        runs = old["runs"]
    save_json(output / "effective_config.json", cfg)
    save_json(output / "dataset_metadata.json", metadata)
    versions = {}
    for package in ["numpy", "pandas", "matplotlib", "gurobipy", "ortools"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    protocol = dict(identity, signature=signature, parameter_sources=PARAMETER_SOURCES,
                    python=sys.version, platform=platform.platform(), cpu=platform.processor(),
                    versions=versions,
                    normalization="separate population/priority/heritage denominators; no empirical min-max",
                    sa_objective_scale=1, exact_runs_per_case=1, stochastic_runs_per_case=cfg["n_runs"],
                    runtime_scope="wall clock including model/population initialization and solve; excludes shared data I/O and imports",
                    time_created_utc=datetime.now(timezone.utc).isoformat())
    if cfg.get("parameter_profile") == "tuned_native_parameters":
        protocol.update(parameter_profile=cfg["parameter_profile"],
                        actual_parameter_source=f"{config_path} frozen after the registered pilot; original notebook sources identify algorithm implementations only",
                        tuning_plan_signature=cfg["tuning_plan_signature"],
                        validation_seeds=list(range(cfg["seed_start"], cfg["seed_start"] + cfg["n_runs"])))
    save_json(output / "protocol.json", protocol)
    for key, problem in datasets.items():
        for M in cfg["M_list"]:
            problem.K = M
            problem.validate_K()
            print(f"\n[{key}] M={M}", flush=True)
            for method in cfg["algorithms"]:
                seeds = [None] if method in EXACT else list(range(cfg["seed_start"], cfg["seed_start"] + cfg["n_runs"]))
                for seed in seeds:
                    if any((r["dataset"], r["M"], r["method"], r["seed"]) == (key, M, method, seed) for r in runs):
                        continue
                    try:
                        if method in EXACT:
                            result = EXACT[method](problem, cfg["exact"], cfg["seed_start"])
                        else:
                            result = run_heuristic(method, problem, seed, cfg["heuristics"][method])
                    except Exception as exc:
                        missing = isinstance(exc, ImportError)
                        solver_failure = method in EXACT and type(exc).__name__ == "GurobiError"
                        if not missing and not solver_failure:
                            raise
                        result = dict(method=method, seed=seed,
                                      status="DEPENDENCY_UNAVAILABLE" if missing else "SOLVER_ERROR",
                                      error=str(exc), F=None, runtime_s=None, upper_bound=None)
                    result.update(dataset=key, M=M)
                    runs.append(result)
                    save_json(statefile, dict(signature=signature, runs=runs))
                    print(f"  {method:8s} seed={str(seed):4s} F={result['F']} time={result['runtime_s']} status={result['status']}", flush=True)
            rows, refs = summarize(runs, cfg, metadata)
            render_tables(output, rows, cfg, metadata, refs)
            save_json(output / "summary.json", dict(config=cfg, datasets=metadata, rows=rows, references=refs,
                                                    complete=all(r["complete"] for r in rows)))
    for run in runs:
        reference = refs[run["dataset"]][str(run["M"])]
        run["reference_kind"] = reference["kind"]
        run["reference_obj"] = reference["value"]
        run["gap_pct"] = (max(0.0, reference["value"] - run["F"]) / reference["value"] * 100
                          if reference["value"] else 0.0) if (reference["kind"] == "certified_optimum"
                                                                          and run["F"] is not None) else None
    flat = [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in run.items()} for run in runs]
    pd.DataFrame(flat).to_csv(output / "runs.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved two tables and per-run results: {output}", flush=True)


if __name__ == "__main__":
    main()
