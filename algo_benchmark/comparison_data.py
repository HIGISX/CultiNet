"""Build the published fully Euclidean instances from portable aligned CSVs.

Coordinates are already in metres and IDs have been aligned to the original
benchmark ordering. No network OD, GIS installation or external drive is used.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from eh_problem import EHMCLPProblem

INTERPRETIVE = {"纪念馆", "世界遗产", "国家级景点", "省级景点", "风景名胜", "旅游景点", "红色景区", "寺庙道观"}
NEEDS_INTERPRETIVE = {"古遗址", "古墓葬", "石窟寺及石刻"}


def resolve(path, base):
    path = Path(path)
    return (path if path.is_absolute() else Path(base) / path).resolve()


def priority_mask(opportunity, population, share):
    if not 0 < share <= 1:
        raise ValueError("priority_share must be in (0,1]")
    if not np.isfinite(opportunity).all():
        raise ValueError("Opportunity must be finite")
    order = np.argsort(opportunity, kind="stable")
    count = int(np.argmin(abs(np.cumsum(population[order]) - share * population.sum()))) + 1
    result = np.zeros(len(population), dtype=bool)
    result[order[:count]] = True
    return result


def euclidean_m(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if (left.ndim != 2 or right.ndim != 2 or left.shape[1] != 2 or right.shape[1] != 2
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError("Coordinates must be finite (n,2) arrays in metres")
    return np.hypot(left[:, 0, None] - right[None, :, 0],
                    left[:, 1, None] - right[None, :, 1])


def _read_points(path, id_column):
    frame = pd.read_csv(path, dtype={id_column: str, "site_id": str}, float_precision="round_trip")
    if not frame[id_column].is_unique or frame[id_column].isna().any():
        raise ValueError(f"Duplicate or missing IDs in {path.name}")
    return frame.sort_values(id_column, kind="stable").reset_index(drop=True)


def _linkage_constraints(path, heritage_ids, candidate_ids):
    frame = pd.read_csv(path, dtype={"heritage_id": str, "candidate_id": str})
    hi, ci = pd.Index(heritage_ids), pd.Index(candidate_ids)
    rr, cc = hi.get_indexer(frame.heritage_id), ci.get_indexer(frame.candidate_id)
    if ((rr < 0).any() or (cc < 0).any()
            or frame.duplicated(["heritage_id", "candidate_id"]).any()
            or len(frame) != len(hi) * len(ci)):
        raise ValueError("Synthetic linkage constraints must contain every H-C pair exactly once")
    arrays = []
    for column in ["program_compatible", "path_crosses_river"]:
        values = frame[column].to_numpy()
        if not np.isin(values, [0, 1]).all():
            raise ValueError(f"{column} must be binary")
        matrix = np.zeros((len(hi), len(ci)), dtype=bool)
        matrix[rr, cc] = values.astype(bool)
        arrays.append(matrix)
    return arrays


def load_dataset(key, config, base, cache_dir=None):
    """Keep the former loader interface; compact inputs need no matrix cache."""
    if key not in {"synthetic", "real"}:
        raise ValueError("Dataset must be synthetic or real")
    if (config.get("distance_metric") != "euclidean"
            or config.get("priority_distance_metric") != "euclidean"):
        raise ValueError("This minimal package supports the fully Euclidean protocol only")
    spec = config[key]
    data = resolve(spec["data_dir"], base)
    sources = [data / f"{label}_points.csv" for label in ["demand", "candidate", "heritage"]]
    d, c, h = [_read_points(path, column) for path, column in zip(
        sources, ["demand_id", "candidate_id", "heritage_id"])]
    xy = [frame[["x_m", "y_m"]].to_numpy(float) for frame in [d, c, h]]
    p, w = d.population.to_numpy(float), h.heritage_weight.to_numpy(float)
    speed = float(spec["walk_speed_m_per_min"])
    T = (float(config["service_radius_m"]) / speed if config.get("service_radius_m") is not None
         else float(config["service_min"]))
    L = float(spec["heritage_min"])
    if min(speed, T, L) <= 0 or not np.isfinite([speed, T, L]).all():
        raise ValueError("Speed and service/linkage thresholds must be finite and positive")
    dc = euclidean_m(xy[0], xy[1]) / speed
    hc = euclidean_m(xy[2], xy[1]) / speed
    dh = euclidean_m(xy[0], xy[2]) / speed
    B = hc <= L
    if key == "synthetic":
        constraints_path = data / "linkage_constraints.csv"
        compatible, crosses_river = _linkage_constraints(
            constraints_path, h.heritage_id.tolist(), c.candidate_id.tolist())
        sources.append(constraints_path)
        B &= compatible
        river_policy = spec.get("euclidean_river_policy", "preserve_baseline_path_constraint")
        if river_policy == "preserve_baseline_path_constraint":
            B &= ~crosses_river
        elif river_policy != "ignore_path_constraint":
            raise ValueError("Unknown euclidean_river_policy")
        alpha = float(spec.get("priority_alpha", 0.05))
        linkage = "euclidean-distance+programme+" + river_policy
    else:
        linkage_policy = spec.get("linkage", "distance+theme")
        if linkage_policy == "distance+theme":
            requires = h.heritage_class.astype(str).str.strip().isin(NEEDS_INTERPRETIVE).to_numpy()
            interprets = c.poi_class.astype(str).str.strip().isin(INTERPRETIVE).to_numpy()
            B &= ~requires[:, None] | interprets[None, :]
        elif linkage_policy != "distance":
            raise ValueError("Real linkage must be distance or distance+theme")
        linkage = "euclidean-distance+" + linkage_policy.removeprefix("distance+")
        alpha_config = spec.get("priority_alpha", "auto_median")
        alpha = (float(np.log(2) / np.median(dh[dh > 0])) if alpha_config == "auto_median"
                 else float(alpha_config))
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("priority_alpha must be finite and positive")
    opportunity = (np.exp(-alpha * dh) * w[None, :]).sum(axis=1)
    U = priority_mask(opportunity, p, float(config["priority_share"]))
    problem = EHMCLPProblem(
        name="Synthetic-scaled-Euclidean" if key == "synthetic" else "Xi'an-delivered-1km-Euclidean",
        A=dc <= T, B=B, p=p, w=w, U=U, eligible=c.eligible.to_numpy(int),
        site_id=c.site_id.tolist(), K=int(config["M_list"][0]), lam=tuple(config["weights"]),
        site_capacity=config.get("site_capacity", 1), cand_id=c.candidate_id.tolist())
    for M in config["M_list"]:
        problem.K = M
        problem.validate_K()
    problem.K = config["M_list"][0]
    metadata = dict(
        dataset=key, name=problem.name, distance_metric="euclidean", priority_distance_metric="euclidean",
        priority_group_unchanged=False, n_demand=problem.nI, n_candidate=problem.nJ,
        n_heritage=problem.nH, eligible_candidates=len(problem.eligible_idx), max_M=problem.max_K,
        service_min=T, equivalent_walk_distance_m=T * speed, heritage_min=L,
        walk_speed_m_per_min=speed, total_population=problem.P, priority_population=problem.PU,
        priority_population_share=problem.PU / problem.P, heritage_weight_sum=problem.W,
        priority_alpha=alpha, linkage=linkage, weights=list(problem.lam),
        A_pairs=int(problem.A.sum()), B_pairs=int(problem.B.sum()),
        priority_mask_sha256=hashlib.sha256(U.tobytes()).hexdigest(),
        matrix_sha256={name: hashlib.sha256(getattr(problem, name).tobytes()).hexdigest()
                       for name in ["A", "B", "U"]},
        coordinate_system="synthetic planar Cartesian" if key == "synthetic" else "CGCS2000 / 3-degree Gauss-Kruger CM 108E",
        coordinate_unit="metre", source_format="aligned_point_tables",
        sources={path.relative_to(Path(base).parent).as_posix(): dict(
            size=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources})
    if key == "synthetic":
        metadata["generation"] = spec.get("generation", {})
        metadata["normalized_coordinate_span_m"] = float(spec["generation"]["span_m"])
    else:
        metadata["coordinate_authority"] = ["EPSG", "4545"]
        metadata["real_surface"] = "delivered_1km_full"
        metadata["grade_weights"] = {"国保": 4.0, "省保": 3.0, "市保": 2.0, "县级": 1.0}
    return problem, metadata
