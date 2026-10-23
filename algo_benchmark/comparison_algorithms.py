"""EH-MCLP adapters; search parameters come from the ORIGINAL notebooks.

PSO and SA execute the original optimize/search methods. GA uses Algorithm/GA
(the class imported by Billboard_MCLP_GA.ipynb), including its 50/20/200/.1
settings. Only fitness and feasibility handling are replaced. NSGA-II keeps
the original random-parent / set-crossover / mutation / crowding procedure,
generalising its two objective coordinates to EH-MCLP's three components.
"""
from __future__ import annotations

import contextlib
import io
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Algorithm.GA import GeneticAlgorithm
from Algorithm.INSGA import NSGAIIForFacilityLocation
from Algorithm.PSO_distance import ParticleSwarmOptimization
from Algorithm.SA_distence import SimulatedAnnealingVectorized


PARAMETER_SOURCES = {
    "PSO": "EF_MCLP_PSO_distanse.ipynb: class defaults from Algorithm/PSO_distance.py",
    "GA": "Billboard_MCLP_GA.ipynb: Algorithm/GA.py::GeneticAlgorithm",
    "SA": "EF_MCLP_SA_distanse.ipynb: max_iter=1000, alpha=0.99; remaining class defaults",
    "NSGA-II": "EF_MCLP_INSGA-Ⅱ_distanse.ipynb: population_size=1000, generations=200, mutation_prob=0.2",
}


class ObjectiveAdapter:
    def bind_problem(self, problem, seed):
        self.problem = problem
        self.rng = np.random.default_rng(seed)
        self.full_ids = list(problem.eligible_idx)
        self.local_id = {j: i for i, j in enumerate(self.full_ids)}
        self.cache = {}
        self.n_evals = 0

    def evaluate(self, chromosome):
        self.n_evals += 1
        full = [self.full_ids[j] for j in chromosome]
        if not self.problem.is_feasible(full):
            full = self.problem.repair(full, self.rng)
            chromosome[:] = [self.local_id[j] for j in full]
        key = tuple(sorted(full))
        if key not in self.cache:
            self.cache[key] = self.problem.fitness(full)
        return self.cache[key]

    def fitness(self, chromosome):
        f1, _, _, F = self.evaluate(chromosome)
        return F, f1


class PSOAdapter(ObjectiveAdapter, ParticleSwarmOptimization):
    pass


class SAAdapter(ObjectiveAdapter, SimulatedAnnealingVectorized):
    pass


class GAAdapter(ObjectiveAdapter, GeneticAlgorithm):
    def fitness(self, chromosome):
        # The original GA minimises uncovered demand. Minimise -F instead.
        return -self.evaluate(chromosome)[3]

    def crossover(self, p1, p2):
        # Original GA returns an actual parent when parents have equal genes.
        # Copy it before mutation, or stored elite fitness becomes stale.
        c1, c2 = super().crossover(p1, p2)
        return list(c1), None if c2 is None else list(c2)

    def mutation(self, chromosome):
        if len(chromosome) == self.fac_num:
            return chromosome
        return super().mutation(chromosome)


class NSGAAdapter(ObjectiveAdapter, NSGAIIForFacilityLocation):
    @staticmethod
    def fronts(objectives):
        """Exact nondominated layers; componentwise, vectorised, stable ties."""
        objs = np.asarray(objectives)
        n = len(objs)
        weak = np.ones((n, n), dtype=bool)
        strict = np.zeros((n, n), dtype=bool)
        for k in range(objs.shape[1]):
            weak &= objs[:, k, None] <= objs[None, :, k]
            strict |= objs[:, k, None] < objs[None, :, k]
        dominates = weak & strict
        remaining = dominates.sum(axis=0)
        alive = np.ones(n, dtype=bool)
        result = []
        while alive.any():
            front = np.flatnonzero(alive & (remaining == 0))
            if not len(front):
                raise RuntimeError("Nondominated sorting cycle")
            result.append(front.tolist())
            alive[front] = False
            remaining -= dominates[front].sum(axis=0)
        return result

    def optimize_eh(self):
        population = [random.sample(range(self.m), self.p) for _ in range(self.population_size)]
        for _ in range(self.generations):
            offspring = []
            while len(offspring) < self.population_size:
                p1, p2 = random.sample(population, 2)
                c1, c2 = self.crossover(p1, p2)
                offspring.append(self.mutate(c1))
                if len(offspring) < self.population_size:
                    offspring.append(self.mutate(c2))
            combined = population + offspring
            objectives = [tuple(-v for v in self.evaluate(ind)[:3]) for ind in combined]
            next_pop = []
            for front in self.fronts(objectives):
                if len(next_pop) + len(front) <= self.population_size:
                    next_pop.extend(combined[i] for i in front)
                else:
                    distances = self.calculate_crowding_distance(front, objectives)
                    ordered = sorted(front, key=lambda i: distances[i], reverse=True)
                    next_pop.extend(combined[i] for i in ordered[:self.population_size - len(next_pop)])
                    break
            population = next_pop
        # Pick the maximum weighted EH-MCLP F on the FINAL Pareto front; do not
        # average different Pareto solutions as if they were independent runs.
        vals = [self.evaluate(ind) for ind in population]
        front = self.fronts([tuple(-v for v in val[:3]) for val in vals])[0]
        return population[max(front, key=lambda i: vals[i][3])], len(front)


def run_heuristic(name, problem, seed, params):
    random.seed(seed)
    np.random.seed(seed)
    cls = {"PSO": PSOAdapter, "SA": SAAdapter, "GA": GAAdapter, "NSGA-II": NSGAAdapter}[name]
    # Constructors/imports are outside measured solver time, as for exact
    # solvers; population initialisation, repair and objective calls are inside.
    if name == "GA":
        alg = cls(problem.nI, len(problem.eligible_idx), problem.K,
                  np.empty((len(problem.eligible_idx), 0)), 0, problem.p)
        for key, val in params.items():
            setattr(alg, key, val)
    else:
        alg = cls(n=problem.nI, m=len(problem.eligible_idx), p=problem.K,
                  users=None, facilities=None, demand=problem.p, w1=0, w2=0,
                  da=0, db=0, D_max=0, A=None, dist=None, **params)
    alg.bind_problem(problem, seed)
    t0 = time.perf_counter()
    pareto_size = None
    with contextlib.redirect_stdout(io.StringIO()):
        if name == "NSGA-II":
            local, pareto_size = alg.optimize_eh()
        elif name == "GA":
            alg.optimize()
            local = alg.top_chromosome.content
        else:
            alg.optimize()
            local = alg.gbest_pos if name == "PSO" else alg.best_solution
    elapsed = time.perf_counter() - t0
    selection = [alg.full_ids[j] for j in local]
    if not problem.is_feasible(selection):
        raise RuntimeError(f"{name} returned an infeasible selection")
    f1, f2, f3, F = problem.fitness(selection)
    return dict(method=name, seed=seed, status="COMPLETED", F=F, F1=f1, F2=f2, F3=f3,
                runtime_s=elapsed, selection=selection, selected_ids=[problem.cand_id[j] for j in selection],
                n_selected=len(selection), n_evals=alg.n_evals, n_unique_evals=len(alg.cache),
                pareto_size=pareto_size, upper_bound=None, certificate_error_abs=None)


def _coefficients(problem):
    resident = problem.p * (problem._w_cov + problem._w_prio * problem.U)
    heritage = problem.w * problem._w_link
    return np.concatenate([resident, heritage])


def _exact_result(method, problem, selection, elapsed, status, upper_bound, error, **extra):
    result = dict(method=method, seed=None, status=status, runtime_s=elapsed,
                  upper_bound=upper_bound, certificate_error_abs=error,
                  n_evals=None, n_unique_evals=None, pareto_size=None, **extra)
    if selection is None:
        return dict(result, selection=[], selected_ids=[], n_selected=0, F=None, F1=None, F2=None, F3=None)
    if not problem.is_feasible(selection):
        raise RuntimeError(f"{method} returned an infeasible selection")
    f1, f2, f3, F = problem.fitness(selection)
    return dict(result, selection=selection, selected_ids=[problem.cand_id[j] for j in selection],
                n_selected=len(selection), F=F, F1=f1, F2=f2, F3=f3)


def gurobi_solve(problem, params, seed=1):
    import gurobipy as gp
    from gurobipy import GRB
    t0 = time.perf_counter()
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model("EH-MCLP", env=env) as model:
            model.Params.TimeLimit = params["time_limit_s"]
            model.Params.Threads = params["threads"]
            model.Params.MIPGap = 0
            model.Params.MIPGapAbs = 0
            model.Params.Seed = seed
            # A fixed positive numerical multiplier improves small-coefficient
            # accuracy in the MIP; heuristics still see the literal F in [0,1].
            scale = 1e6
            x = model.addVars(problem.nJ, vtype=GRB.BINARY, name="x")
            for j in range(problem.nJ):
                x[j].UB = int(problem.eligible[j])
            y = model.addVars(problem.nI + problem.nH, vtype=GRB.BINARY, name="covered")
            model.addConstr(x.sum() == problem.K)
            if problem.site_capacity is not None:
                for indices in problem.groups.values():
                    model.addConstr(gp.quicksum(x[j] for j in indices) <= problem.site_capacity)
            for i, row in enumerate(np.vstack([problem.A, problem.B])):
                indices = np.flatnonzero(row)
                model.addConstr(y[i] <= gp.quicksum(x[int(j)] for j in indices))
            coeffs = _coefficients(problem)
            model.setObjective(gp.quicksum(float(c * scale) * y[i] for i, c in enumerate(coeffs)), GRB.MAXIMIZE)
            model.optimize()
            status = {GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.INFEASIBLE: "INFEASIBLE",
                      GRB.INTERRUPTED: "INTERRUPTED"}.get(model.Status, f"STATUS_{model.Status}")
            selection = [j for j in range(problem.nJ) if x[j].X > 0.5] if model.SolCount else None
            bound = float(model.ObjBound / scale) if np.isfinite(model.ObjBound) else None
            return _exact_result("Gurobi", problem, selection, time.perf_counter() - t0, status, bound, 0.0,
                                 solver_gap=float(model.MIPGap) if model.SolCount else None)


def ortools_solve(problem, params, seed=1):
    from ortools.sat.python import cp_model
    t0 = time.perf_counter()
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{j}") for j in range(problem.nJ)]
    y = [model.NewBoolVar(f"covered{i}") for i in range(problem.nI + problem.nH)]
    model.Add(sum(x) == problem.K)
    for j in range(problem.nJ):
        if not problem.eligible[j]:
            model.Add(x[j] == 0)
    if problem.site_capacity is not None:
        for indices in problem.groups.values():
            model.Add(sum(x[j] for j in indices) <= problem.site_capacity)
    for i, row in enumerate(np.vstack([problem.A, problem.B])):
        model.Add(y[i] <= sum(x[int(j)] for j in np.flatnonzero(row)))
    scale = int(params["integer_scale"])
    coeffs = _coefficients(problem)
    integers = np.rint(coeffs * scale).astype(np.int64)
    if int(integers.sum()) >= 2**62:
        raise ValueError("CP-SAT integer objective may overflow; reduce integer_scale")
    error = float(np.abs(coeffs - integers / scale).sum())
    model.Maximize(sum(int(c) * y[i] for i, c in enumerate(integers) if c))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = params["time_limit_s"]
    solver.parameters.num_search_workers = params["threads"]
    solver.parameters.random_seed = seed
    solver.parameters.linearization_level = params.get("linearization_level", 2)
    solver.parameters.relative_gap_limit = 0
    solver.parameters.absolute_gap_limit = 0
    st = solver.Solve(model)
    feasible = st in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    selection = [j for j in range(problem.nJ) if solver.Value(x[j])] if feasible else None
    # UB(original F) <= UB(integer objective)/scale + sum coefficient errors.
    # OPTIMAL is a certificate for the rounded problem, hence both the
    # original-objective bound and rounding error are retained explicitly.
    bound = float(solver.BestObjectiveBound()) / scale + error if feasible else None
    return _exact_result("OR-Tools", problem, selection, time.perf_counter() - t0,
                         solver.StatusName(st), bound, error, integer_scale=scale)
