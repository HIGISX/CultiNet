"""
Unified EH-MCLP problem object for algorithm benchmarking.
===============================================================================
Wraps the delivered EH-MCLP instance (A, B, p, w, U, s, K, M_g) behind one
interface used by all four algorithms and by the exact solvers, so that
Gurobi / OR-Tools / SA / PSO / INSGA-II all optimise the SAME objective under
the SAME constraints.

Objective (model doc eq. 10/11)
    F = 0.35*F1 + 0.40*F2 + 0.25*F3
Constraints (doc 12, 18, 19)
    sum_j x_j = K,  x_j <= s_j,  sum_{j in J_g} x_j <= M_g

Fitness is evaluated with 64-bit bitsets, which makes tens of thousands of
objective evaluations per second possible even for the 5,530 x 587 real
instance.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


def _pack_bits(mat: np.ndarray) -> np.ndarray:
    """(nJ, nI) 0/1 matrix -> (nJ, n_words) uint64 bitmask."""
    nJ, nI = mat.shape
    packed = np.packbits(mat.astype(np.uint8), axis=1, bitorder="little")
    n_words = (nI + 63) // 64
    out = np.zeros((nJ, n_words * 8), dtype=np.uint8)
    out[:, :packed.shape[1]] = packed
    return out.view(np.uint64)


@dataclass
class EHMCLPProblem:
    """One EH-MCLP instance."""

    name: str
    A: np.ndarray            # (nI, nJ) resident covering matrix
    B: np.ndarray            # (nH, nJ) heritage linkage matrix
    p: np.ndarray            # (nI,) population
    w: np.ndarray            # (nH,) heritage weight
    U: np.ndarray            # (nI,) bool priority mask
    eligible: np.ndarray     # (nJ,) 0/1 implementability s_j
    site_id: List[str]       # (nJ,) site of each candidate
    K: int
    lam: Tuple[float, float, float] = (0.35, 0.40, 0.25)
    site_capacity: int = 1
    objective_mode: str = "norm"   # "norm" = doc eq.10 ratios | "raw" = total quantities

    # derived
    cand_id: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.A = np.asarray(self.A, dtype=bool)
        self.B = np.asarray(self.B, dtype=bool)
        self.p = np.asarray(self.p, dtype=float)
        self.w = np.asarray(self.w, dtype=float)
        self.U = np.asarray(self.U, dtype=bool)
        self.eligible = np.asarray(self.eligible, dtype=int)
        self.nI, self.nJ = self.A.shape
        self.nH = self.B.shape[0]
        if (self.B.shape[1] != self.nJ or self.p.shape != (self.nI,)
                or self.w.shape != (self.nH,) or self.U.shape != (self.nI,)
                or self.eligible.shape != (self.nJ,) or len(self.site_id) != self.nJ):
            raise ValueError("EH-MCLP matrix / attribute dimensions do not agree")
        if (not np.isfinite(self.p).all() or not np.isfinite(self.w).all()
                or (self.p < 0).any() or (self.w < 0).any()
                or not np.isin(self.eligible, [0, 1]).all()):
            raise ValueError("Population, weights and eligibility must be valid nonnegative data")
        if self.objective_mode not in {"norm", "raw"}:
            raise ValueError("objective_mode must be norm or raw")
        if self.site_capacity is not None and (not isinstance(self.site_capacity, int)
                                               or self.site_capacity < 1):
            raise ValueError("site_capacity must be a positive integer or None")
        self.P = float(self.p.sum())
        self.PU = float(self.p[self.U].sum())
        self.W = float(self.w.sum())
        self.lam = tuple(float(v) for v in self.lam)
        if (len(self.lam) != 3 or not np.isfinite(self.lam).all()
                or min(self.lam) < 0 or not np.isclose(sum(self.lam), 1.0)):
            raise ValueError("The three EH-MCLP weights must be nonnegative and sum to 1")
        if min(self.P, self.PU, self.W) <= 0:
            raise ValueError("Total population, priority population and heritage weight must be positive")
        # For objective="raw" the model maximises the TOTAL quantities
        #     R = l1*sum p_i y_i + l2*sum_U p_i y_i + l3*sum w_h z_h
        # divided by the constant R_max = l1*P + l2*P_U + l3*W so that the
        # value stays in [0,1] and SA's temperature calibration is unchanged.
        # The divisor is a positive constant, so the optimum is identical.
        self._raw_denom = (self.lam[0] * self.P + self.lam[1] * self.PU
                           + self.lam[2] * self.W) if self.objective_mode == "raw" else 1.0
        self._w_cov = (self.lam[0] / self._raw_denom) if self.objective_mode == "raw" else self.lam[0] / self.P
        self._w_prio = (self.lam[1] / self._raw_denom) if self.objective_mode == "raw" else self.lam[1] / self.PU
        self._w_link = (self.lam[2] / self._raw_denom) if self.objective_mode == "raw" else self.lam[2] / self.W

        # ---- fast bitset fitness ------------------------------------------
        self._Abit = _pack_bits(self.A.T)          # (nJ, words) coverage
        self._Bbit = _pack_bits(self.B.T)          # (nJ, words) heritage link
        self._nwA = (self.nI + 63) // 64
        self._nwB = (self.nH + 63) // 64
        self._u16A = self._Abit.view(np.uint16)
        self._u16B = self._Bbit.view(np.uint16)

        # ---- feasibility --------------------------------------------------
        self.eligible_idx = np.flatnonzero(self.eligible == 1).tolist()
        groups: dict = {}
        for j, s in enumerate(self.site_id):
            groups.setdefault(s, []).append(j)
        self.groups = {g: v for g, v in groups.items() if len(v) > 1}
        self.max_K = self._max_feasible()
        self.validate_K()

        # ---- single-candidate marginal contributions (for greedy / warm start)
        self.cov_pop = self.A.T.astype(np.float64) @ self.p            # (nJ,)
        self.cov_prio = self.A.T.astype(np.float64) @ (self.p * self.U)
        self.link_w = self.B.T.astype(np.float64) @ self.w

    # ------------------------------------------------------------------
    def _max_feasible(self) -> int:
        counts = {}
        for j in self.eligible_idx:
            s = self.site_id[j]
            counts[s] = counts.get(s, 0) + 1
        return sum(min(n, self.site_capacity) if self.site_capacity is not None else n
                   for n in counts.values())

    def validate_K(self) -> None:
        if (isinstance(self.K, bool) or not isinstance(self.K, (int, np.integer))
                or not 1 <= self.K <= self.max_K):
            raise ValueError(f"K={self.K} is infeasible; allowed range is 1..{self.max_K}")

    # ------------------------------------------------------------------
    def is_feasible(self, sel: Sequence[int]) -> bool:
        if len(sel) != self.K:
            return False
        if len(set(sel)) != len(sel):
            return False
        for j in sel:
            if not isinstance(j, (int, np.integer)) or not 0 <= j < self.nJ:
                return False
            if self.eligible[j] != 1:
                return False
        counts: dict = {}
        for j in sel:
            s = self.site_id[j]
            counts[s] = counts.get(s, 0) + 1
            if self.site_capacity is not None and counts[s] > self.site_capacity:
                return False
        return True

    def repair(self, sel: Sequence[int], rng: np.random.Generator) -> List[int]:
        """Make `sel` feasible: drop ineligible / site-conflicting, refill."""
        self.validate_K()
        counts, out, used = {}, [], set()
        for j in sel:
            j = int(j)
            if not 0 <= j < self.nJ or self.eligible[j] != 1 or j in used:
                continue
            s = self.site_id[j]
            if self.site_capacity is not None and counts.get(s, 0) >= self.site_capacity:
                continue
            out.append(j)
            used.add(j)
            counts[s] = counts.get(s, 0) + 1
            if len(out) == self.K:
                return out
        # Recheck capacity after EVERY insertion. A batch sample can choose two
        # candidates on the same previously unused site and violate the model.
        for j in rng.permutation(self.eligible_idx):
            j = int(j)
            s = self.site_id[j]
            if j in used or (self.site_capacity is not None
                             and counts.get(s, 0) >= self.site_capacity):
                continue
            out.append(j)
            used.add(j)
            counts[s] = counts.get(s, 0) + 1
            if len(out) == self.K:
                return out
        raise RuntimeError("Unable to repair a selection despite feasible K")

    def random_solution(self, rng: np.random.Generator) -> List[int]:
        return self.repair(rng.choice(self.eligible_idx, size=self.K, replace=False), rng)

    # ------------------------------------------------------------------
    def fitness(self, sel: Sequence[int]) -> Tuple[float, float, float, float]:
        """(F1, F2, F3, F) of a selection."""
        if len(sel) == 0:
            return 0.0, 0.0, 0.0, 0.0
        idx = np.asarray(sel, dtype=np.intp)
        uA = np.bitwise_or.reduce(self._u16A[idx], axis=0)
        cov = np.unpackbits(uA.view(np.uint8), bitorder="little")[: self.nI].astype(bool)
        uB = np.bitwise_or.reduce(self._u16B[idx], axis=0)
        lin = np.unpackbits(uB.view(np.uint8), bitorder="little")[: self.nH].astype(bool)
        F1 = float(self.p[cov].sum() / self.P)
        F2 = float(self.p[cov & self.U].sum() / self.PU)
        F3 = float(self.w[lin].sum() / self.W)
        # Raw and component-normalised objectives generally have DIFFERENT optima.
        F = float(self._w_cov * self.p[cov].sum()
                  + self._w_prio * self.p[cov & self.U].sum()
                  + self._w_link * self.w[lin].sum())
        return F1, F2, F3, F

    def objective(self, sel: Sequence[int]) -> float:
        return self.fitness(sel)[3]

    def objectives3(self, sel: Sequence[int]) -> Tuple[float, float, float]:
        f1, f2, f3, _ = self.fitness(sel)
        return f1, f2, f3

    # ------------------------------------------------------------------
    def describe(self) -> str:
        return (f"[{self.name}] |I|={self.nI:,} |J|={self.nJ:,} |H|={self.nH} "
                f"eligible={len(self.eligible_idx)} K={self.K} "
                f"P={self.P:,.0f} P_U={self.PU:,.0f} W={self.W:.0f} "
                f"|A|={int(self.A.sum()):,} |B|={int(self.B.sum()):,} "
                f"shared_sites={len(self.groups)} max_K={self.max_K}")


# ---------------------------------------------------------------------------
def load_synthetic(objective: str = "norm", T: float = 15.0,
                   L: float = 10.0) -> EHMCLPProblem:
    """The delivered CultiNet synthetic benchmark v1 (K = 8)."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "eh_mclp_gurobi"))
    import eh_mclp_gurobi as eh
    inst = eh.load_instance(T=T, L=L)
    return EHMCLPProblem(
        name="Synthetic-v1", A=inst.A, B=inst.B, p=inst.p, w=inst.w, U=inst.U,
        eligible=inst.s, site_id=list(inst.site_id), K=8,
        lam=(0.35, 0.40, 0.25), site_capacity=1, cand_id=list(inst.cand_id),
        objective_mode=objective)


def load_real(K: int = 50, T: float = 15.0, L: float = 30.0, objective: str = "norm") -> EHMCLPProblem:
    """The Xi'an real instance on the 300 m built-up demand surface."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "xian_eh_mclp"))
    import xian_run as xr
    inst, _ = xr.build_instance("300m_builtup", T=T, L=L)
    return EHMCLPProblem(
        name="Xi'an-300m", A=inst.A, B=inst.B, p=inst.p, w=inst.w, U=inst.U,
        eligible=inst.s, site_id=list(inst.site_id), K=K,
        lam=(0.35, 0.40, 0.25), site_capacity=1, cand_id=list(inst.cand_id),
        objective_mode=objective)


if __name__ == "__main__":
    for pr in (load_synthetic(), load_real()):
        print(pr.describe())
        rng = np.random.default_rng(0)
        s = pr.random_solution(rng)
        print("   random solution fitness:", tuple(round(v, 4) for v in pr.fitness(s)))
