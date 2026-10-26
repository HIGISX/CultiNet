import math
import sys
from pathlib import Path

import torch


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(MODEL_DIR))

from problems.MCLP.problem_MCLP import MCLP


def _dataset(facility_x, demand=1.0, radius=0.1):
    batch_size = len(facility_x)
    return {
        "users": torch.zeros(batch_size, 1, 2),
        "facilities": torch.tensor(
            [[[x, 0.0] for x in row] for row in facility_x],
            dtype=torch.float32,
        ),
        "demand": torch.full((batch_size, 1, 1), demand),
        "r": torch.full((batch_size,), radius),
    }


def test_uses_best_complete_pair_score_when_decay_values_tie():
    # Both facilities are in F(d) == 1's plateau.  The complete objective must
    # therefore select the closer facility, independent of pi's order.
    dataset = _dataset([[0.02, 0.01]])
    objective = MCLP.get_total_num(dataset, torch.tensor([[0, 1]]))

    expected = 0.6 - 0.4 * 0.01
    assert math.isclose(objective.item(), expected, rel_tol=0.0, abs_tol=1e-6)


def test_user_may_remain_unassigned_when_every_pair_score_is_negative():
    dataset = _dataset([[0.04]], demand=0.001)
    objective = MCLP.get_total_num(dataset, torch.tensor([[0]]))

    assert objective.item() == 0.0


def test_radius_boundary_is_included_like_the_solver_adjacency_matrix():
    dataset = _dataset([[0.01]], radius=0.01)
    objective = MCLP.get_total_num(dataset, torch.tensor([[0]]))

    expected = 0.6 - 0.4 * 0.01
    assert math.isclose(objective.item(), expected, rel_tol=0.0, abs_tol=1e-6)


def test_supports_one_radius_per_batch_item_and_two_dimensional_demand():
    dataset = _dataset([[0.01], [0.01]], radius=0.01)
    dataset["r"] = torch.tensor([0.01, 0.005])
    dataset["demand"] = dataset["demand"].squeeze(-1)

    objective = MCLP.get_total_num(dataset, torch.tensor([[0], [0]]))

    expected = torch.tensor([0.6 - 0.4 * 0.01, 0.0])
    torch.testing.assert_close(objective, expected)
