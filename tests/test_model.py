"""Checks the worked example from the slides: K = 2, starting at B and E."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kmeans_model import WORKED_EXAMPLE, WORKED_START, KMeansModel  # noqa: E402


def make_worked():
    m = KMeansModel()
    m.points = [[n, x, y] for n, x, y in WORKED_EXAMPLE]
    m.k = 2
    m.centroids = [c[:] for c in WORKED_START]
    m.clear_run()
    return m


def test_worked_example_converges():
    m = make_worked()
    while m.phase != "converged":
        m.step()
    assert m.iteration == 3
    assert m.centroids == [[5.0, 7.0], [4.0, 2.0]]
    assert [round(j, 2) for j in m.J_hist] == [66.0, 13.25, 8.0]


def test_objective_never_increases():
    m = make_worked()
    while m.phase != "converged":
        m.step()
    assert all(a >= b for a, b in zip(m.J_hist, m.J_hist[1:]))
