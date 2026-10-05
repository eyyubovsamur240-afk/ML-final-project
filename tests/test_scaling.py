"""Tests for the fit-time scaling helpers (src/scaling.py)."""

import numpy as np

from src.scaling import loglog_slope


def test_loglog_slope_recovers_the_power():
    n = np.array([1_000, 2_000, 4_000, 8_000, 16_000])
    np.testing.assert_allclose(loglog_slope(n, 3e-6 * n ** 1.0), 1.0)
    np.testing.assert_allclose(loglog_slope(n, 1e-9 * n ** 2.0), 2.0)


def test_loglog_slope_uses_only_the_largest_sizes():
    n = np.array([10, 100, 1_000, 10_000])
    t = np.array([0.5, 0.5, 1e-3, 1e-2])          # fixed overhead dominates the small sizes
    np.testing.assert_allclose(loglog_slope(n, t, last=2), 1.0)


def test_loglog_slope_needs_two_positive_times():
    assert np.isnan(loglog_slope([10, 20], [0.0, 0.0]))
