"""Tests for the tournament and calibration harness."""

import pytest

from perudo.agents import CountingAgent, RandomAgent
from perudo.inference import BayesianAgent
from perudo.simulate import (
    brier_score,
    collect_calibration,
    head_to_head,
    reliability_curve,
    tournament,
    win_rate,
)


def test_tournament_wins_sum_to_games():
    factories = [lambda: CountingAgent("C"), lambda: RandomAgent("R", seed=0)]
    wins = tournament(factories, n_games=20, seed=0)
    assert sum(wins) == 20
    assert len(wins) == 2


def test_win_rate_helper():
    assert win_rate([7, 3], 0) == pytest.approx(0.7)
    assert win_rate([0, 0], 0) == 0.0


def test_counting_beats_random_decisively():
    rate = head_to_head(
        lambda: CountingAgent("C"), lambda: RandomAgent("R", seed=0), n_games=40
    )
    assert rate > 0.8


def test_bayesian_beats_counting():
    # The headline result: reading the bids wins more games.
    rate = head_to_head(
        lambda: BayesianAgent("B"), lambda: CountingAgent("C"), n_games=80
    )
    assert rate > 0.55


# --- calibration metrics ---------------------------------------------------


def test_brier_score_extremes():
    assert brier_score([(1.0, True), (0.0, False)]) == pytest.approx(0.0)
    assert brier_score([(0.5, True), (0.5, False)]) == pytest.approx(0.25)


def test_brier_score_empty_is_nan():
    import math

    assert math.isnan(brier_score([]))


def test_reliability_curve_structure():
    records = [(0.05, False), (0.15, False), (0.95, True), (0.85, True)]
    rows = reliability_curve(records, n_bins=10)
    assert rows
    for row in rows:
        assert 0.0 <= row["mean_pred"] <= 1.0
        assert 0.0 <= row["emp_freq"] <= 1.0
        assert row["count"] >= 1
    assert sum(r["count"] for r in rows) == len(records)


def test_collect_calibration_produces_valid_records():
    records = collect_calibration(n_games=10, seed=0)
    assert records
    assert all(0.0 <= p <= 1.0 for p, _ in records)
    assert all(isinstance(o, bool) for _, o in records)


def test_bayesian_is_reasonably_calibrated():
    # Over enough bids the Brier score should beat the uninformative 0.25.
    records = collect_calibration(n_games=60, seed=1)
    assert brier_score(records) < 0.25
