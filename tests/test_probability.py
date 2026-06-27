"""Tests for the binomial counting model, cross-checked against brute force."""

import itertools

import numpy as np
import pytest

from perudo.bids import Bid
from perudo.dice import count_face
from perudo.probability import (
    at_least,
    bid_true_probability,
    binomial_pmf,
    convolve,
    match_probability,
    unknown_match_pmf,
)


def brute_force_match_pmf(num_unknown, face):
    """Exact PMF of matches among ``num_unknown`` fair dice, by enumeration."""
    pmf = np.zeros(num_unknown + 1)
    for outcome in itertools.product(range(1, 7), repeat=num_unknown):
        pmf[count_face(outcome, face)] += 1
    return pmf / pmf.sum()


def test_match_probability_values():
    assert match_probability(1) == pytest.approx(1 / 6)
    for face in range(2, 7):
        assert match_probability(face) == pytest.approx(2 / 6)


def test_binomial_pmf_is_normalized():
    pmf = binomial_pmf(5, 1 / 3)
    assert pmf.sum() == pytest.approx(1.0)
    assert len(pmf) == 6


@pytest.mark.parametrize("num_unknown", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("face", [1, 4])
def test_unknown_match_pmf_matches_brute_force(num_unknown, face):
    got = unknown_match_pmf(num_unknown, face)
    want = brute_force_match_pmf(num_unknown, face)
    assert np.allclose(got, want)


def test_convolve_sums_independent_counts():
    # two independent dice, matches to face 4 (p = 1/3 each)
    single = unknown_match_pmf(1, 4)
    got = convolve(single, single)
    want = unknown_match_pmf(2, 4)
    assert np.allclose(got, want)


def test_at_least_edges_and_value():
    pmf = unknown_match_pmf(3, 4)
    assert at_least(pmf, 0) == pytest.approx(1.0)
    assert at_least(pmf, 4) == 0.0  # impossible: only 3 dice
    assert at_least(pmf, 1) == pytest.approx(1 - pmf[0])


@pytest.mark.parametrize("quantity", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("face", [1, 5])
@pytest.mark.parametrize("known,num_unknown", [(0, 5), (2, 3), (1, 4)])
def test_bid_true_probability_matches_brute_force(quantity, face, known, num_unknown):
    got = bid_true_probability(Bid(quantity, face), known, num_unknown)
    pmf = brute_force_match_pmf(num_unknown, face)
    want = at_least(pmf, quantity - known)
    assert got == pytest.approx(want)


def test_bid_true_probability_certain_and_impossible():
    # already satisfied by known dice
    assert bid_true_probability(Bid(2, 5), known_matches=2, num_unknown=4) == 1.0
    # needs more than the unknown dice can supply
    assert bid_true_probability(Bid(9, 5), known_matches=0, num_unknown=4) == 0.0
