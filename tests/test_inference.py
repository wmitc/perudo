"""Tests for the Bayesian opponent-belief model."""

from math import comb

import numpy as np
import pytest

from perudo.agents import CountingAgent, RandomAgent, play_game
from perudo.bids import Bid
from perudo.game import GameState
from perudo.inference import BayesianAgent, OpponentBelief, _enumerate
from perudo.probability import unknown_match_pmf


# --- enumeration -----------------------------------------------------------


@pytest.mark.parametrize("m", [1, 2, 3, 5])
def test_enumeration_counts_and_prior(m):
    vectors, match_counts, prior = _enumerate(m)
    assert len(vectors) == comb(m + 5, 5)
    assert all(v.sum() == m for v in vectors)
    assert prior.sum() == pytest.approx(1.0)
    assert match_counts.shape == vectors.shape


def test_match_counts_are_ace_aware():
    vectors, match_counts, _ = _enumerate(5)
    for v, mc in zip(vectors, match_counts):
        aces = v[0]
        assert mc[0] == aces  # aces face: literal aces only
        for f in range(2, 7):
            assert mc[f - 1] == v[f - 1] + aces  # other faces: self + wild aces


# --- prior recovers the binomial counting model ----------------------------


def test_prior_match_pmf_matches_binomial():
    # With no observations, the posterior is the prior, so an opponent's match
    # distribution must equal the naive binomial used by the counting model.
    belief = OpponentBelief(num_dice=5, beta=1.0)
    for face in (1, 4):
        got = belief.match_pmf(face)
        want = unknown_match_pmf(5, face)
        assert np.allclose(got, want)


def test_match_pmf_is_normalized():
    belief = OpponentBelief(num_dice=5, beta=1.0)
    belief.update(5)
    pmf = belief.match_pmf(5)
    assert pmf.sum() == pytest.approx(1.0)
    assert len(pmf) == 6


# --- the core inference property -------------------------------------------


def test_posterior_concentrates_on_bid_face():
    belief = OpponentBelief(num_dice=5, beta=1.0)
    prior_mean = belief.expected_matches(5)
    for _ in range(3):
        belief.update(5)  # opponent keeps bidding 5s
    posterior_mean = belief.expected_matches(5)
    assert posterior_mean > prior_mean


def test_beta_zero_ignores_bids():
    belief = OpponentBelief(num_dice=5, beta=0.0)
    before = belief.match_pmf(5).copy()
    belief.update(5)
    after = belief.match_pmf(5)
    assert np.allclose(before, after)  # no learning when beta = 0


def test_higher_beta_concentrates_more():
    weak = OpponentBelief(num_dice=5, beta=0.5)
    strong = OpponentBelief(num_dice=5, beta=2.0)
    weak.update(5)
    strong.update(5)
    assert strong.expected_matches(5) > weak.expected_matches(5)


# --- informed estimate beats naive on a known setup ------------------------


def make_view(agent, opponent_dice, my_hand, standing_bid, opponent_bids):
    """Set up a 2-player round where the opponent has made some bids."""
    gs = GameState(num_players=2, seed=0)
    gs.dice_counts = [opponent_dice, len(my_hand)]
    gs.hands = [tuple(range(0)), tuple(my_hand)]  # opponent hand hidden/irrelevant
    gs.current_player = 1
    agent.observe_round_start(_view(gs, 1))
    for bid in opponent_bids:
        agent.observe_bid(0, bid)
    gs.current_bid = standing_bid
    gs.last_bidder = 0
    return _view(gs, 1)


def _view(gs, player):
    from perudo.agents import PlayerView

    return PlayerView.from_state(gs, player)


def test_informed_probability_exceeds_naive_after_aggressive_bidding():
    # Opponent (5 dice) has bid 5s repeatedly; I hold no 5s. The Bayesian bot
    # should think a 5s bid is MORE likely true than the history-blind counter.
    agent = BayesianAgent(beta=1.0)
    standing = Bid(4, 5)
    view = make_view(
        agent,
        opponent_dice=5,
        my_hand=[2, 3, 4, 6, 6],  # zero 5s of my own
        standing_bid=standing,
        opponent_bids=[Bid(2, 5), Bid(3, 5), Bid(4, 5)],
    )
    informed = agent.informed_bid_probability(standing)
    naive = agent.naive_bid_probability(view, standing)
    assert informed > naive


def test_informed_equals_naive_without_observations():
    # No bids observed → posterior is the prior → informed must equal naive.
    agent = BayesianAgent(beta=1.0)
    standing = Bid(3, 5)
    view = make_view(
        agent,
        opponent_dice=5,
        my_hand=[2, 3, 4, 6, 6],
        standing_bid=standing,
        opponent_bids=[],
    )
    informed = agent.informed_bid_probability(standing)
    naive = agent.naive_bid_probability(view, standing)
    assert informed == pytest.approx(naive)


# --- integration -----------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_bayesian_agent_plays_a_full_game(seed):
    agents = [BayesianAgent("B"), CountingAgent("C"), RandomAgent("R", seed=seed)]
    winner = play_game(agents, seed=seed)
    assert 0 <= winner < 3


def test_belief_snapshot_populated_after_decision():
    agent = BayesianAgent(beta=1.0)
    view = make_view(
        agent,
        opponent_dice=5,
        my_hand=[2, 3, 4, 6, 6],
        standing_bid=Bid(3, 5),
        opponent_bids=[Bid(2, 5)],
    )
    agent.decide(view)
    assert set(agent.last_belief) == {1, 2, 3, 4, 5, 6}
    assert all(v >= 0 for v in agent.last_belief.values())
