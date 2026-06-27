"""Tests for baseline agents and the game driver."""

import random

import pytest

from perudo.agents import (
    DUDO,
    Agent,
    CountingAgent,
    HumanAgent,
    PlayerView,
    RandomAgent,
    play_game,
)
from perudo.bids import Bid, is_legal_raise
from perudo.game import GameState


def view_after_bid(hand_other, hand_me, bid, me=1):
    """Build a PlayerView for `me` with a standing `bid` made by the other player."""
    gs = GameState(num_players=2, seed=0)
    gs.hands = [tuple(hand_other), tuple(hand_me)]
    gs.dice_counts = [len(hand_other), len(hand_me)]
    gs.current_bid = bid
    gs.last_bidder = 1 - me
    gs.current_player = me
    return PlayerView.from_state(gs, me)


def test_playerview_unknown_count():
    view = view_after_bid([2, 3, 4], [5, 5], Bid(2, 5))
    assert view.total_dice == 5
    assert view.num_unknown == 3  # five total minus my two


# --- RandomAgent -----------------------------------------------------------


def test_random_agent_opens_with_legal_bid():
    gs = GameState(num_players=2, seed=1)
    gs.start_round(0)
    agent = RandomAgent(seed=1)
    action = agent.decide(PlayerView.from_state(gs, 0))
    assert isinstance(action, Bid)
    assert is_legal_raise(None, action)


def test_random_agent_dudos_when_no_legal_raise():
    # standing bid is the maximum possible: all dice on the top face
    view = view_after_bid([6, 6, 6], [6, 6], Bid(5, 6))
    agent = RandomAgent(seed=1)
    assert agent.decide(view) is DUDO


# --- CountingAgent ---------------------------------------------------------


def test_counting_agent_dudos_an_impossible_bid():
    # bid claims 5 sixes but I hold none and only 3 unknown dice exist
    view = view_after_bid([2, 3, 4], [2, 2], Bid(5, 6))
    agent = CountingAgent()
    assert agent.decide(view) is DUDO


def test_counting_agent_trusts_a_bid_it_can_back():
    # I hold three 5s (incl. ace); bid of two 5s is essentially certain → raise, not Dudo
    view = view_after_bid([2, 3, 4], [1, 5, 5], Bid(2, 5))
    agent = CountingAgent()
    action = agent.decide(view)
    assert isinstance(action, Bid)
    assert is_legal_raise(view.current_bid, action)


def test_counting_agent_opening_reflects_its_hand():
    gs = GameState(num_players=2, seed=2)
    gs.hands = [(3, 5, 5, 5, 6), ()]
    gs.dice_counts = [5, 0]
    gs.current_bid = None
    gs.current_player = 0
    action = CountingAgent().decide(PlayerView.from_state(gs, 0))
    assert isinstance(action, Bid)
    assert action.face == 5 and action.quantity == 3  # holds the most 5s


def test_counting_agent_probability_monotonic_in_quantity():
    view = view_after_bid([2, 3, 4], [5, 5], Bid(2, 5))
    agent = CountingAgent()
    p_low = agent.bid_probability(view, Bid(2, 5))
    p_high = agent.bid_probability(view, Bid(5, 5))
    assert p_low > p_high


# --- HumanAgent ------------------------------------------------------------


def test_human_agent_parses_dudo_and_bid():
    view = view_after_bid([2, 3, 4], [5, 5], Bid(2, 5))
    moves = iter(["d"])
    agent = HumanAgent(input_fn=lambda _: next(moves), output_fn=lambda *_: None)
    assert agent.decide(view) is DUDO

    moves2 = iter(["3 5"])
    agent2 = HumanAgent(input_fn=lambda _: next(moves2), output_fn=lambda *_: None)
    assert agent2.decide(view) == Bid(3, 5)


def test_human_agent_rejects_then_accepts():
    view = view_after_bid([2, 3, 4], [5, 5], Bid(2, 5))
    moves = iter(["nonsense", "2 5", "3 5"])  # bad parse, illegal raise, then valid
    msgs = []
    agent = HumanAgent(input_fn=lambda _: next(moves), output_fn=msgs.append)
    assert agent.decide(view) == Bid(3, 5)
    assert len(msgs) == 2  # two rejections before the valid move


# --- play_game -------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_play_game_terminates_with_a_winner(seed):
    agents = [CountingAgent(f"A{i}") for i in range(3)]
    winner = play_game(agents, seed=seed)
    assert 0 <= winner < 3


def test_play_game_random_agents_terminate():
    agents = [RandomAgent(f"R{i}", seed=i) for i in range(3)]
    winner = play_game(agents, seed=123)
    assert 0 <= winner < 3


def test_observation_hooks_fire():
    class Spy(CountingAgent):
        def __init__(self, name):
            super().__init__(name)
            self.bids_seen = 0
            self.dudos_seen = 0
            self.rounds_seen = 0

        def observe_bid(self, player, bid):
            self.bids_seen += 1

        def observe_dudo(self, result):
            self.dudos_seen += 1

        def observe_round_start(self, view):
            self.rounds_seen += 1

    agents = [Spy(f"S{i}") for i in range(2)]
    play_game(agents, seed=7)
    assert all(a.rounds_seen > 0 for a in agents)
    assert all(a.dudos_seen > 0 for a in agents)
    assert all(a.bids_seen > 0 for a in agents)
