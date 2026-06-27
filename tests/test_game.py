"""Tests for game state, Dudo resolution, and round bookkeeping."""

import pytest

from perudo.bids import Bid
from perudo.game import GameState


def make_state(hands, dice_counts=None):
    """A GameState with hands forced for deterministic resolution tests."""
    n = len(hands)
    gs = GameState(num_players=n, seed=0)
    gs.hands = [tuple(h) for h in hands]
    gs.dice_counts = list(dice_counts) if dice_counts else [len(h) for h in hands]
    return gs


def test_total_and_active_players():
    gs = GameState(num_players=3, seed=1)
    assert gs.total_dice() == 15
    assert gs.active_players() == [0, 1, 2]
    gs.dice_counts[1] = 0
    assert gs.active_players() == [0, 2]


def test_next_active_player_skips_eliminated():
    gs = GameState(num_players=4, seed=1)
    gs.dice_counts = [2, 0, 0, 3]
    assert gs.next_active_player(0) == 3
    assert gs.next_active_player(3) == 0


def test_count_face_total_is_ace_aware():
    gs = make_state([(1, 5, 5), (1, 3, 6)])
    # 5s: two 5s + two aces = 4
    assert gs.count_face_total(5) == 4
    # aces themselves: 2
    assert gs.count_face_total(1) == 2


def test_dudo_when_bid_holds_challenger_loses():
    gs = make_state([(5, 5, 5), (1, 2, 4)])  # four 5s on the table (incl. ace)
    gs.current_bid = Bid(4, 5)
    gs.last_bidder = 0
    gs.current_player = 1  # player 1 challenges
    result = gs.apply_dudo(1)
    assert result.actual_count == 4
    assert result.bid_holds is True
    assert result.loser == 1
    assert gs.dice_counts[1] == 2  # lost one die


def test_dudo_when_bid_fails_bidder_loses():
    gs = make_state([(5, 5, 2), (2, 3, 4)])  # only two 5s on the table
    gs.current_bid = Bid(4, 5)
    gs.last_bidder = 0
    gs.current_player = 1
    result = gs.apply_dudo(1)
    assert result.actual_count == 2
    assert result.bid_holds is False
    assert result.loser == 0
    assert gs.dice_counts[0] == 2


def test_dudo_exact_count_holds_for_bidder():
    # actual == quantity counts as the bid holding (>=)
    gs = make_state([(5, 5, 2), (2, 3, 4)])
    gs.current_bid = Bid(2, 5)
    gs.last_bidder = 0
    gs.current_player = 1
    result = gs.apply_dudo(1)
    assert result.bid_holds is True
    assert result.loser == 1


def test_cannot_dudo_without_standing_bid():
    gs = make_state([(2, 3), (4, 5)])
    gs.current_player = 0
    with pytest.raises(ValueError):
        gs.apply_dudo(0)


def test_apply_bid_enforces_turn_and_legality():
    gs = GameState(num_players=2, seed=2)
    gs.start_round(0)
    with pytest.raises(ValueError):
        gs.apply_bid(1, Bid(1, 5))  # not player 1's turn
    gs.apply_bid(0, Bid(2, 5))
    assert gs.current_bid == Bid(2, 5)
    assert gs.current_player == 1
    with pytest.raises(ValueError):
        gs.apply_bid(1, Bid(1, 6))  # not a legal raise


def test_apply_bid_rejects_quantity_above_dice_in_play():
    gs = GameState(num_players=2, seed=3)  # 10 dice
    gs.start_round(0)
    with pytest.raises(ValueError):
        gs.apply_bid(0, Bid(11, 4))


def test_elimination_and_winner_flow():
    gs = GameState(num_players=2, seed=4)
    gs.dice_counts = [1, 1]
    gs.hands = [(5,), (2,)]
    gs.current_bid = Bid(2, 5)  # claims two 5s; only one exists
    gs.last_bidder = 0
    gs.current_player = 1
    result = gs.apply_dudo(1)
    assert result.loser == 0
    assert gs.dice_counts[0] == 0
    assert gs.is_over()
    assert gs.winner() == 1


def test_next_starting_player_after_elimination():
    gs = GameState(num_players=3, seed=5)
    gs.dice_counts = [1, 0, 2]  # player 1 already out
    # if loser 1 is out, next starter is the next active player after 1
    assert gs.next_starting_player(1) == 2
    # if loser still has dice, they start
    assert gs.next_starting_player(0) == 0


def test_start_round_rolls_correct_counts():
    gs = GameState(num_players=3, seed=6)
    gs.dice_counts = [3, 0, 5]
    gs.start_round(0)
    assert [len(h) for h in gs.hands] == [3, 0, 5]
    assert gs.current_player == 0
    assert gs.current_bid is None
    assert gs.round_number == 1
