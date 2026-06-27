"""Tests for bid construction and the raise-legality ladder."""

import math

import pytest

from perudo.bids import Bid, is_legal_raise, legal_raises


def test_bid_validation():
    with pytest.raises(ValueError):
        Bid(0, 3)
    with pytest.raises(ValueError):
        Bid(2, 7)


def test_opening_bid_cannot_be_aces():
    assert is_legal_raise(None, Bid(3, 5))
    assert not is_legal_raise(None, Bid(3, 1))


def test_normal_raise_higher_quantity_any_face():
    prev = Bid(3, 5)
    assert is_legal_raise(prev, Bid(4, 2))  # more dice, lower face is fine
    assert is_legal_raise(prev, Bid(4, 6))


def test_normal_raise_same_quantity_needs_higher_face():
    prev = Bid(3, 4)
    assert is_legal_raise(prev, Bid(3, 5))
    assert not is_legal_raise(prev, Bid(3, 4))
    assert not is_legal_raise(prev, Bid(3, 3))


def test_lower_bid_is_illegal():
    prev = Bid(4, 4)
    assert not is_legal_raise(prev, Bid(3, 6))


def test_switch_to_aces_halves_quantity():
    prev = Bid(5, 4)
    # need ceil(5/2) = 3 aces
    assert is_legal_raise(prev, Bid(3, 1))
    assert not is_legal_raise(prev, Bid(2, 1))


def test_switch_to_aces_rounds_up():
    prev = Bid(4, 6)
    assert math.ceil(4 / 2) == 2
    assert is_legal_raise(prev, Bid(2, 1))
    assert not is_legal_raise(prev, Bid(1, 1))


def test_staying_on_aces_must_increase():
    prev = Bid(2, 1)
    assert is_legal_raise(prev, Bid(3, 1))
    assert not is_legal_raise(prev, Bid(2, 1))


def test_switch_from_aces_doubles_plus_one():
    prev = Bid(2, 1)
    # need 2*2 + 1 = 5 of a normal face
    assert is_legal_raise(prev, Bid(5, 6))
    assert not is_legal_raise(prev, Bid(4, 6))


def test_legal_raises_are_all_legal_and_bounded():
    prev = Bid(2, 3)
    raises = legal_raises(prev, max_quantity=10)
    assert raises  # non-empty
    assert all(is_legal_raise(prev, b) for b in raises)
    assert all(b.quantity <= 10 for b in raises)
    # a known-good raise is present; a known-bad one is not
    assert Bid(3, 2) in raises
    assert Bid(2, 2) not in raises
