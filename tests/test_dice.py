"""Tests for dice rolling and ace-aware counting."""

import random

import pytest

from perudo.dice import ACE, NUM_FACES, count_face, roll_hand


def test_roll_hand_size_and_range():
    rng = random.Random(0)
    hand = roll_hand(5, rng)
    assert len(hand) == 5
    assert all(1 <= d <= NUM_FACES for d in hand)


def test_roll_hand_is_sorted_and_reproducible():
    h1 = roll_hand(5, random.Random(42))
    h2 = roll_hand(5, random.Random(42))
    assert h1 == h2
    assert list(h1) == sorted(h1)


def test_roll_zero_dice_is_empty():
    assert roll_hand(0, random.Random(1)) == ()


def test_aces_are_wild_for_normal_faces():
    hand = (1, 1, 3, 5, 5)
    # two 5s plus two wild aces = 4
    assert count_face(hand, 5) == 4
    # one 3 plus two wild aces = 3
    assert count_face(hand, 3) == 3


def test_counting_aces_does_not_double_count():
    hand = (1, 1, 3, 5, 5)
    assert count_face(hand, ACE) == 2


def test_aces_wild_can_be_disabled():
    hand = (1, 1, 5, 5, 5)
    assert count_face(hand, 5, aces_wild=False) == 3


def test_count_face_rejects_bad_face():
    with pytest.raises(ValueError):
        count_face((1, 2, 3), 7)
