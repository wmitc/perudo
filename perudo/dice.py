"""Dice and hands for Perudo.

A *hand* is just an immutable tuple of die faces (ints in ``1..6``). The one rule
that lives here is **ones (aces) are wild**: when counting a non-ace face, aces
count toward it too. Keeping that rule in :func:`count_face` means every other
module (probability, inference, game resolution) counts dice the same way.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

#: Number of distinct faces on a die.
NUM_FACES = 6
#: The wild face. Aces (ones) count toward every other face.
ACE = 1
#: Standard starting dice per player in core Perudo.
DICE_PER_PLAYER = 5

#: A hand is an immutable tuple of die faces.
Hand = tuple[int, ...]


def roll_hand(n: int, rng: random.Random | None = None) -> Hand:
    """Roll ``n`` dice and return them as a sorted hand.

    Sorting makes hands canonical (order never matters in Perudo) and keeps test
    output stable. Pass a seeded :class:`random.Random` for reproducibility.
    """
    if n < 0:
        raise ValueError(f"cannot roll a negative number of dice: {n}")
    rng = rng or random
    return tuple(sorted(rng.randint(1, NUM_FACES) for _ in range(n)))


def count_face(hand: Iterable[int], face: int, *, aces_wild: bool = True) -> int:
    """Count dice in ``hand`` matching ``face``.

    With ``aces_wild`` (the default, and the core rule), aces also count toward
    any non-ace face. Counting the aces face itself (``face == ACE``) only ever
    counts literal aces.
    """
    if not 1 <= face <= NUM_FACES:
        raise ValueError(f"face must be in 1..{NUM_FACES}, got {face}")
    if face == ACE or not aces_wild:
        return sum(1 for d in hand if d == face)
    return sum(1 for d in hand if d == face or d == ACE)
