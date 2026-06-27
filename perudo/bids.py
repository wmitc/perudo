"""Bids and the legality rules that order them.

A :class:`Bid` is a ``(quantity, face)`` claim about the total count of ``face``
across *every* die on the table (aces wild). The only non-obvious part is the
**ace conversion ladder**, which keeps aces "worth double" because they are wild:

* Normal raise: more quantity (any face), or same quantity at a higher face.
* Switch *to* aces: quantity may halve — need ``ceil(prev_quantity / 2)`` aces.
* Switch *from* aces to a normal face: need ``2 * prev_quantity + 1``.
* The opening bid of a round may not be aces.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dice import ACE, NUM_FACES


@dataclass(frozen=True, order=True)
class Bid:
    """A claim that at least ``quantity`` dice on the table show ``face``."""

    quantity: int
    face: int

    def __post_init__(self) -> None:
        if self.quantity < 1:
            raise ValueError(f"bid quantity must be >= 1, got {self.quantity}")
        if not 1 <= self.face <= NUM_FACES:
            raise ValueError(f"bid face must be in 1..{NUM_FACES}, got {self.face}")

    @property
    def is_ace(self) -> bool:
        return self.face == ACE

    def __str__(self) -> str:
        suffix = " (aces, wild)" if self.is_ace else ""
        return f"{self.quantity}×{self.face}{suffix}"


def is_legal_raise(prev: Bid | None, new: Bid) -> bool:
    """Whether ``new`` is a legal bid following ``prev``.

    ``prev is None`` means ``new`` opens the round (any non-ace face).
    """
    if prev is None:
        return not new.is_ace  # cannot open the bidding with aces

    if prev.is_ace and new.is_ace:
        return new.quantity > prev.quantity
    if new.is_ace:  # switching to aces
        return new.quantity >= (prev.quantity + 1) // 2
    if prev.is_ace:  # switching from aces back to a normal face
        return new.quantity >= 2 * prev.quantity + 1

    # both normal faces
    if new.quantity > prev.quantity:
        return True
    return new.quantity == prev.quantity and new.face > prev.face


def legal_raises(prev: Bid | None, max_quantity: int) -> list[Bid]:
    """All legal bids following ``prev`` with quantity up to ``max_quantity``.

    ``max_quantity`` is normally the total number of dice on the table — no bid
    above that can ever be true. Used by agents to enumerate candidate moves.
    """
    out = [
        bid
        for face in range(1, NUM_FACES + 1)
        for q in range(1, max_quantity + 1)
        if is_legal_raise(prev, bid := Bid(q, face))
    ]
    return out
