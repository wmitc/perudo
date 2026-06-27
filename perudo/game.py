"""Core Perudo game state and rules (Dudo-only variant).

:class:`GameState` owns the dice, the current bid, and the bidding history, and
exposes primitive moves — :meth:`apply_bid` and :meth:`apply_dudo` — plus the
round bookkeeping a driver (CLI or simulator) needs. It deliberately does *not*
know about agents; it just enforces the rules.

Rules implemented: 5 dice per player to start, ones wild, raise-or-Dudo turns.
No Calza and no Palifico.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .bids import Bid, is_legal_raise
from .dice import DICE_PER_PLAYER, Hand, count_face, roll_hand


@dataclass(frozen=True)
class DudoResult:
    """Outcome of a Dudo challenge, with everything a driver needs to narrate it."""

    challenger: int
    bidder: int
    bid: Bid
    actual_count: int
    bid_holds: bool  # True if the table actually had >= bid.quantity
    loser: int
    hands: tuple[Hand, ...]  # snapshot of all hands at reveal time


@dataclass
class GameState:
    """Mutable state of a single Perudo game across many rounds."""

    num_players: int
    dice_per_player: int = DICE_PER_PLAYER
    seed: int | None = None

    dice_counts: list[int] = field(init=False)
    hands: list[Hand] = field(init=False)
    current_player: int = field(init=False, default=0)
    current_bid: Bid | None = field(init=False, default=None)
    last_bidder: int | None = field(init=False, default=None)
    bid_history: list[tuple[int, Bid]] = field(init=False, default_factory=list)
    round_number: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.num_players < 2:
            raise ValueError("need at least 2 players")
        self.rng = random.Random(self.seed)
        self.dice_counts = [self.dice_per_player] * self.num_players
        self.hands = [() for _ in range(self.num_players)]

    # --- queries ---------------------------------------------------------

    def total_dice(self) -> int:
        return sum(self.dice_counts)

    def active_players(self) -> list[int]:
        """Indices of players who still have dice."""
        return [i for i, c in enumerate(self.dice_counts) if c > 0]

    def is_over(self) -> bool:
        return len(self.active_players()) <= 1

    def winner(self) -> int | None:
        active = self.active_players()
        return active[0] if self.is_over() and active else None

    def next_active_player(self, idx: int) -> int:
        """The next player with dice, going clockwise from ``idx`` (exclusive)."""
        for step in range(1, self.num_players + 1):
            cand = (idx + step) % self.num_players
            if self.dice_counts[cand] > 0:
                return cand
        raise RuntimeError("no active players remain")

    def count_face_total(self, face: int) -> int:
        """Total dice across all hands matching ``face`` (aces wild)."""
        return sum(count_face(hand, face) for hand in self.hands)

    def can_dudo(self) -> bool:
        return self.current_bid is not None

    # --- round / move transitions ---------------------------------------

    def start_round(self, starting_player: int) -> None:
        """Roll fresh hidden hands and reset the bidding for a new round."""
        if self.dice_counts[starting_player] <= 0:
            raise ValueError(f"starting player {starting_player} has no dice")
        self.round_number += 1
        self.hands = [roll_hand(c, self.rng) for c in self.dice_counts]
        self.current_player = starting_player
        self.current_bid = None
        self.last_bidder = None
        self.bid_history = []

    def apply_bid(self, player: int, bid: Bid) -> None:
        """Record a raise by ``player`` and advance the turn."""
        if player != self.current_player:
            raise ValueError(f"not player {player}'s turn (it is {self.current_player}'s)")
        max_q = self.total_dice()
        if bid.quantity > max_q:
            raise ValueError(f"bid quantity {bid.quantity} exceeds {max_q} dice in play")
        if not is_legal_raise(self.current_bid, bid):
            raise ValueError(f"illegal raise {bid} over {self.current_bid}")
        self.bid_history.append((player, bid))
        self.current_bid = bid
        self.last_bidder = player
        self.current_player = self.next_active_player(player)

    def apply_dudo(self, challenger: int) -> DudoResult:
        """Resolve a Dudo by ``challenger`` against the standing bid.

        If the table actually holds at least the bid quantity, the bid holds and
        the challenger loses a die; otherwise the bidder does. The round's bid is
        cleared. Caller picks the next round's starter via
        :meth:`next_starting_player`.
        """
        if challenger != self.current_player:
            raise ValueError(f"not player {challenger}'s turn to act")
        if self.current_bid is None or self.last_bidder is None:
            raise ValueError("cannot call Dudo with no standing bid")

        bid = self.current_bid
        bidder = self.last_bidder
        actual = self.count_face_total(bid.face)
        bid_holds = actual >= bid.quantity
        loser = challenger if bid_holds else bidder

        self.dice_counts[loser] -= 1
        result = DudoResult(
            challenger=challenger,
            bidder=bidder,
            bid=bid,
            actual_count=actual,
            bid_holds=bid_holds,
            loser=loser,
            hands=tuple(self.hands),
        )
        self.current_bid = None
        self.last_bidder = None
        return result

    def next_starting_player(self, loser: int) -> int:
        """Who starts the next round: the loser if still in, else the next active."""
        if self.dice_counts[loser] > 0:
            return loser
        return self.next_active_player(loser)
