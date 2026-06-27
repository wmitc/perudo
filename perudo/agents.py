"""Agents and the game driver.

Every agent implements one method, :meth:`Agent.decide`, returning either a
:class:`~perudo.bids.Bid` (a raise) or the :data:`DUDO` sentinel (a challenge).
Agents only ever see a :class:`PlayerView` — their own hand plus public
information — never an opponent's dice.

Stateful agents (the Bayesian bot in :mod:`perudo.inference`) also receive
observation callbacks so they can update beliefs as the round unfolds. The base
class makes those no-ops, so simple agents ignore them.

This module also hosts :func:`play_game`, the loop that wires agents to a
:class:`~perudo.game.GameState`; the simulator and CLI both build on it.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .bids import Bid, legal_raises
from .dice import ACE, DICE_PER_PLAYER, NUM_FACES, Hand, count_face
from .game import DudoResult, GameState
from .probability import bid_true_probability


class _Dudo:
    """Singleton action: challenge the standing bid."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "DUDO"


#: The action meaning "call Dudo on the standing bid".
DUDO = _Dudo()

#: An agent's move: a raise, or a challenge.
Action = Bid | _Dudo


@dataclass(frozen=True)
class PlayerView:
    """Everything an agent is allowed to know on its turn."""

    player: int
    hand: Hand
    dice_counts: tuple[int, ...]
    current_bid: Bid | None
    current_player: int
    bid_history: tuple[tuple[int, Bid], ...]

    @property
    def total_dice(self) -> int:
        return sum(self.dice_counts)

    @property
    def num_unknown(self) -> int:
        """Dice not in my own hand."""
        return self.total_dice - len(self.hand)

    @classmethod
    def from_state(cls, gs: GameState, player: int) -> PlayerView:
        return cls(
            player=player,
            hand=gs.hands[player],
            dice_counts=tuple(gs.dice_counts),
            current_bid=gs.current_bid,
            current_player=gs.current_player,
            bid_history=tuple(gs.bid_history),
        )


class Agent(ABC):
    """Base agent. Subclasses implement :meth:`decide`; observation hooks are optional."""

    name: str = "agent"

    def __init__(self, name: str | None = None):
        if name is not None:
            self.name = name
        #: Set by deciding agents; the CLI shows it to explain a move.
        self.last_rationale: str = ""

    @abstractmethod
    def decide(self, view: PlayerView) -> Action:
        """Choose a raise or DUDO given the current view."""

    # --- observation hooks (no-ops by default) ---------------------------

    def reset(self) -> None:
        """Called once at the start of a game."""

    def observe_round_start(self, view: PlayerView) -> None:
        """Called for every player at the start of each round."""

    def observe_bid(self, player: int, bid: Bid) -> None:
        """Called for every agent whenever any player makes a bid."""

    def observe_dudo(self, result: DudoResult) -> None:
        """Called for every agent when a Dudo is resolved."""


def _best_opening(view: PlayerView) -> Bid:
    """A safe honest opening: claim the face we hold the most of."""
    counts = {face: count_face(view.hand, face) for face in range(2, NUM_FACES + 1)}
    best_face = max(counts, key=lambda f: counts[f])
    quantity = max(1, counts[best_face])
    return Bid(quantity, best_face)


class RandomAgent(Agent):
    """Bids and challenges at random; a baseline opponent and a sanity check."""

    name = "Random"

    def __init__(self, name: str | None = None, *, dudo_prob: float = 0.2, seed: int | None = None):
        super().__init__(name)
        self.dudo_prob = dudo_prob
        self.rng = random.Random(seed)

    def decide(self, view: PlayerView) -> Action:
        if view.current_bid is None:
            bid = self.rng.choice(legal_raises(None, view.total_dice))
            self.last_rationale = "opening at random"
            return bid
        raises = legal_raises(view.current_bid, view.total_dice)
        if not raises or self.rng.random() < self.dudo_prob:
            self.last_rationale = "challenging at random"
            return DUDO
        self.last_rationale = "raising at random"
        return self.rng.choice(raises)


class CountingAgent(Agent):
    """The naive baseline bot.

    Uses only the binomial counting model (:mod:`perudo.probability`): it knows
    its own dice and treats every other die as fair. It ignores what the bidding
    history implies — that is exactly the gap the Bayesian bot closes.
    """

    name = "Counter"

    def __init__(
        self,
        name: str | None = None,
        *,
        dudo_threshold: float = 0.30,
        raise_confidence: float = 0.40,
    ):
        super().__init__(name)
        #: Challenge if the standing bid's truth probability is below this.
        self.dudo_threshold = dudo_threshold
        #: Only make raises at least this likely to be true.
        self.raise_confidence = raise_confidence

    def bid_probability(self, view: PlayerView, bid: Bid) -> float:
        known = count_face(view.hand, bid.face)
        return bid_true_probability(bid, known, view.num_unknown)

    def decide(self, view: PlayerView) -> Action:
        if view.current_bid is None:
            bid = _best_opening(view)
            self.last_rationale = f"opening with what I hold: {bid}"
            return bid

        p_standing = self.bid_probability(view, view.current_bid)
        if p_standing < self.dudo_threshold:
            self.last_rationale = (
                f"Dudo: {view.current_bid} is only {p_standing:.0%} likely"
            )
            return DUDO

        viable = [
            (self.bid_probability(view, b), b)
            for b in legal_raises(view.current_bid, view.total_dice)
        ]
        viable = [(p, b) for p, b in viable if p >= self.raise_confidence]
        if not viable:
            self.last_rationale = (
                f"Dudo: no confident raise; {view.current_bid} at {p_standing:.0%}"
            )
            return DUDO

        # Prefer the most likely-true raise, breaking ties toward more pressure.
        p, bid = max(viable, key=lambda pb: (pb[0], pb[1].quantity, pb[1].face))
        self.last_rationale = f"raise to {bid} ({p:.0%} likely true)"
        return bid


class HumanAgent(Agent):
    """A human at the keyboard. Input/output are injected so the CLI can wire stdin.

    Commands: ``d`` / ``dudo`` to challenge, or ``<quantity> <face>`` to bid.
    """

    name = "You"

    def __init__(self, name: str | None = None, *, input_fn=input, output_fn=print):
        super().__init__(name)
        self._input = input_fn
        self._output = output_fn

    def decide(self, view: PlayerView) -> Action:
        while True:
            raw = self._input("Your move ('d' to Dudo, or 'qty face'): ").strip().lower()
            if raw in ("d", "dudo"):
                if view.current_bid is None:
                    self._output("Nothing to challenge yet — make a bid.")
                    continue
                return DUDO
            parts = raw.split()
            if len(parts) == 2 and all(p.lstrip("-").isdigit() for p in parts):
                quantity, face = int(parts[0]), int(parts[1])
                try:
                    bid = Bid(quantity, face)
                except ValueError as exc:
                    self._output(f"Invalid bid: {exc}")
                    continue
                from .bids import is_legal_raise

                if bid.quantity > view.total_dice:
                    self._output(f"Only {view.total_dice} dice are in play.")
                    continue
                if not is_legal_raise(view.current_bid, bid):
                    self._output(f"{bid} is not a legal raise over {view.current_bid}.")
                    continue
                return bid
            self._output("Didn't understand. Use 'd' or two numbers like '3 5'.")


def play_game(
    agents: list[Agent],
    *,
    seed: int | None = None,
    dice_per_player: int = DICE_PER_PLAYER,
    on_event=None,
    max_rounds: int = 100_000,
) -> int:
    """Play a full game to completion and return the winning agent's index.

    ``on_event`` (optional) receives ``(event_name, payload)`` tuples for a CLI
    or logger: ``"round_start"``, ``"bid"`` (``(player, bid)``), and ``"dudo"``
    (a :class:`DudoResult`).
    """
    n = len(agents)
    gs = GameState(num_players=n, dice_per_player=dice_per_player, seed=seed)
    for agent in agents:
        agent.reset()
    starter = gs.rng.randrange(n)

    for _ in range(max_rounds):
        if gs.is_over():
            break
        gs.start_round(starter)
        for i, agent in enumerate(agents):
            agent.observe_round_start(PlayerView.from_state(gs, i))
        if on_event:
            on_event("round_start", gs)

        while True:
            player = gs.current_player
            action = agents[player].decide(PlayerView.from_state(gs, player))
            if isinstance(action, _Dudo):
                result = gs.apply_dudo(player)
                for agent in agents:
                    agent.observe_dudo(result)
                if on_event:
                    on_event("dudo", result)
                starter = gs.next_starting_player(result.loser)
                break
            gs.apply_bid(player, action)
            for agent in agents:
                agent.observe_bid(player, action)
            if on_event:
                on_event("bid", (player, action))

    winner = gs.winner()
    if winner is None:  # pragma: no cover - guard against pathological non-termination
        raise RuntimeError("game did not terminate in max_rounds")
    return winner
