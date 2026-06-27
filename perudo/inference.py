"""The Bayesian opponent-belief model — the heart of the project.

The counting bot (:mod:`perudo.probability`) assumes every die it cannot see is
fair. That throws away real information: **an opponent's bid is a signal about
their hidden dice.** This module models that signal and does proper Bayesian
updates over it.

Model
-----
Each opponent's hand of ``m`` dice is summarised by its per-face count vector
``c = (c_1, ..., c_6)`` with ``sum(c) = m``. There are only ``C(m+5, 5)`` such
vectors (252 for ``m = 5``), so we enumerate them exactly.

* **Prior.** Dice are fair, so ``c ~ Multinomial(m, [1/6]*6)``.
* **Likelihood (noisy-rational bidder).** A player tends to bid faces they
  actually hold. Treat each bid of face ``f`` as soft evidence with likelihood
  ``P(bid f | c) ∝ exp(β · matches(c, f))``, where ``matches(c, f)`` is how many
  of face ``f`` the hand holds (aces wild) and ``β`` is a rationality knob
  (``β = 0`` ignores bids; larger ``β`` trusts them more).
* **Posterior.** Multiply prior by the likelihood of each observed bid and
  renormalise. Beliefs reset every round (dice are re-rolled).

From each opponent's posterior we read off the distribution of their
contribution to a face, sum opponents by convolution (a mean-field independence
approximation), add our own known dice, and get an *informed* distribution of
the table total — a sharper ``P(bid true)`` than the counting model.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import gammaln

from .agents import DUDO, Action, Agent, PlayerView, _best_opening
from .bids import Bid, legal_raises
from .dice import NUM_FACES, count_face
from .probability import at_least, bid_true_probability, convolve


@lru_cache(maxsize=None)
def _enumerate(m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(vectors, match_counts, prior)`` for a hand of ``m`` dice.

    * ``vectors[i]`` is a count vector ``(c_1..c_6)``.
    * ``match_counts[i, f-1]`` is how many dice match face ``f`` (aces wild).
    * ``prior[i]`` is the multinomial prior probability of that vector.

    Cached arrays are treated as read-only by callers.
    """

    def compositions(total: int, parts: int):
        if parts == 1:
            yield (total,)
            return
        for first in range(total + 1):
            for rest in compositions(total - first, parts - 1):
                yield (first, *rest)

    vectors = np.array(list(compositions(m, NUM_FACES)), dtype=np.int64)

    match_counts = np.empty_like(vectors)
    aces = vectors[:, 0]
    match_counts[:, 0] = aces  # the aces face counts only literal aces
    for f in range(1, NUM_FACES):
        match_counts[:, f] = vectors[:, f] + aces  # face f+1: itself plus wild aces

    log_prior = (
        gammaln(m + 1)
        - gammaln(vectors + 1).sum(axis=1)
        + m * np.log(1.0 / NUM_FACES)
    )
    prior = np.exp(log_prior)
    return vectors, match_counts, prior


class OpponentBelief:
    """Posterior over one opponent's hidden count vector, updated by their bids."""

    def __init__(self, num_dice: int, beta: float):
        self.m = num_dice
        self.beta = beta
        _, self._match_counts, prior = _enumerate(num_dice)
        self.weights = prior.copy()

    def update(self, bid_face: int) -> None:
        """Bayesian update from observing this opponent bid ``bid_face``."""
        self.weights = self.weights * np.exp(self.beta * self._match_counts[:, bid_face - 1])
        total = self.weights.sum()
        if total > 0:
            self.weights /= total

    def match_pmf(self, face: int) -> np.ndarray:
        """Posterior PMF of this opponent's contribution to ``face`` (length m+1)."""
        mc = self._match_counts[:, face - 1]
        return np.bincount(mc, weights=self.weights, minlength=self.m + 1)

    def expected_matches(self, face: int) -> float:
        """Posterior mean number of dice this opponent has matching ``face``."""
        mc = self._match_counts[:, face - 1]
        return float(np.dot(mc, self.weights))


class BayesianAgent(Agent):
    """A bot that reads the bidding history via Bayesian inference.

    It maintains an :class:`OpponentBelief` per live opponent, updates it on every
    bid, and evaluates bids against the *informed* table-total distribution rather
    than the naive binomial. Decision policy mirrors the baseline
    :class:`~perudo.agents.CountingAgent` so the only difference being measured is
    the inference.
    """

    name = "Bayesian"

    def __init__(
        self,
        name: str | None = None,
        *,
        beta: float = 1.0,
        dudo_threshold: float = 0.30,
        raise_confidence: float = 0.40,
    ):
        super().__init__(name)
        self.beta = beta
        self.dudo_threshold = dudo_threshold
        self.raise_confidence = raise_confidence
        self.beliefs: dict[int, OpponentBelief] = {}
        self.hand: tuple[int, ...] = ()
        #: Snapshot of expected table counts per face, for the CLI to display.
        self.last_belief: dict[int, float] = {}

    def reset(self) -> None:
        self.beliefs = {}

    def observe_round_start(self, view: PlayerView) -> None:
        self.hand = view.hand
        self.beliefs = {
            i: OpponentBelief(count, self.beta)
            for i, count in enumerate(view.dice_counts)
            if i != view.player and count > 0
        }

    def observe_bid(self, player: int, bid: Bid) -> None:
        belief = self.beliefs.get(player)
        if belief is not None:
            belief.update(bid.face)

    # --- probabilities ---------------------------------------------------

    def informed_bid_probability(self, bid: Bid) -> float:
        """``P(bid true)`` using the posterior over opponents' dice."""
        own = count_face(self.hand, bid.face)
        pmfs = [b.match_pmf(bid.face) for b in self.beliefs.values()]
        opp_total = convolve(*pmfs) if pmfs else np.array([1.0])
        return at_least(opp_total, bid.quantity - own)

    def naive_bid_probability(self, view: PlayerView, bid: Bid) -> float:
        """``P(bid true)`` under the counting model, for side-by-side rationale."""
        known = count_face(view.hand, bid.face)
        return bid_true_probability(bid, known, view.num_unknown)

    def _snapshot_belief(self) -> None:
        self.last_belief = {
            face: count_face(self.hand, face)
            + sum(b.expected_matches(face) for b in self.beliefs.values())
            for face in range(1, NUM_FACES + 1)
        }

    # --- policy ----------------------------------------------------------

    def decide(self, view: PlayerView) -> Action:
        self._snapshot_belief()
        if view.current_bid is None:
            bid = _best_opening(view)
            self.last_rationale = f"opening with what I hold: {bid}"
            return bid

        standing = view.current_bid
        p_inf = self.informed_bid_probability(standing)
        p_naive = self.naive_bid_probability(view, standing)

        if p_inf < self.dudo_threshold:
            self.last_rationale = (
                f"Dudo: {standing} is {p_inf:.0%} likely given the bidding "
                f"(naive {p_naive:.0%})"
            )
            return DUDO

        viable = [
            (self.informed_bid_probability(b), b)
            for b in legal_raises(standing, view.total_dice)
        ]
        viable = [(p, b) for p, b in viable if p >= self.raise_confidence]
        if not viable:
            self.last_rationale = (
                f"Dudo: no confident raise; {standing} at {p_inf:.0%} "
                f"(naive {p_naive:.0%})"
            )
            return DUDO

        p, bid = max(viable, key=lambda pb: (pb[0], pb[1].quantity, pb[1].face))
        self.last_rationale = f"raise to {bid} ({p:.0%} likely given the bidding)"
        return bid
