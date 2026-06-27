"""The baseline dice-counting probability model.

This is the *naive* model: every die you cannot see is an independent fair die,
so it matches a given face with a fixed probability — ``2/6`` for a normal face
(the face itself or a wild ace) and ``1/6`` for aces. Under that assumption the
unknown matches are Binomial, and the truth of a bid is a tail probability.

It deliberately ignores the bidding history. The Bayesian model in
:mod:`perudo.inference` reuses the small building blocks here
(:func:`binomial_pmf`, :func:`convolve`, :func:`at_least`) but replaces the
"every unknown die is fair" assumption with a posterior informed by the bids.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom

from .bids import Bid
from .dice import ACE, NUM_FACES


def match_probability(face: int) -> float:
    """Probability a single unseen fair die matches ``face`` (aces wild).

    ``1/6`` for the aces face itself; ``2/6`` for any other face (that face plus
    the wild ace).
    """
    return 1.0 / NUM_FACES if face == ACE else 2.0 / NUM_FACES


def binomial_pmf(n: int, p: float) -> np.ndarray:
    """PMF of ``Binomial(n, p)`` as an array indexed by count ``0..n``."""
    k = np.arange(n + 1)
    return binom.pmf(k, n, p)


def unknown_match_pmf(num_unknown: int, face: int) -> np.ndarray:
    """Distribution of how many of ``num_unknown`` unseen dice match ``face``."""
    return binomial_pmf(num_unknown, match_probability(face))


def convolve(*pmfs: np.ndarray) -> np.ndarray:
    """Distribution of the sum of independent count variables.

    Convolving the per-source PMFs gives the PMF of their total — used both for
    "known + unknown" totals here and for summing opponent beliefs in inference.
    """
    result = np.array([1.0])
    for pmf in pmfs:
        result = np.convolve(result, pmf)
    return result


def at_least(pmf: np.ndarray, k: int) -> float:
    """``P(X >= k)`` for a count variable with the given PMF."""
    if k <= 0:
        return 1.0
    if k >= len(pmf):
        return 0.0
    return float(pmf[k:].sum())


def bid_true_probability(bid: Bid, known_matches: int, num_unknown: int) -> float:
    """``P(bid holds)`` given your own ``known_matches`` and ``num_unknown`` dice.

    Equivalent to ``P(known + Binomial(num_unknown, p) >= bid.quantity)`` with
    ``p = match_probability(bid.face)``.
    """
    need = bid.quantity - known_matches
    if need <= 0:
        return 1.0
    if need > num_unknown:
        return 0.0
    p = match_probability(bid.face)
    return float(binom.sf(need - 1, num_unknown, p))
