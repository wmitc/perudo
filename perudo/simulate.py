"""Bot-vs-bot tournaments and probability calibration.

Two kinds of evidence that the inference is worth its weight:

1. **Win rate.** :func:`tournament` / :func:`head_to_head` play many games with
   rotated seating (to cancel any first-mover edge) and report who wins.
2. **Calibration.** A good probabilistic player should be *calibrated*: bids it
   calls 70% likely should come true about 70% of the time.
   :func:`collect_calibration` logs each Bayesian bid's predicted truth
   probability and its realised outcome (read off the dice revealed at the next
   Dudo); :func:`reliability_curve` and :func:`brier_score` summarise it.

Run ``python -m perudo.simulate`` for a quick head-to-head and calibration report.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .agents import Agent, CountingAgent, RandomAgent, play_game
from .bids import Bid
from .dice import count_face
from .inference import BayesianAgent

AgentFactory = Callable[[], Agent]

#: One (predicted_probability, actually_true) pair.
CalibrationRecord = tuple[float, bool]


# --- tournaments -----------------------------------------------------------


def tournament(
    factories: list[AgentFactory],
    n_games: int,
    *,
    seed: int = 0,
    rotate: bool = True,
) -> list[int]:
    """Play ``n_games`` and return each factory's win count.

    Fresh agents are built per game (agents are stateful). With ``rotate`` the
    seating is rotated each game so no factory keeps a fixed turn position.
    """
    k = len(factories)
    wins = [0] * k
    for g in range(n_games):
        shift = g % k if rotate else 0
        order = [(i + shift) % k for i in range(k)]  # seat s plays factory order[s]
        agents = [factories[order[s]]() for s in range(k)]
        winning_seat = play_game(agents, seed=seed + g)
        wins[order[winning_seat]] += 1
    return wins


def win_rate(wins: list[int], index: int) -> float:
    total = sum(wins)
    return wins[index] / total if total else 0.0


def head_to_head(
    factory_a: AgentFactory,
    factory_b: AgentFactory,
    n_games: int,
    *,
    seed: int = 0,
) -> float:
    """Fraction of games won by ``factory_a`` against ``factory_b``."""
    wins = tournament([factory_a, factory_b], n_games, seed=seed)
    return win_rate(wins, 0)


# --- calibration -----------------------------------------------------------


class CalibratingBayesianAgent(BayesianAgent):
    """A Bayesian bot that logs each bid's predicted truth probability and outcome.

    The predicted probability is what the bot itself assigned to the bid it chose;
    the outcome is resolved from the hands revealed at the round's Dudo (dice do
    not change mid-round, so the reveal settles every bid made that round).
    """

    def __init__(self, log: list[CalibrationRecord], **kwargs):
        super().__init__(**kwargs)
        self._log = log
        self._pending: list[tuple[int, int, float]] = []  # (quantity, face, pred)

    def reset(self) -> None:
        super().reset()
        self._pending = []

    def observe_round_start(self, view) -> None:
        super().observe_round_start(view)
        self._pending = []

    def decide(self, view):
        action = super().decide(view)
        if isinstance(action, Bid):
            pred = self.informed_bid_probability(action)
            self._pending.append((action.quantity, action.face, pred))
        return action

    def observe_dudo(self, result) -> None:
        for quantity, face, pred in self._pending:
            actual = sum(count_face(hand, face) for hand in result.hands)
            self._log.append((pred, actual >= quantity))
        self._pending = []
        super().observe_dudo(result)


def collect_calibration(
    n_games: int,
    *,
    beta: float = 1.0,
    num_opponents: int = 2,
    seed: int = 0,
) -> list[CalibrationRecord]:
    """Play games and gather calibration records for a Bayesian bot."""
    log: list[CalibrationRecord] = []
    for g in range(n_games):
        agents: list[Agent] = [CalibratingBayesianAgent(log, beta=beta)]
        agents += [CountingAgent(f"C{i}") for i in range(num_opponents)]
        play_game(agents, seed=seed + g)
    return log


def brier_score(records: list[CalibrationRecord]) -> float:
    """Mean squared error of the predicted probabilities (lower is better)."""
    if not records:
        return float("nan")
    preds = np.array([p for p, _ in records])
    outcomes = np.array([float(o) for _, o in records])
    return float(np.mean((preds - outcomes) ** 2))


def reliability_curve(records: list[CalibrationRecord], n_bins: int = 10) -> list[dict]:
    """Bin predictions and compare mean prediction to empirical frequency.

    Returns one row per non-empty bin with ``mean_pred``, ``emp_freq`` and
    ``count`` — the data behind a reliability diagram (plotted in the notebook).
    """
    if not records:
        return []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    preds = np.array([p for p, _ in records])
    outcomes = np.array([float(o) for _, o in records])
    bins = np.clip(np.digitize(preds, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "mean_pred": float(preds[mask].mean()),
                "emp_freq": float(outcomes[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return rows


# --- demo ------------------------------------------------------------------


def main() -> None:  # pragma: no cover - reporting convenience
    n = 300
    rate = head_to_head(
        lambda: BayesianAgent("Bayesian"), lambda: CountingAgent("Counter"), n
    )
    print(f"Bayesian vs Counting over {n} games: Bayesian wins {rate:.1%}")

    records = collect_calibration(150)
    print(f"\nCalibration over {len(records)} logged bids:")
    print(f"  Brier score: {brier_score(records):.3f} (lower is better)")
    print("  reliability (mean_pred -> empirical_freq, count):")
    for row in reliability_curve(records):
        print(
            f"    {row['mean_pred']:.2f} -> {row['emp_freq']:.2f}  (n={row['count']})"
        )


if __name__ == "__main__":
    main()
