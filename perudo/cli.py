"""Interactive terminal Perudo.

Play against the bots, or ``--watch`` them play each other. The bots narrate
their reasoning — the Bayesian bot shows the probability it assigns to each bid
*given the bidding so far*, alongside the history-blind naive number — so you can
watch the inference at work.

Run with::

    python -m perudo.cli              # you vs a Bayesian and a Counter bot
    python -m perudo.cli --watch      # watch the bots play
    python -m perudo.cli --show-beliefs
"""

from __future__ import annotations

import argparse

from .agents import (
    Agent,
    CountingAgent,
    HumanAgent,
    RandomAgent,
    play_game,
)
from .game import DudoResult, GameState
from .inference import BayesianAgent


def _hand_str(hand) -> str:
    return "[" + " ".join(str(d) for d in hand) + "]" if hand else "[—]"


def format_round_start(gs: GameState, names: list[str], human_index: int | None) -> str:
    counts = ", ".join(
        f"{names[i]}:{c}" for i, c in enumerate(gs.dice_counts) if c > 0
    )
    lines = [f"\n── Round {gs.round_number} ── dice: {counts}"]
    if human_index is not None:
        lines.append(f"   Your hand: {_hand_str(gs.hands[human_index])}")
    return "\n".join(lines)


def format_bid(
    player: int,
    bid,
    agents: list[Agent],
    names: list[str],
    *,
    human_index: int | None,
    show_beliefs: bool,
) -> str:
    line = f"   {names[player]} bids {bid}"
    agent = agents[player]
    if player != human_index and agent.last_rationale:
        line += f"  — {agent.last_rationale}"
    if show_beliefs and isinstance(agent, BayesianAgent) and agent.last_belief:
        belief = ", ".join(
            f"{face}:{exp:.1f}" for face, exp in sorted(agent.last_belief.items())
        )
        line += f"\n      (expected on table — {belief})"
    return line


def format_dudo(result: DudoResult, names: list[str]) -> str:
    verdict = "holds" if result.bid_holds else "is wrong"
    reveal = " | ".join(
        f"{names[i]} {_hand_str(h)}" for i, h in enumerate(result.hands) if h
    )
    return (
        f"   {names[result.challenger]} calls DUDO on {names[result.bidder]}'s "
        f"{result.bid}!\n"
        f"   Reveal: {reveal}\n"
        f"   Actual {result.bid.face}s on table: {result.actual_count} — bid {verdict}. "
        f"{names[result.loser]} loses a die."
    )


def run(
    agents: list[Agent],
    names: list[str],
    *,
    human_index: int | None = None,
    seed: int | None = None,
    show_beliefs: bool = False,
    out=print,
) -> int:
    """Play one game start to finish, narrating events. Returns the winner index."""

    def on_event(event: str, payload) -> None:
        if event == "round_start":
            out(format_round_start(payload, names, human_index))
        elif event == "bid":
            player, bid = payload
            out(
                format_bid(
                    player,
                    bid,
                    agents,
                    names,
                    human_index=human_index,
                    show_beliefs=show_beliefs,
                )
            )
        elif event == "dudo":
            out(format_dudo(payload, names))

    out("Perudo — core rules, ones wild, Dudo only. Good luck!")
    winner = play_game(agents, seed=seed, on_event=on_event)
    out(f"\n🏆 {names[winner]} wins the game!")
    return winner


def build_opponents(n: int, opp_type: str, beta: float) -> list[Agent]:
    opponents: list[Agent] = []
    for i in range(n):
        if opp_type == "counter":
            opponents.append(CountingAgent(f"Counter{i + 1}"))
        elif opp_type == "random":
            opponents.append(RandomAgent(f"Random{i + 1}", seed=i))
        elif opp_type == "bayesian":
            opponents.append(BayesianAgent(f"Bayesian{i + 1}", beta=beta))
        else:  # mixed
            opponents.append(
                BayesianAgent(f"Bayesian{i + 1}", beta=beta)
                if i % 2 == 0
                else CountingAgent(f"Counter{i + 1}")
            )
    return opponents


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Play Perudo against Bayesian bots.")
    p.add_argument("--watch", action="store_true", help="bots only, no human player")
    p.add_argument("--opponents", type=int, default=2, help="number of bot opponents")
    p.add_argument(
        "--opp-type",
        choices=["mixed", "bayesian", "counter", "random"],
        default="mixed",
        help="kind of bot opponents",
    )
    p.add_argument("--beta", type=float, default=1.0, help="Bayesian rationality knob")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument(
        "--show-beliefs",
        action="store_true",
        help="show the Bayesian bot's expected counts each turn",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.watch:
        agents: list[Agent] = [
            BayesianAgent("Bayesian", beta=args.beta),
            CountingAgent("Counter"),
            RandomAgent("Random", seed=args.seed),
        ]
        human_index = None
    else:
        agents = [HumanAgent("You")]
        agents += build_opponents(args.opponents, args.opp_type, args.beta)
        human_index = 0

    names = [a.name for a in agents]
    try:
        run(
            agents,
            names,
            human_index=human_index,
            seed=args.seed,
            show_beliefs=args.show_beliefs,
        )
    except (KeyboardInterrupt, EOFError):
        print("\nGood game — bye!")


if __name__ == "__main__":
    main()
