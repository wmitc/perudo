"""Tests for the CLI narration and the bot-only game runner."""

from perudo.agents import CountingAgent, RandomAgent
from perudo.bids import Bid
from perudo.cli import (
    build_opponents,
    build_parser,
    format_bid,
    format_dudo,
    run,
)
from perudo.game import DudoResult
from perudo.inference import BayesianAgent


def test_format_bid_shows_bot_rationale():
    bot = CountingAgent("Counter")
    bot.last_rationale = "raise to 3×5 (60% likely true)"
    line = format_bid(
        0, Bid(3, 5), [bot], ["Counter"], human_index=None, show_beliefs=False
    )
    assert "Counter bids 3×5" in line
    assert "raise to 3×5" in line


def test_format_bid_hides_rationale_for_human():
    human = CountingAgent("You")  # stand-in; human_index suppresses rationale
    human.last_rationale = "secret"
    line = format_bid(
        0, Bid(2, 4), [human], ["You"], human_index=0, show_beliefs=False
    )
    assert "secret" not in line


def test_format_bid_show_beliefs_for_bayesian():
    bot = BayesianAgent("Bayesian")
    bot.last_belief = {1: 1.0, 2: 1.5, 3: 1.0, 4: 1.0, 5: 2.5, 6: 1.0}
    bot.last_rationale = "raise"
    line = format_bid(
        0, Bid(3, 5), [bot], ["Bayesian"], human_index=None, show_beliefs=True
    )
    assert "expected on table" in line
    assert "5:2.5" in line


def test_format_dudo_reports_outcome():
    result = DudoResult(
        challenger=1,
        bidder=0,
        bid=Bid(4, 5),
        actual_count=2,
        bid_holds=False,
        loser=0,
        hands=((5, 2, 3), (4, 6, 6)),
    )
    text = format_dudo(result, ["A", "B"])
    assert "B calls DUDO on A" in text
    assert "is wrong" in text
    assert "A loses a die" in text


def test_run_bot_game_announces_winner():
    agents = [
        BayesianAgent("Bayesian"),
        CountingAgent("Counter"),
        RandomAgent("Random", seed=1),
    ]
    names = [a.name for a in agents]
    lines: list[str] = []
    winner = run(agents, names, seed=3, out=lines.append)
    assert 0 <= winner < 3
    assert any("wins the game" in ln for ln in lines)
    assert any("Round 1" in ln for ln in lines)


def test_build_opponents_mixed():
    opps = build_opponents(2, "mixed", beta=1.0)
    assert isinstance(opps[0], BayesianAgent)
    assert isinstance(opps[1], CountingAgent)


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.opponents == 2
    assert args.opp_type == "mixed"
    assert not args.watch
