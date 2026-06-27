# Perudo

Play the dice game **Perudo** (Liar's Dice) against a bot that does proper
**Bayesian inference** on what's hidden under the cups — using the bidding
history as evidence, not just naive dice-counting. Can you beat the bot?

## Why this project

It's a compact showcase for probabilistic reasoning and inference: the
interesting problem isn't the rules, it's that an opponent's bid is a *signal*
about their hidden dice. The bot defines a likelihood `P(bid | hand)` and runs
Bayesian updates over the bidding history to sharpen its belief about the dice
on the table — and it can explain every step.

## Rules (core variant)

- Each player starts with **5 six-sided dice**, rolled hidden under a cup.
- A **bid** is a `(quantity, face)` claim, e.g. "four 5s", about the total count
  of that face across *all* dice on the table.
- **Ones (aces) are wild** — they count toward every face.
- Each raise must be strictly higher: more quantity (any face), or the same
  quantity at a higher face. (Standard ace bid-conversion ladder applies.)
- Instead of raising, a player may call **Dudo** to challenge the last bid. The
  dice are revealed and counted: if the bid holds, the challenger loses a die;
  otherwise the bidder does. A player at zero dice is out; last player standing
  wins.

This variant uses **Dudo only** — no Calza (exact call) and no Palifico round.

## Layout

```
perudo/
  dice.py         # hands, rolling, ace-aware counting
  bids.py         # Bid type, legality and ordering
  game.py         # game state, turn loop, Dudo resolution
  probability.py  # binomial dice-counting model
  inference.py    # Bayesian opponent-belief model (the showcase)
  agents.py       # Random / Counting / Bayesian / Human agents
  cli.py          # interactive terminal play
  simulate.py     # bot-vs-bot tournaments + calibration metrics
notebooks/
  perudo_inference.ipynb   # derivation, visualizations, calibration plots
```

## Install & run

```bash
pip install -e ".[dev]"     # runtime + pytest
pytest                      # run the test suite
python -m perudo.cli        # play against the bots (coming soon)
```

For the notebook extras: `pip install -e ".[notebook]"`, then open
`notebooks/perudo_inference.ipynb` for the full derivation and plots.

Watch the bots play and explain themselves:

```bash
python -m perudo.cli --watch --show-beliefs
python -m perudo.simulate          # win rates + calibration report
```

## Results

Both bots share the *same* decision policy; the only difference is that the
Bayesian bot evaluates bids against a posterior informed by the bidding history.
In self-play with rotated seating:

- **Bayesian beats the counting baseline ~73%** of head-to-head games.
- The bot stays **well-calibrated** — Brier score ≈ **0.125** (vs. 0.25 for an
  uninformative coin-flip) — with a mild, tunable overconfidence in the upper-mid
  probability range.

See `notebooks/perudo_inference.ipynb` for the model, the posterior-update demo,
and these plots.
