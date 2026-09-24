"""Tournois bot contre bot (couleurs alternées), avec relevés optionnels."""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

from .bots import make_bot
from .engine import Game
from .notation import export_record


def play_game(bot_specs: list[str], seed: int, mode: str = "2J", units=None,
              max_rounds: int = 150) -> Game:
    g = Game(mode, units, seed=seed, max_rounds=max_rounds)
    bots = [make_bot(s, seed=seed * 10 + i) for i, s in enumerate(bot_specs)]
    while not g.done:
        g.apply(bots[g.to_move].choose(g))
    return g


def match(a: str, b: str, games: int = 20, mode: str = "2J", seed: int = 0,
          save_dir: str | None = None, verbose: bool = True) -> Counter:
    score: Counter = Counter()
    t0 = time.time()
    for i in range(games):
        swap = i % 2 == 1
        specs = [b, a] if swap else [a, b]
        if mode == "4J":
            specs = specs * 2
        g = play_game(specs, seed + i, mode)
        if g.winner is None:
            score["nul"] += 1
        else:
            winner_is_a = (g.winner == 0) != swap
            score[a if winner_is_a else b] += 1
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            Path(save_dir, f"partie_{seed + i:05d}.nch").write_text(
                export_record(g, headers={"Blanc": specs[0], "Noir": specs[1]}), encoding="utf-8")
        if verbose:
            print(f"partie {i + 1}/{games} : {dict(score)}  ({time.time() - t0:.0f}s)", flush=True)
    return score
