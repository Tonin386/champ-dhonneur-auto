"""Interface commune des bots."""
from __future__ import annotations

import random

from ..engine import Action, Game


class Bot:
    name = "bot"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose(self, game: Game) -> Action:
        """Choisit une action pour game.to_move.

        Le bot ne doit utiliser que l'information de son camp : les bots
        « honnêtes » passent par game.determinize(joueur).
        """
        raise NotImplementedError


class RandomBot(Bot):
    name = "aleatoire"

    def choose(self, game: Game) -> Action:
        return self.rng.choice(game.legal_actions())
