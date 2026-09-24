"""Évaluation heuristique d'une position et bot glouton."""
from __future__ import annotations

import math

from ..engine import ATTACK, CONTROL, Action, Game
from ..units import UNITS
from .base import Bot

W_MARKER = 40.0     # Lieu contrôlé
W_UNIT = 6.0        # unité présente sur le plateau
W_COIN = 2.5        # pièce supplémentaire dans une pile
W_ARMY = 1.2        # pièce dans le sac / la main / la défausse
W_RESERVE = 0.3     # pièce encore recrutable
W_ON_LOC = 9.0      # unité sur un Lieu qu'elle peut contrôler
W_NEAR_LOC = 2.5    # unité adjacente à un Lieu non contrôlé
W_THREAT = 3.0      # unité ennemie adjacente attaquable


def score(game: Game, team: int) -> float:
    """Score positif = avantage pour `team`."""
    if game.done:
        if game.winner is None:
            return 0.0
        return 1e4 if game.winner == team else -1e4
    s = 0.0
    total = 6 if game.n_players == 2 else 8
    for t in range(2):
        sign = 1 if t == team else -1
        placed = total - game.markers_left[t]
        s += sign * W_MARKER * placed
        # proche de la victoire : bonus non linéaire
        if game.markers_left[t] == 1:
            s += sign * 60
    sp = game.spec
    for pos, u in game.board.items():
        t = game.team(u.owner)
        sign = 1 if t == team else -1
        s += sign * (W_UNIT + W_COIN * (u.coins - 1))
        if pos in game.loc_set and game.control[pos] != t:
            s += sign * W_ON_LOC
        for nb in sp.neighbors[pos]:
            if nb in game.loc_set and game.control[nb] != t and nb not in game.board:
                s += sign * W_NEAR_LOC
            v = game.board.get(nb)
            if v is not None and game.team(v.owner) != t and UNITS[u.utype].normal_attack:
                if not (v.utype == "N" and u.coins < 2):
                    s += sign * W_THREAT / max(v.coins, 1)
    for pl in game.players:
        sign = 1 if pl.team == team else -1
        s += sign * W_ARMY * pl.total_coins()
        s += sign * W_RESERVE * sum(pl.reserve.values())
    return s


def value(game: Game, team: int) -> float:
    """Valeur dans [-1, 1] pour `team`."""
    if game.done:
        return 0.0 if game.winner is None else (1.0 if game.winner == team else -1.0)
    return math.tanh(score(game, team) / 80.0)


class GreedyBot(Bot):
    """Joue l'action qui maximise l'heuristique après application (1 coup)."""
    name = "glouton"

    def choose(self, game: Game) -> Action:
        if game.in_draft:   # l'heuristique ne sait pas évaluer une composition d'armée
            return self.rng.choice(game.legal_actions())
        me = game.to_move
        team = game.team(me)
        view = game.determinize(me, self.rng)
        best, best_s = None, -math.inf
        for a in view.legal_actions():
            g = view.copy()
            g.apply(a)
            s = score(g, team) + self.rng.random() * 0.5
            if a.kind in (ATTACK, CONTROL):
                s += 0.5
            if s > best_s:
                best, best_s = a, s
        return best
