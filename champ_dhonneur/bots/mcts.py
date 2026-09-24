"""IS-MCTS (Single-Observer Information Set Monte Carlo Tree Search).

À chaque itération, l'information cachée (main, sac et défausse cachée de
l'adversaire, ordre des pioches) est ré-échantillonnée, puis on descend dans un
arbre partagé indexé par les actions. La sélection utilise UCB1 corrigé par la
disponibilité des actions (Cowling, Powley & Whitehouse 2012). Les feuilles sont
évaluées par l'heuristique après une courte simulation.

C'est la base de référence ; la phase 2 remplacera l'heuristique par un réseau
valeur/politique entraîné par auto-jeu.
"""
from __future__ import annotations

import math
import time

from ..engine import Action, Game
from .base import Bot
from .heuristic import value


class Node:
    __slots__ = ("children", "n", "w", "avail", "team")

    def __init__(self, team: int):
        self.children: dict[Action, Node] = {}
        self.n = 0
        self.w = 0.0       # somme des valeurs du point de vue de `team`
        self.avail = 1
        self.team = team   # équipe qui a choisi l'action menant à ce nœud


class MCTSBot(Bot):
    name = "mcts"

    def __init__(self, iterations: int = 800, time_limit: float | None = None,
                 c: float = 0.3, rollout_depth: int = 0, seed: int | None = None):
        super().__init__(seed)
        self.iterations = iterations
        self.time_limit = time_limit
        self.c = c
        self.rollout_depth = rollout_depth
        self.last_stats: list[tuple[Action, int, float]] = []

    def choose(self, game: Game) -> Action:
        legal = game.legal_actions()
        if len(legal) == 1:
            return legal[0]
        me = game.to_move
        root = Node(game.team(me))
        deadline = time.time() + self.time_limit if self.time_limit else None
        it = 0
        while True:
            it += 1
            if deadline is not None:
                if time.time() > deadline:
                    break
            elif it > self.iterations:
                break
            self._iterate(game.determinize(me, self.rng), root)
        best = max(root.children.items(), key=lambda kv: kv[1].n)
        self.last_stats = sorted(((a, ch.n, ch.w / max(ch.n, 1)) for a, ch in root.children.items()),
                                 key=lambda t: -t[1])
        return best[0]

    def _iterate(self, g: Game, root: Node) -> None:
        node = root
        path = [root]
        while not g.done:
            legal = g.legal_actions()
            team = g.team(g.to_move)
            untried = [a for a in legal if a not in node.children]
            for a in legal:
                ch = node.children.get(a)
                if ch is not None:
                    ch.avail += 1
            if untried:
                a = self.rng.choice(untried)
                child = Node(team)
                node.children[a] = child
                g.apply(a)
                path.append(child)
                break
            log_c = self.c
            best_a, best_u = None, -math.inf
            for a in legal:
                ch = node.children[a]
                u = ch.w / ch.n + log_c * math.sqrt(math.log(ch.avail) / ch.n)
                if u > best_u:
                    best_a, best_u = a, u
            g.apply(best_a)
            node = node.children[best_a]
            path.append(node)
        # courte simulation aléatoire puis évaluation
        for _ in range(self.rollout_depth):
            if g.done:
                break
            g.apply(self.rng.choice(g.legal_actions()))
        v0 = value(g, 0)
        for n in path:
            n.n += 1
            n.w += v0 if n.team == 0 else -v0
