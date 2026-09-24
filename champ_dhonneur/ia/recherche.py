"""Recherche arborescente Gumbel sur ensembles d'information (Gumbel IS-MCTS).

Combine deux idées récentes :

* **Gumbel AlphaZero / MuZero** (Danihelka et al., ICLR 2022) : à la racine,
  on tire des actions sans remise avec l'astuce Gumbel-Top-k, puis on répartit
  le budget par *halving séquentiel*. La cible d'apprentissage est la
  politique améliorée π' = softmax(logits + σ(Q complété)). Cette méthode
  garantit une amélioration de la politique même avec très peu de
  simulations (16 à 64), ce qui est crucial avec un moteur Python.
* **IS-MCTS** (Cowling et al. 2012) : chaque simulation tire une
  *déterminisation* (main, sac et défausse cachée de l'adversaire, ordre des
  pioches) compatible avec ce que sait le joueur, puis descend dans un arbre
  partagé indexé par les actions. Sous la racine, la sélection est PUCT avec
  a priori du réseau renormalisés sur les actions disponibles.

Une simulation s'arrête dès qu'elle atteint une position dont les actions
légales n'ont pas toutes été évaluées : le réseau est alors interrogé (une
évaluation par simulation, comme AlphaZero).

Protocole : la recherche est un générateur qui produit des *listes* de
requêtes (partie, actions légales) et reçoit les listes de réponses
(logits, valeur). Un pilote peut ainsi regrouper les requêtes de centaines de
parties en un seul lot GPU. Les simulations d'une même recherche peuvent être
lancées par vagues parallèles grâce à la perte virtuelle.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from ..engine import Action, Game


@dataclass
class ParamsRecherche:
    simulations: int = 64
    m: int = 16              # actions candidates à la racine (Gumbel-Top-m)
    c_visit: float = 50.0
    c_scale: float = 0.1     # valeurs de référence de mctx (DeepMind)
    normaliser_q: bool = True   # Q complétés ramenés dans [0, 1] par min-max (comme mctx)
    c_puct: float = 1.25
    fpu: float = 0.25        # réduction « first play urgency »
    bruit: bool = True       # bruit de Gumbel (exploration en auto-jeu)
    parallele: int = 1       # simulations par vague (perte virtuelle)
    perte_virtuelle: float = 1.0


class Noeud:
    __slots__ = ("enfants", "logits", "n", "w0", "equipe")

    def __init__(self, equipe: int):
        self.enfants: dict[Action, Noeud] = {}
        self.logits: dict[Action, float] | None = None
        self.n = 0.0
        self.w0 = 0.0          # somme des valeurs du point de vue de l'équipe 0
        self.equipe = equipe   # équipe qui a choisi l'action menant ici


@dataclass
class Resultat:
    action: Action
    legal: list[Action]
    politique: np.ndarray    # cible π'
    visites: np.ndarray
    q: np.ndarray            # Q complété (point de vue du joueur), dans [-1, 1]
    logits: np.ndarray
    v_reseau: float
    valeur: float            # valeur moyenne à la racine (point de vue du joueur)
    simulations: int = 0
    infos: dict = field(default_factory=dict)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class RechercheGumbel:
    def __init__(self, params: ParamsRecherche | None = None, seed: int | None = None):
        self.p = params or ParamsRecherche()
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- racine
    def generateur(self, game: Game, simulations: int | None = None):
        P = self.p
        n_sims = P.simulations if simulations is None else simulations
        me = game.to_move
        team = game.team(me)
        sign = 1.0 if team == 0 else -1.0
        legal = game.legal_actions()
        [(logits, v_hat)] = yield [(game, legal)]
        logits = np.asarray(logits, np.float64)
        k = len(legal)
        racine = Noeud(1 - team)
        racine.logits = {a: float(l) for a, l in zip(legal, logits)}
        racine.n, racine.w0 = 1.0, sign * v_hat
        for a in legal:
            racine.enfants[a] = Noeud(team)
        gumbel = self.np_rng.gumbel(size=k) if P.bruit else np.zeros(k)
        used = 0
        if k > 1 and n_sims > 0:
            m = min(P.m, k, n_sims)
            cand = list(np.argsort(-(gumbel + logits))[:m])
            n_phases = max(1, math.ceil(math.log2(m)))
            while True:
                if len(cand) <= 2:
                    per = math.ceil((n_sims - used) / len(cand))
                else:
                    per = max(1, n_sims // (n_phases * len(cand)))
                taches = [legal[i] for _ in range(per) for i in cand][: n_sims - used]
                for s in range(0, len(taches), P.parallele):
                    yield from self._vague(game, me, racine, taches[s:s + P.parallele])
                used += len(taches)
                if len(cand) == 1 or used >= n_sims:
                    break
                q = self._q_complete(racine, legal, logits, v_hat, sign)
                sc = gumbel + logits + self._sigma(q, racine, legal)
                cand = sorted(cand, key=lambda i: -sc[i])[: math.ceil(len(cand) / 2)]
        q = self._q_complete(racine, legal, logits, v_hat, sign)
        sig = self._sigma(q, racine, legal)
        pi = _softmax(logits + sig)
        visites = np.array([racine.enfants[a].n for a in legal])
        if k == 1:
            best = 0
        elif used == 0:
            best = int(np.argmax(gumbel + logits))
        else:
            # parmi les candidats survivants du halving séquentiel (les plus visités)
            sc = gumbel + logits + sig
            best = max(cand, key=lambda i: sc[i])
        return Resultat(legal[best], legal, pi.astype(np.float32), visites, q, logits, v_hat,
                        sign * racine.w0 / racine.n, used)

    def _sigma(self, q: np.ndarray, racine: Noeud, legal: list[Action]) -> np.ndarray:
        max_n = max(racine.enfants[a].n for a in legal)
        if self.p.normaliser_q:
            lo, hi = q.min(), q.max()
            q01 = (q - lo) / max(hi - lo, 1e-8)
        else:
            q01 = (q + 1.0) / 2.0
        return (self.p.c_visit + max_n) * self.p.c_scale * q01

    @staticmethod
    def _q_complete(racine: Noeud, legal, logits, v_hat: float, sign: float) -> np.ndarray:
        pri = _softmax(logits)
        q = np.empty(len(legal))
        vis = np.zeros(len(legal), bool)
        sum_n = 0.0
        for i, a in enumerate(legal):
            ch = racine.enfants[a]
            if ch.n > 0:
                q[i] = sign * ch.w0 / ch.n
                vis[i] = True
                sum_n += ch.n
        if sum_n > 0:
            pv = pri[vis].sum()
            v_mix = (v_hat + sum_n * (pri[vis] * q[vis]).sum() / max(pv, 1e-12)) / (1.0 + sum_n)
        else:
            v_mix = v_hat
        q[~vis] = v_mix
        return np.clip(q, -1.0, 1.0)

    # ------------------------------------------------------------ simulations
    def _vague(self, game: Game, me: int, racine: Noeud, taches: list[Action]):
        vivants, requetes = [], []
        for a0 in taches:
            gen = self._simuler(game, me, racine, a0)
            try:
                requetes.append(next(gen))
                vivants.append(gen)
            except StopIteration:
                pass
        if requetes:
            reponses = yield requetes
            for gen, rep in zip(vivants, reponses):
                try:
                    gen.send(rep)
                except StopIteration:
                    pass
                else:
                    raise RuntimeError("une simulation ne doit interroger le réseau qu'une fois")

    def _simuler(self, game: Game, me: int, racine: Noeud, a0: Action):
        P = self.p
        vl = P.perte_virtuelle if P.parallele > 1 else 0.0
        g = game.determinize(me, self.rng, rapide=True)
        node = racine.enfants[a0]
        chemin = [node]
        g.apply(a0)
        if vl:
            node.n += vl
            node.w0 -= vl if node.equipe == 0 else -vl
        while True:
            if g.done:
                v0 = 0.0 if g.winner is None else (1.0 if g.winner == 0 else -1.0)
                break
            legal = g.legal_actions()
            tm = g.team(g.to_move)
            s = 1.0 if tm == 0 else -1.0
            if len(legal) > 1:
                lg = node.logits
                if lg is None or any(a not in lg for a in legal):
                    logits, v = yield (g, legal)
                    if lg is None:
                        lg = node.logits = {}
                    for a, l in zip(legal, logits):
                        if a not in lg:
                            lg[a] = float(l)
                    v0 = s * v
                    break
                a = self._puct(node, legal, lg, s)
            else:
                a = legal[0]
            ch = node.enfants.get(a)
            if ch is None:
                ch = node.enfants[a] = Noeud(tm)
            g.apply(a)
            node = ch
            chemin.append(node)
            if vl:
                node.n += vl
                node.w0 -= vl if node.equipe == 0 else -vl
        racine.n += 1
        racine.w0 += v0
        for nd in chemin:
            if vl:
                nd.n -= vl
                nd.w0 += vl if nd.equipe == 0 else -vl
            nd.n += 1
            nd.w0 += v0

    def _puct(self, node: Noeud, legal: list[Action], lg: dict, s: float) -> Action:
        mx = max(lg[a] for a in legal)
        ex = [math.exp(lg[a] - mx) for a in legal]
        z = sum(ex)
        tot = 0.0
        mass = 0.0
        kids = node.enfants
        for i, a in enumerate(legal):
            ch = kids.get(a)
            if ch is not None and ch.n > 0:
                tot += ch.n
                mass += ex[i]
        parent_q = s * node.w0 / node.n if node.n > 0 else 0.0
        fpu = parent_q - self.p.fpu * math.sqrt(mass / z)
        c = self.p.c_puct * math.sqrt(max(tot, 1.0)) / z
        best, best_u = legal[0], -math.inf
        for i, a in enumerate(legal):
            ch = kids.get(a)
            if ch is not None and ch.n > 0:
                u = s * ch.w0 / ch.n + c * ex[i] / (1.0 + ch.n)
            else:
                u = fpu + c * ex[i]
            if u > best_u:
                best, best_u = a, u
        return best


def executer(generateurs: list, evaluateur) -> list[Resultat]:
    """Fait tourner plusieurs recherches en regroupant leurs requêtes en lots."""
    resultats: list[Resultat | None] = [None] * len(generateurs)
    actifs: dict[int, list] = {}
    for i, gen in enumerate(generateurs):
        actifs[i] = next(gen)
    while actifs:
        ids = list(actifs)
        flat = [r for i in ids for r in actifs[i]]
        reps = evaluateur.evaluer(flat)
        pos = 0
        for i in ids:
            n = len(actifs[i])
            try:
                actifs[i] = generateurs[i].send(reps[pos:pos + n])
            except StopIteration as fin:
                resultats[i] = fin.value
                del actifs[i]
            pos += n
    return resultats
