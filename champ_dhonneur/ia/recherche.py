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

Deux façons de conduire la recherche :

* `generateur` : budget fixe de simulations, un seul halving séquentiel (auto-jeu, évaluations,
  bot à niveau fixe) ;
* `approfondir` : recherche progressive pour l'analyse et le jeu au temps, arrêtée par une durée,
  une profondeur, un nombre de simulations ou une demande extérieure (analyse infinie). Elle
  procède par **approfondissement itératif** : la passe d (profondeur d) est un halving séquentiel
  de 32·2^(d−1) simulations sur les meilleurs candidats du moment, l'arbre étant conservé d'une
  passe à l'autre ; chaque profondeur coûte donc autant que toutes les précédentes réunies, comme
  pour un moteur d'échecs. Dès 1 024 simulations par passe (profondeur 6), tous les coups légaux
  sont candidats : aucun coup n'échappe à une analyse longue.

Choix du coup (`choisir`) : parmi les coups les plus explorés (au moins la moitié des visites du
plus visité), celui dont le score est le meilleur. Le classement affiché par l'analyse
(`classement`) suit la même règle : le premier coup de l'analyse est toujours celui que l'IA joue.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from ..engine import CONTROL, Action, Game


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
    # coup joué : False = le meilleur coup de la recherche, sans bruit (le bruit ne sert alors
    # qu'à choisir les candidats explorés) ; True = tiré avec le bruit de Gumbel (exploration)
    bruit_coup: bool = True
    coup_gagnant: bool = False   # un coup qui gagne immédiatement est toujours joué
    max_noeuds: int = 250_000    # recherche progressive : au-delà, l'arbre est élagué (≈ 3 Ko par nœud)


@dataclass
class Limite:
    """Arrêt d'une recherche progressive : la première limite atteinte l'arrête. Sans aucune
    limite, elle ne s'arrête que sur demande (`arret.set()`, analyse infinie)."""
    simulations: int | None = None
    secondes: float | None = None
    profondeur: int | None = None
    arret: object | None = None      # threading.Event (ou tout objet muni de is_set())

    def atteinte(self, simulations: int, debut: float) -> bool:
        return ((self.simulations is not None and simulations >= self.simulations)
                or (self.secondes is not None and time.monotonic() - debut >= self.secondes)
                or (self.arret is not None and self.arret.is_set()))


class _Suivi:
    """Mesures d'une recherche progressive : horizon des simulations (en coups) et taille de l'arbre."""
    __slots__ = ("somme", "n", "max", "noeuds")

    def __init__(self, noeuds: int):
        self.somme = self.n = self.max = 0
        self.noeuds = noeuds

    def fin(self, coups: int) -> None:
        self.somme += coups
        self.n += 1
        if coups > self.max:
            self.max = coups


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


def coups_gagnants(game: Game, legal: list[Action]) -> list[int]:
    """Indices des coups qui gagnent immédiatement (dernier marqueur Contrôle posé). La victoire ne
    dépend que des marqueurs, information publique : le test se fait sur une copie de la partie."""
    team = game.team(game.to_move)
    if game.markers_left[team] != 1:
        return []
    out = []
    for i, a in enumerate(legal):
        if a.kind == CONTROL:
            h = game.copy(log=False)
            h.apply(a)
            if h.done and h.winner == team:
                out.append(i)
    return out


def choisir(visites: np.ndarray, score: np.ndarray) -> int:
    """Coup joué : le meilleur score parmi les coups les plus explorés (au moins la moitié des
    visites du plus visité). Sans visite, le meilleur score (a priori du réseau)."""
    mx = visites.max() if len(visites) else 0.0
    if mx <= 0:
        return int(np.argmax(score))
    return int(np.argmax(np.where(visites >= mx / 2, score, -np.inf)))


def classement(visites: np.ndarray, score: np.ndarray) -> list[int]:
    """Ordre d'affichage des coups, cohérent avec `choisir` : par niveau d'exploration (chaque
    niveau a moitié moins de visites que le précédent, comme les éliminations du halving
    séquentiel), puis par score. Le premier est le coup joué."""
    mx = visites.max() if len(visites) else 0.0
    if mx <= 0:
        return sorted(range(len(score)), key=lambda i: -score[i])
    niveau = [math.floor(math.log2(mx / n)) if n > 0 else 99 for n in visites]
    return sorted(range(len(score)), key=lambda i: (niveau[i], -score[i]))


class RechercheGumbel:
    def __init__(self, params: ParamsRecherche | None = None, seed: int | None = None):
        self.p = params or ParamsRecherche()
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- racine
    @staticmethod
    def _racine(legal: list[Action], logits: np.ndarray, v_hat: float, team: int, sign: float) -> Noeud:
        racine = Noeud(1 - team)
        racine.logits = {a: float(l) for a, l in zip(legal, logits)}
        racine.n, racine.w0 = 1.0, sign * v_hat
        for a in legal:
            racine.enfants[a] = Noeud(team)
        return racine

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
        racine = self.racine = self._racine(legal, logits, v_hat, team, sign)
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
        g_coup = gumbel if P.bruit_coup else 0.0
        if k == 1:
            best = 0
        elif used == 0:
            best = int(np.argmax(g_coup + logits))
        else:
            # parmi les candidats survivants du halving séquentiel (les plus visités)
            sc = g_coup + logits + sig
            best = max(cand, key=lambda i: sc[i])
        if P.coup_gagnant and k > 1:
            gagnants = coups_gagnants(game, legal)
            if gagnants and best not in gagnants:
                best = gagnants[0]
        return Resultat(legal[best], legal, pi.astype(np.float32), visites, q, logits, v_hat,
                        sign * racine.w0 / racine.n, used)

    # ------------------------------------------------------- recherche progressive
    def approfondir(self, game: Game, limite: Limite, suivi=None, base: int = 32, intervalle: float = 0.5):
        """Recherche progressive (approfondissement itératif, voir l'en-tête du module).

        `suivi(resultat)` est appelé à la fin de chaque profondeur et toutes les `intervalle`
        secondes : `resultat.infos` donne la profondeur atteinte, l'horizon des simulations (coups
        anticipés, moyen et maximal), la durée et le classement des coups. Au moins une première
        passe est toujours menée à son terme, sauf arrêt demandé de l'extérieur. Le coup renvoyé est
        celui de `choisir` (score : logits + σ(Q complété), sans bruit)."""
        P = self.p
        debut = time.monotonic()
        me = game.to_move
        team = game.team(me)
        sign = 1.0 if team == 0 else -1.0
        legal = game.legal_actions()
        [(logits, v_hat)] = yield [(game, legal)]
        logits = np.asarray(logits, np.float64)
        k = len(legal)
        racine = self.racine = self._racine(legal, logits, v_hat, team, sign)
        stats = _Suivi(1 + k)
        etat = {"profondeur": 0, "elagages": 0, "used": 0}
        gagnants = coups_gagnants(game, legal) if k > 1 else []

        def resultat(en_cours: bool) -> Resultat:
            q = self._q_complete(racine, legal, logits, v_hat, sign)
            sig = self._sigma(q, racine, legal)
            score = logits + sig
            visites = np.array([racine.enfants[a].n for a in legal])
            best = gagnants[0] if gagnants else choisir(visites, score)
            ordre = classement(visites, score)
            if gagnants:
                ordre = [best] + [i for i in ordre if i != best]
            r = Resultat(legal[best], legal, _softmax(score).astype(np.float32), visites, q, logits, v_hat,
                         sign * racine.w0 / racine.n, etat["used"])
            r.infos = {"profondeur": etat["profondeur"], "en_cours": en_cours, "ordre": ordre,
                       "horizon": stats.somme / stats.n if stats.n else 0.0, "horizon_max": stats.max,
                       "secondes": time.monotonic() - debut, "noeuds": stats.noeuds,
                       "elagages": etat["elagages"]}
            return r

        dernier = time.monotonic()
        arret = False
        while not arret:
            budget = base << etat["profondeur"]
            m = min(k, max(P.m, budget // 16))
            # vagues plus grosses pour les longues passes : lots GPU plus efficaces (perte virtuelle)
            par = max(P.parallele, min(32, budget // 64))
            q = self._q_complete(racine, legal, logits, v_hat, sign)
            sc = logits + self._sigma(q, racine, legal)
            cand = sorted(range(k), key=lambda i: -sc[i])[:m]
            n_phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1
            fait = 0
            while True:
                if len(cand) <= 2:
                    per = math.ceil((budget - fait) / len(cand))
                else:
                    per = max(1, budget // (n_phases * len(cand)))
                taches = [legal[i] for _ in range(per) for i in cand][: budget - fait]
                for s in range(0, len(taches), par):
                    # la première passe va à son terme (sauf arrêt demandé) : il faut un coup
                    premiere = etat["profondeur"] == 0 and not (limite.arret is not None and limite.arret.is_set())
                    if not premiere and limite.atteinte(etat["used"], debut):
                        arret = True
                        break
                    vague = taches[s:s + par]
                    yield from self._vague(game, me, racine, vague, stats, par)
                    etat["used"] += len(vague)
                    fait += len(vague)
                    if stats.noeuds > P.max_noeuds:
                        self._elaguer(racine, stats, P.max_noeuds // 2)
                        etat["elagages"] += 1
                    if suivi is not None and time.monotonic() - dernier >= intervalle:
                        suivi(resultat(True))
                        dernier = time.monotonic()
                if arret or len(cand) == 1 or fait >= budget:
                    break
                q = self._q_complete(racine, legal, logits, v_hat, sign)
                sc = logits + self._sigma(q, racine, legal)
                cand = sorted(cand, key=lambda i: -sc[i])[: math.ceil(len(cand) / 2)]
            if arret:
                break
            etat["profondeur"] += 1
            fini = (limite.profondeur is not None and etat["profondeur"] >= limite.profondeur) \
                or limite.atteinte(etat["used"], debut)
            if suivi is not None and not fini:
                suivi(resultat(True))
                dernier = time.monotonic()
            arret = fini
        r = resultat(False)
        if suivi is not None:
            suivi(r)
        return r

    @staticmethod
    def _elaguer(racine: Noeud, stats: _Suivi, cible: int) -> None:
        """Libère la mémoire d'une longue recherche : retire les nœuds les moins visités (et leurs
        sous-arbres), jamais les coups de la racine ; les statistiques des parents sont conservées
        et un nœud retiré est simplement réexploré s'il est de nouveau atteint."""
        seuil = 1.0
        while True:
            total = 1 + len(racine.enfants)
            pile = list(racine.enfants.values())
            while pile:
                nd = pile.pop()
                for a in [a for a, c in nd.enfants.items() if c.n <= seuil]:
                    del nd.enfants[a]
                total += len(nd.enfants)
                pile.extend(nd.enfants.values())
            if total <= cible or seuil > 1e9:
                stats.noeuds = total
                return
            seuil *= 2

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
    def _vague(self, game: Game, me: int, racine: Noeud, taches: list[Action], stats: _Suivi | None = None,
               parallele: int | None = None):
        vivants, requetes = [], []
        for a0 in taches:
            gen = self._simuler(game, me, racine, a0, stats, parallele)
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

    def _simuler(self, game: Game, me: int, racine: Noeud, a0: Action, stats: _Suivi | None = None,
                 parallele: int | None = None):
        P = self.p
        vl = P.perte_virtuelle if (parallele or P.parallele) > 1 else 0.0
        g = game.determinize(me, self.rng, rapide=True)
        node = racine.enfants[a0]
        chemin = [node]
        coups = 0 if g.pending else 1      # horizon : décisions principales (pièces jouées)
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
                if stats is not None:
                    stats.noeuds += 1
            if not g.pending:
                coups += 1
            g.apply(a)
            node = ch
            chemin.append(node)
            if vl:
                node.n += vl
                node.w0 -= vl if node.equipe == 0 else -vl
        if stats is not None:
            stats.fin(coups)
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
