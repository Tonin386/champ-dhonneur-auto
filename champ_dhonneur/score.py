"""Score d'une position, à la manière des moteurs d'échecs, et victoires forcées.

Échelle : **un bastion (Lieu contrôlé) d'avance sur l'adversaire vaut 10 points**. Or (équipe 0,
en bas du plateau) a un score positif quand il mène, Argent (équipe 1) un score négatif.

Le réseau estime l'espérance de gain v ∈ [-1, 1] (victoire − défaite). Sur les parties de
l'entraînement, l'espérance observée à d bastions d'avance suit v ≈ tanh(K · d) : le score est donc
l'avance en bastions qui donnerait les mêmes chances, ×10 (« champ calibrer » recalcule K).

Notation affichée (voir docs/NOTATION.md) :

    +12,4    Or mène d'un peu plus d'un bastion (équivalent)
    −3,0     léger avantage Argent
    #3       Or pose son dernier marqueur Contrôle en 3 coups au plus, quoi que fasse Argent
             et quels que soient les tirages (« mat en 3 »)
    #-2      Argent gagne en 2 coups
    1-0      partie terminée (0-1, ½-½)

Une victoire forcée est cherchée jusqu'à la fin de la manche en cours : au-delà, les pièces
piochées dans le sac sont inconnues et rien n'est garanti. Un « coup » est une pièce jouée
(avec ses effets enchaînés : Berserk, Moine soldat…), comme un coup aux échecs.
"""
from __future__ import annotations

import math
import time

from .engine import CONTROL, Action, Game

K = 1.05            # espérance de gain à 1 bastion d'avance : tanh(K) ≈ 0,78 (≈ 89 % de victoires)
POINTS_BASTION = 10.0
EQUIPES = ("Or", "Argent")


# ------------------------------------------------------------------ échelle
def bastions(game: Game) -> list[int]:
    """Lieux contrôlés par chaque équipe."""
    return [sum(1 for t in game.control.values() if t == e) for e in (0, 1)]


def depuis_valeur(v_or: float, k: float | None = None) -> float:
    """Espérance de gain d'Or (dans [-1, 1]) → score (10 = un bastion d'avance).

    `k` : échelle recalibrée par l'entraînement du modèle (runs/<nom>/echelle.json), sinon K."""
    v = max(-0.9999, min(0.9999, v_or))
    return POINTS_BASTION * math.atanh(v) / (k or K)


def vers_valeur(score: float, k: float | None = None) -> float:
    return math.tanh(score * (k or K) / POINTS_BASTION)


def texte_score(score: float) -> str:
    s = f"{abs(score):.1f}".replace(".", ",")
    return ("+" if score >= 0.05 else "−" if score <= -0.05 else "") + s


def texte_mat(equipe: int, coups: int) -> str:
    return f"#{coups}" if equipe == 0 else f"#-{coups}"


def appreciation(score: float) -> tuple[str, str]:
    """Symbole (échecs) et libellé : = ⩲ ± +− (et leurs symétriques)."""
    a = abs(score)
    if a < 3:
        return "=", "Égalité"
    qui = EQUIPES[0 if score > 0 else 1]
    if a < 7:
        return ("⩲" if score > 0 else "⩱"), f"Léger avantage {qui}"
    if a < 15:
        return ("±" if score > 0 else "∓"), f"Net avantage {qui}"
    return ("+−" if score > 0 else "−+"), f"Avantage décisif {qui}"


# ------------------------------------------------------------------ victoire forcée
class _Budget(Exception):
    pass


class Solveur:
    """Cherche une victoire forcée (toutes défenses, tous tirages) avant la fin de la manche.

    `observateur` : joueur dont on respecte l'information. None = analyse omnisciente (mains
    réelles). Sinon, seule l'équipe de l'observateur est examinée, et l'adversaire est supposé
    pouvoir jouer n'importe quelle pièce qu'il pourrait avoir (sac, main, défausse cachée) :
    la preuve vaut alors quelle que soit sa main réelle.
    """

    def __init__(self, observateur: int | None = None, noeuds: int = 60_000, secondes: float = 2.0):
        self.observateur = observateur
        self.max_noeuds, self.secondes = noeuds, secondes

    def chercher(self, game: Game, max_coups: int = 3) -> dict | None:
        """{"equipe", "coups", "action"} (action : premier coup gagnant si l'équipe a le trait)."""
        if game.done or game.mode != "2J" or game.in_draft:
            return None
        self.manche = game.round
        self.noeuds, self.fin = 0, time.monotonic() + self.secondes
        equipes = [game.team(game.to_move), 1 - game.team(game.to_move)]
        if self.observateur is not None:
            equipes = [game.team(self.observateur)]
        try:
            for n in range(1, max_coups + 1):
                for t in equipes:
                    self.t = t
                    if not self._possible(game, n):
                        continue
                    self.premier = None
                    if self._gagne(game, n, racine=True):
                        return {"equipe": t, "coups": n, "action": self.premier}
        except _Budget:
            return None
        return None

    # ---- borne : combien de marqueurs l'équipe peut-elle encore poser avec n pièces ?
    def _possible(self, g: Game, n: int) -> bool:
        t = self.t
        besoin = g.markers_left[t]
        if besoin > 3 * n + 5:
            return False
        joueurs = [p for p in g.players if p.team == t]
        pieces: list[int] = []
        for pl in joueurs:
            merc = any(u.owner == pl.idx and u.utype == "M" for u in g.board.values()) and pl.reserve.get("M", 0) > 0
            for c in pl.hand:
                if c == "B":
                    k = max((u.coins for u in g.board.values() if u.owner == pl.idx and u.utype == "B"), default=0)
                elif c == "F":
                    k = 2 if sum(1 for u in g.board.values() if u.owner == pl.idx and u.utype == "F") == 2 else 1
                elif c == "R":
                    k = 1 + len(pl.bag) + len(pl.disc_up) + len(pl.disc_down)
                elif c == "*":
                    k = 0
                else:
                    k = 1
                pieces.append(k + (1 if merc else 0))
        pieces.sort(reverse=True)
        borne = sum(pieces[:n])
        if g.pending and g.team(g.pending[-1].player) == t:
            borne += 6
        return borne >= besoin

    # ---- recherche ET/OU
    def _compter(self) -> None:
        self.noeuds += 1
        if self.noeuds > self.max_noeuds or (self.noeuds & 255 == 0 and time.monotonic() > self.fin):
            raise _Budget

    def _gagne(self, g: Game, n: int, racine: bool = False) -> bool:
        """L'équipe self.t gagne-t-elle à coup sûr en jouant au plus n pièces de plus ?"""
        if g.done:
            return g.winner == self.t
        if g.round != self.manche:
            return False                         # nouvelle pioche : plus rien de garanti
        self._compter()
        p = g.to_move
        a_moi = g.team(p) == self.t
        principal = not g.pending
        if a_moi:
            if principal:
                if n <= 0 or not self._possible(g, n):
                    return False
                n -= 1
            actions = sorted(g.legal_actions(), key=lambda a: a.kind != CONTROL)
            for a in actions:
                if all(self._gagne(h, n) for h in self._suites(g, a)):
                    if racine:
                        self.premier = a
                    return True
            return False
        # défense : toutes les réponses doivent perdre
        for h0, actions in self._defenses(g, principal):
            for a in actions:
                if not all(self._gagne(h, n) for h in self._suites(h0, a)):
                    return False
        return True

    def _cache(self, p: int) -> bool:
        """La main de p est-elle inconnue de l'observateur ?"""
        return self.observateur is not None and p != self.observateur

    def _defenses(self, g: Game, principal: bool):
        """(position, actions) à réfuter. Main cachée : chaque pièce que l'adversaire pourrait avoir."""
        p = g.to_move
        if not (principal and self._cache(p)):
            yield g, g.legal_actions()
            return
        pl = g.players[p]
        for c in sorted(set(pl.bag + pl.hand + pl.disc_down)):
            h = g.copy(log=False)
            _mettre_en_main(h.players[p], c)
            yield h, [a for a in h.legal_actions() if a.coin == c]

    def _suites(self, g: Game, a: Action) -> list[Game]:
        """Positions après a, une par pièce que le Moine soldat pourrait piocher."""
        tirages: list[list[str]] = []
        h = self._appliquer(g, a, None, tirages)
        if not tirages:
            return [h]
        out = [h]
        for c in tirages[0][1:]:
            out.append(self._appliquer(g, a, c, []))
        return out

    def _appliquer(self, g: Game, a: Action, impose: str | None, tirages: list) -> Game:
        h = g.copy(log=False)
        manche = self.manche
        solveur = self

        def tirer(p: int) -> str | None:
            pl = h.players[p]
            if not pl.bag:
                pl.bag = pl.disc_up + pl.disc_down
                pl.disc_up, pl.disc_down = [], []
            if not pl.bag:
                return None
            if h.round != manche:                # pioche de la manche suivante : sans importance
                return pl.bag.pop()
            possibles = sorted(set(pl.bag + (pl.hand + pl.disc_down if solveur._cache(p) else [])))
            c = impose if impose is not None else possibles[0]
            tirages.append(possibles)
            if c not in pl.bag:                  # pièce cachée ailleurs (main, défausse) : échange
                zone = pl.hand if c in pl.hand else pl.disc_down
                zone[zone.index(c)] = pl.bag.pop()
            else:
                pl.bag.remove(c)
            return c

        h._draw = tirer
        h.apply(a)
        return h


def _mettre_en_main(pl, c: str) -> None:
    """Échange une pièce de la main contre c (pris dans le sac ou la défausse cachée)."""
    if c in pl.hand or not pl.hand:
        return
    zone = pl.bag if c in pl.bag else pl.disc_down
    i = zone.index(c)
    zone[i], pl.hand[0] = pl.hand[0], zone[i]


# ------------------------------------------------------------------ calibrage
def calibrer(fichiers) -> dict:
    """Ajuste K sur des relevés .nch : espérance de gain d'Or ≈ tanh(K · avance en bastions)."""
    from .notation import import_record, parse_headers
    points: list[tuple[int, float]] = []
    for f in fichiers:
        texte = open(f, encoding="utf-8").read()
        z = {"1-0": 1.0, "0-1": -1.0}.get(parse_headers(texte).get("Resultat", ""), 0.0)
        fin = import_record(texte)
        g = fin.neuve()
        for _, _, a in fin.log:
            if not g.pending:
                b = bastions(g)
                points.append((b[0] - b[1], z))
            g.apply(a)
    if not points:
        raise ValueError("aucune position")

    def vraisemblance(k: float) -> float:
        s = 0.0
        for d, z in points:
            p = min(max((1 + math.tanh(k * d)) / 2, 1e-6), 1 - 1e-6)
            y = (1 + z) / 2
            s += y * math.log(p) + (1 - y) * math.log(1 - p)
        return s

    k = max((i / 100 for i in range(5, 300)), key=vraisemblance)
    return {"K": k, "positions": len(points), "gain_1_bastion": round((1 + math.tanh(k)) / 2, 3)}
