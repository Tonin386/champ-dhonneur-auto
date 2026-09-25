"""Partie hybride : une partie jouée sur un vrai plateau, dont l'écran tient le rôle de l'IA.

Voir docs/HYBRIDE.md. Deux joueurs :
  * l'IA, dont les pioches sont saisies (pièces tirées de son sac sur la table) ;
  * le joueur plateau, dont on ne saisit que ce qui est public : ses coups, pièce « cachée »
    pour les coups face cachée. Le moteur lui garde une main fictive, dont seule la taille
    compte ; une pièce révélée y est échangée avec une pièce cachée (sac, main, défausse cachée),
    ce que l'IA ne peut pas distinguer.

Le moteur n'est pas modifié : les pioches imposées passent par `Game.force_draws`, et le coup
qui déclenche une pioche est d'abord essayé sur une copie dont `_draw` signale la pièce attendue.
"""
from __future__ import annotations

from collections import Counter, deque

from .engine import FACE_DOWN, Action, Game, IllegalAction

CACHEE = "?"          # pièce d'un coup face cachée du joueur plateau (inconnue de l'écran)


def _pioche_moine(g: Game, p: int):
    """Pioche du Moine soldat de p en attente de décision (sinon None)."""
    pd = g.pending[-1] if g.pending else None
    return pd if pd is not None and pd.kind == "priest" and pd.player == p else None


def pieces_possibles(g: Game, p: int) -> Counter:
    """Pièces que p pourrait jouer d'après l'information publique : ses pièces cachées (sac, main,
    défausse cachée, pièce piochée par le Moine soldat)."""
    pl = g.players[p]
    cachees = pl.bag + pl.hand + pl.disc_down
    pd = _pioche_moine(g, p)
    if pd is not None and pd.coin is not None:
        cachees = cachees + [pd.coin]
    return Counter(cachees)


def coups_libres(g: Game, p: int) -> list[Action]:
    """Décisions que p pourrait prendre, quelle que soit sa main réelle. Les coups face cachée
    portent la pièce CACHEE ; les suites et le draft sont publics."""
    if g.done or g.to_move != p:
        return []
    pd = _pioche_moine(g, p)
    if g.in_draft or (g.pending and pd is None):
        return g.legal_actions()
    if pd is None and not g.players[p].hand:
        return []
    out: list[Action] = []
    vus: set[Action] = set()
    for c in sorted(pieces_possibles(g, p)):
        for a in g._coin_actions(p, c):
            if a.kind in FACE_DOWN:
                a = a._replace(coin=CACHEE)
            if a not in vus:
                vus.add(a)
                out.append(a)
    return out


def _echanger(pl, c: str, ailleurs: str | None) -> str:
    """Retire une pièce c des zones cachées de pl et y met `ailleurs` à sa place."""
    for zone in (pl.bag, pl.disc_down, pl.hand):
        if c in zone:
            i = zone.index(c)
            zone[i] = ailleurs
            return c
    raise IllegalAction(f"Pièce impossible : aucune pièce {c} ne peut être cachée chez ce joueur")


def concretiser(g: Game, p: int, a: Action) -> Action:
    """Rend jouable dans g le coup libre `a` de p (modifie les pièces fictives de p) et renvoie le
    coup concret. Déterministe : rejouer la même suite de coups redonne le même état."""
    if a.coin is None or g.to_move != p:
        return a
    pl = g.players[p]
    pd = _pioche_moine(g, p)
    if pd is not None:                        # pièce piochée par le Moine soldat
        if a.coin == CACHEE or a.coin == pd.coin:
            return a._replace(coin=pd.coin)
        pd.coin = _echanger(pl, a.coin, pd.coin)
        return a
    if a.coin == CACHEE:                      # face cachée : une pièce fictive de la main
        return a._replace(coin=sorted(pl.hand)[0])
    if a.coin not in pl.hand:                 # pièce révélée : elle était en main
        h0 = pl.hand[0]
        _echanger(pl, a.coin, h0)
        pl.hand[0] = a.coin
    return a


class TirageRequis(Exception):
    """Le coup essayé fait piocher l'IA et il manque une pièce saisie."""

    def __init__(self, joueur: int, sac: list[str], melange: bool, rang: int, total: int, moine: bool):
        super().__init__(f"pioche du joueur {joueur} : pièce {rang}/{total} parmi {''.join(sac)}")
        self.joueur, self.sac, self.melange = joueur, sac, melange
        self.rang, self.total, self.moine = rang, total, moine


def essayer(g: Game, a: Action, ia: int, tirages: list[str]) -> Game:
    """Applique `a` sur une copie de g, les pioches de l'IA étant `tirages` (dans l'ordre).
    Lève TirageRequis à la première pioche non saisie, IllegalAction si une pièce saisie n'est pas
    dans le sac ou si des pièces sont en trop."""
    h = g.copy(log=False)
    h.forced = {}
    file = list(tirages)

    def tirer(p: int) -> str | None:
        if p != ia:
            return Game._draw(h, p)
        pl = h.players[p]
        if not file:
            sac = pl.bag or pl.disc_up + pl.disc_down
            if not sac:
                return None                   # plus aucune pièce : la pioche s'arrête
            moine = _pioche_moine(h, p) is not None
            total = 1 if moine else min(3, len(pl.hand) + len(pl.bag) + len(pl.disc_up) + len(pl.disc_down))
            raise TirageRequis(p, sorted(sac), not pl.bag, 1 if moine else len(pl.hand) + 1, total, moine)
        h.forced[p] = deque([file.pop(0)])
        return Game._draw(h, p)                # remise en sac, puis vérifie la pièce imposée

    h._draw = tirer
    h.apply(a)
    if file:
        raise IllegalAction(f"Pièces saisies en trop : {''.join(file)}")
    return h


def retirer_main(g: Game, p: int) -> None:
    """Remet la main de p dans son sac (pioche initiale saisie ensuite)."""
    pl = g.players[p]
    pl.bag += pl.hand
    pl.hand = []


def piocher_initial(g: Game, p: int, pieces: list[str]) -> None:
    """Première main de p, saisie (le sac est plein : pas de remise en sac)."""
    pl = g.players[p]
    for c in pieces:
        if c not in pl.bag:
            raise IllegalAction(f"Pioche impossible : {c} n'est pas dans le sac")
        pl.bag.remove(c)
        pl.hand.append(c)
