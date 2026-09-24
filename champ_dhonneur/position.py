"""Position d'une partie à 2 joueurs : export et reconstruction (éditeur de position).

Format JSON :

    {"unites": [["S","P","X","H"], ["A","C","L","E"]],
     "plateau": [{"case": "b2", "joueur": 0, "unite": "S", "pieces": 2}, …],
     "controle": {"b1": 0, "e1": 0, "c6": 1, "f5": 1},
     "trait": 0, "initiative": 0, "manche": 1,
     "joueurs": [{"main": ["S", "*"], "sac": […], "defausse": […], "defausse_cachee": […],
                  "reserve": {"S": 3, …}}, …]}

Les pièces éliminées (remises dans la boîte) se déduisent du reste : pour chaque unité,
plateau + main + sac + défausses + réserve ≤ nombre de pièces de l'unité. Les marqueurs
Contrôle restant à poser se déduisent des Lieux contrôlés.
"""
from __future__ import annotations

import random
from collections import Counter

from .engine import Game, Unit
from .units import ROYAL, UNITS

MARQUEURS = 6


def position(game: Game) -> dict:
    if game.mode != "2J":
        raise ValueError("Positions : partie à 2 joueurs uniquement")
    n = game.spec.names
    return {
        "unites": [list(p.units) for p in game.players],
        "plateau": [{"case": n[c], "joueur": u.owner, "unite": u.utype, "pieces": u.coins}
                    for c, u in sorted(game.board.items())],
        "controle": {n[c]: t for c, t in sorted(game.control.items()) if t is not None},
        "trait": game.to_move if not game.pending else game.current,
        "initiative": game.initiative,
        "manche": game.round,
        "joueurs": [{"main": sorted(p.hand), "sac": sorted(p.bag), "defausse": sorted(p.disc_up),
                     "defausse_cachee": sorted(p.disc_down), "reserve": dict(p.reserve)}
                    for p in game.players],
    }


def depuis_position(pos: dict, graine: int | None = None) -> Game:
    """Reconstruit une partie ; ValueError (message en français) si la position est impossible."""
    try:
        unites = [[str(u).upper() for u in l] for l in pos["unites"]]
    except (KeyError, TypeError):
        raise ValueError("Unités des deux joueurs manquantes")
    if len(unites) != 2:
        raise ValueError("Deux joueurs attendus")
    for j, l in enumerate(unites):
        if len(l) != 4 or len(set(l)) != 4 or any(u not in UNITS for u in l):
            raise ValueError(f"Joueur {j + 1} : quatre unités différentes attendues")
    if set(unites[0]) & set(unites[1]):
        raise ValueError("Les deux joueurs ne peuvent pas partager une unité")

    g = Game("2J", unites, seed=graine if graine is not None else random.randrange(2**31), first=0)
    sp = g.spec
    trait = _joueur(pos.get("trait", 0), "trait")
    initiative = _joueur(pos.get("initiative", trait), "initiative")

    g.board = {}
    for e in pos.get("plateau", []):
        c = _case(sp, e.get("case"))
        j = _joueur(e.get("joueur"), "unité")
        u = str(e.get("unite", "")).upper()
        k = int(e.get("pieces", 1))
        if u not in unites[j]:
            raise ValueError(f"{sp.names[c]} : l'unité {u} n'appartient pas au joueur {j + 1}")
        if not 1 <= k <= UNITS[u].count:
            raise ValueError(f"{sp.names[c]} : {k} pièces impossibles pour {UNITS[u].name}")
        if c in g.board:
            raise ValueError(f"{sp.names[c]} : deux unités sur la même case")
        g.board[c] = Unit(j, u, k)
    for j in (0, 1):
        for u in unites[j]:
            if len(g.units_of(j, u)) > UNITS[u].max_units:
                raise ValueError(f"{UNITS[u].name} : {UNITS[u].max_units} unité(s) au plus sur le plateau")

    g.control = {loc: None for loc in sp.locations}
    for nom, t in (pos.get("controle") or {}).items():
        c = _case(sp, nom)
        if c not in g.loc_set:
            raise ValueError(f"{nom} n'est pas un Lieu")
        if t is not None:
            g.control[c] = _joueur(t, "contrôle")
    places = [sum(1 for t in g.control.values() if t == e) for e in (0, 1)]
    for e in (0, 1):
        if places[e] >= MARQUEURS:
            raise ValueError("Un joueur qui contrôle 6 Lieux a déjà gagné")
    g.markers_left = [MARQUEURS - places[0], MARQUEURS - places[1]]

    zones = pos.get("joueurs") or [{}, {}]
    for j, pl in enumerate(g.players):
        z = zones[j] if j < len(zones) else {}
        pl.hand = _pieces(z.get("main", []), unites[j], j, "main")
        pl.bag = _pieces(z.get("sac", []), unites[j], j, "sac")
        pl.disc_up = _pieces(z.get("defausse", []), unites[j], j, "défausse")
        pl.disc_down = _pieces(z.get("defausse_cachee", []), unites[j], j, "défausse cachée")
        reserve = z.get("reserve")
        pl.reserve = {u: int((reserve or {}).get(u, 0)) for u in unites[j]}
        if ROYAL in pl.disc_up:
            raise ValueError(f"Joueur {j + 1} : le Sceau royal ne se défausse que face cachée")
        if Counter(pl.hand + pl.bag + pl.disc_down)[ROYAL] != 1:
            raise ValueError(f"Joueur {j + 1} : un Sceau royal (main, sac ou défausse cachée) attendu")
        if len(pl.hand) > 3:
            raise ValueError(f"Joueur {j + 1} : 3 pièces en main au plus")
        compte = Counter(pl.hand + pl.bag + pl.disc_up + pl.disc_down)
        for c, u in g.board.items():
            if u.owner == j:
                compte[u.utype] += u.coins
        pl.lost = {}
        for u in unites[j]:
            if pl.reserve[u] < 0:
                raise ValueError(f"Joueur {j + 1} : réserve négative ({UNITS[u].name})")
            reste = UNITS[u].count - compte[u] - pl.reserve[u]
            if reste < 0:
                raise ValueError(f"Joueur {j + 1} : {-reste} pièce(s) {UNITS[u].name} de trop "
                                 f"({UNITS[u].count} au total)")
            pl.lost[u] = reste

    if not g.players[trait].hand:
        raise ValueError("Le joueur au trait doit avoir au moins une pièce en main")
    g.round = max(1, int(pos.get("manche", 1)))
    g.initiative = initiative
    g.round_first = initiative
    g.initiative_moved = False
    g.first_player = initiative
    g.current = trait
    g.pending = []
    g.done, g.winner = False, None
    g.forced = {}
    g.log = []
    return g


def _joueur(v, quoi: str) -> int:
    if v not in (0, 1):
        raise ValueError(f"Joueur invalide ({quoi}) : {v!r}")
    return int(v)


def _case(sp, nom) -> int:
    if nom not in sp.index:
        raise ValueError(f"Case inconnue : {nom!r}")
    return sp.index[nom]


def _pieces(l, unites: list[str], j: int, zone: str) -> list[str]:
    out = [str(c).upper() for c in l]
    for c in out:
        if c != ROYAL and c not in unites:
            raise ValueError(f"Joueur {j + 1}, {zone} : pièce {c} étrangère à ses unités")
    return out
