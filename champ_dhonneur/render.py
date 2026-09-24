"""Affichage texte du plateau (console).

Chaque case occupe deux lignes :
    ligne haute : coordonnée + Lieu ([ ] neutre, [B] Blanc, [N] Noir)
    ligne basse : unité — MAJUSCULE = Blanc, minuscule = Noir, suivie du
                  nombre de pièces de la pile (ex. « S2 », « a1 »).
"""
from __future__ import annotations

from .engine import Game
from .units import unit_name

CELL_W = 6


def board_text(game: Game) -> str:
    sp = game.spec
    max_y2 = max(y for _, y in sp.xy2) + 1
    n_cols = len(sp.col_lengths)
    grid = [[" " * CELL_W for _ in range(n_cols)] for _ in range(max_y2 + 1)]
    for i, (x, y2) in enumerate(sp.xy2):
        name = sp.names[i]
        if i in game.loc_set:
            ctl = game.control[i]
            flag = "[ ]" if ctl is None else ("[B]" if ctl == 0 else "[N]")
        else:
            flag = "   "
        top = f"{name:<3}{flag}"[:CELL_W]
        u = game.board.get(i)
        if u is None:
            bottom = "  ·   "
        else:
            letter = u.utype if game.team(u.owner) == 0 else u.utype.lower()
            tag = f"{letter}{u.coins}"
            if game.n_players == 4:
                tag += str(u.owner + 1)
            bottom = f" {tag:<5}"
        row_top = max_y2 - (y2 + 1)
        row_bot = max_y2 - y2
        grid[row_top][x] = top.ljust(CELL_W)
        grid[row_bot][x] = bottom.ljust(CELL_W)
    lines = ["".join(r).rstrip() for r in grid]
    return "\n".join(l for l in lines if l.strip())


def player_text(game: Game, p: int, reveal: bool) -> str:
    pl = game.players[p]
    team = "Blanc" if pl.team == 0 else "Noir"
    who = f"J{p + 1} ({team})" if game.n_players == 4 else team
    units = ", ".join(f"{u}={unit_name(u)}" for u in pl.units)
    hand = " ".join(sorted(pl.hand)) if reveal else f"{len(pl.hand)} pièce(s)"
    res = " ".join(f"{u}:{n}" for u, n in pl.reserve.items())
    up = " ".join(sorted(pl.disc_up)) or "—"
    extra = ""
    if game.initiative == p:
        extra = "  ◆ initiative"
    return (f"{who}{extra}\n  unités : {units}\n  main : {hand}   sac : {len(pl.bag)}"
            f"   défausse visible : {up}   cachée : {len(pl.disc_down)}\n  réserve : {res}"
            f"   marqueurs à placer : {game.markers_left[pl.team]}")


def full_text(game: Game, viewer: int | None) -> str:
    parts = [f"Manche {game.round}", board_text(game), ""]
    for p in range(game.n_players):
        parts.append(player_text(game, p, reveal=viewer is None or p == viewer))
    return "\n".join(parts)
