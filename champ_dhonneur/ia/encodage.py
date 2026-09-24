"""Encodage des positions et des actions pour le réseau de neurones (2 joueurs).

Principes :
  * **Point de vue du joueur au trait.** Tout est exprimé en « moi / adversaire ».
    Le plateau 2J est symétrique par rotation de 180° : pour Noir, on tourne
    le plateau, si bien que le réseau voit toujours sa propre armée en bas.
  * **Aucune fuite d'information.** On n'encode que ce que le joueur au trait
    sait réellement : sa main, la composition de son sac, sa défausse cachée ;
    pour l'adversaire, seulement le multiensemble {sac + main + défausse cachée}
    (déductible de l'information publique) et la taille de chaque zone.
  * **Actions par caractéristiques.** Au lieu d'un immense vecteur de sortie
    fixe, chaque action légale est décrite par quelques entiers (type, pièce,
    unité, cases). Le réseau calcule un score par action à partir de ces
    entiers et des représentations des cases concernées (réseau « pointeur »).
    Cela généralise naturellement aux 1820 compositions d'armées possibles.
  * **Mise en place avancée (draft).** Les 8 jetons d'unité sont les 8 cartes en jeu :
    mes unités (camp 0), celles de l'adversaire (camp 1), puis les cartes encore
    disponibles (camp 2). Choisir une carte est une action comme les autres. Hors draft,
    les caractéristiques propres au draft sont nulles : une position issue d'un draft
    s'encode exactement comme la même position à armées imposées.

Version 2 de l'encodage (`VERSION`) : incompatible avec les modèles de la version 1.

Tout est en numpy pur pour pouvoir encoder dans les processus d'auto-jeu.
"""
from __future__ import annotations

import numpy as np

from ..board import get_board
from ..engine import (ATTACK, BOLSTER, CONTROL, DEPLOY, DRAFT, INITIATIVE, MOVE, PASS, RECRUIT,
                      RG_RESERVE, RG_UNIT, SKIP, TACTIC, Action, Game)
from ..units import ALL_LETTERS, ROYAL, UNITS

VERSION = 2

SPEC = get_board("2J")
N_CELLS = SPEC.n_cells                 # 37
NONE_CELL = N_CELLS                    # index « pas de case »
N_UNIT_TOKENS = 8                      # les 8 cartes en jeu : à moi, adverses, disponibles
N_TOKENS = N_CELLS + N_UNIT_TOKENS + 1  # + jeton global

# types de pièces : 0 = aucune, 1..16 = unités, 17 = Sceau royal
COIN_ID = {u: i + 1 for i, u in enumerate(ALL_LETTERS)}
COIN_ID[ROYAL] = len(ALL_LETTERS) + 1
N_COIN_TYPES = len(ALL_LETTERS) + 2    # 18

KINDS = [DEPLOY, BOLSTER, MOVE, CONTROL, ATTACK, TACTIC, INITIATIVE, RECRUIT, PASS,
         SKIP, RG_RESERVE, RG_UNIT, DRAFT]
KIND_ID = {k: i for i, k in enumerate(KINDS)}
N_KINDS = len(KINDS)

PENDING_KINDS = ["", "berserk", "soldat", "merc", "footman", "priest", "rg", "draft"]
PENDING_ID = {k: i for i, k in enumerate(PENDING_KINDS)}

MAX_COINS_EMB = 8

# Rotation de 180° : (colonne x, rang n) -> (6 - x, longueur - n + 1). Involution.
_ROT = [SPEC.index[f"{'abcdefg'[6 - x]}{SPEC.col_lengths[6 - x] - n + 1}"] for x, n in SPEC.cells]
PERM = [list(range(N_CELLS)), _ROT]    # PERM[équipe][case réelle] = case vue
LOC_SET = frozenset(SPEC.locations)

CELL_F = 3    # flotteurs par case
UNIT_F = 12   # flotteurs par jeton d'unité
GLOB_F = 26   # flotteurs du jeton global
ACT_F = 7     # entiers par action : type, pièce, unité, extra, c0, c1, c2


def encode_state(g: Game) -> dict[str, np.ndarray]:
    """Observation du joueur au trait (le jeu doit être en mode 2J)."""
    p = g.to_move
    me, opp = g.players[p], g.players[1 - p]
    t = me.team
    perm = PERM[t]

    cell_i = np.zeros((N_CELLS, 4), np.int64)   # unité, propriétaire, contrôle, pièces
    cell_f = np.zeros((N_CELLS, CELL_F), np.float32)
    for loc in SPEC.locations:
        c = g.control[loc]
        cell_i[perm[loc], 2] = 1 if c is None else (2 if c == t else 3)
    board_coins = [{}, {}]
    n_units = [{}, {}]
    for pos, u in g.board.items():
        v = perm[pos]
        side = 0 if u.owner == p else 1
        cell_i[v, 0] = COIN_ID[u.utype]
        cell_i[v, 1] = 1 + side
        cell_i[v, 3] = min(u.coins, MAX_COINS_EMB)
        cell_f[v, 0] = u.coins / 4.0
        board_coins[side][u.utype] = board_coins[side].get(u.utype, 0) + u.coins
        n_units[side][u.utype] = n_units[side].get(u.utype, 0) + 1
    pending = g.pending[-1] if g.pending else None
    if pending is not None:
        if pending.pos >= 0:
            cell_f[perm[pending.pos], 1] = 1.0
        for pos in pending.positions:
            cell_f[perm[pos], 2] = 1.0

    ui = []
    uf = []
    dr = g.draft if g.in_draft else None
    for side, pl in enumerate((me, opp)):
        bc, nu = board_coins[side], n_units[side]
        hand, bag, dd, du = pl.hand, pl.bag, pl.disc_down, pl.disc_up
        for u in sorted(pl.units):
            ui.append((COIN_ID[u], side))
            h, b, d, up = hand.count(u), bag.count(u), dd.count(u), du.count(u)
            mine = side == 0
            uf.append((h / 2.0 if mine else 0.0, b / 2.0 if mine else 0.0, d / 2.0 if mine else 0.0,
                       (h + b + d) / 3.0, up / 2.0, pl.reserve.get(u, 0) / 3.0,
                       pl.lost.get(u, 0) / 3.0, bc.get(u, 0) / 3.0, float(nu.get(u, 0)),
                       UNITS[u].count / 5.0, (h + b + d + up) / 4.0, 1.0 if (mine and h) else 0.0))
    if dr is not None:
        for u in dr.available:
            ui.append((COIN_ID[u], 2))
            uf.append((0.0,) * 9 + (UNITS[u].count / 5.0, 0.0, 0.0))
    unit_i = np.array(ui, np.int64)
    unit_f = np.array(uf, np.float32)

    glob_i = np.zeros(2, np.int64)
    glob_f = np.zeros(GLOB_F, np.float32)
    if pending is not None:
        glob_i[0] = PENDING_ID[pending.kind]
        if pending.coin is not None:
            glob_i[1] = COIN_ID[pending.coin]
    elif dr is not None:
        glob_i[0] = PENDING_ID["draft"]
    total_markers = 6
    gf = glob_f
    gf[0] = g.markers_left[t] / total_markers
    gf[1] = g.markers_left[1 - t] / total_markers
    gf[2] = 1.0 if g.initiative == p else 0.0
    gf[3] = 1.0 if g.initiative_moved else 0.0
    gf[4] = 1.0 if g.round_first == p else 0.0
    gf[5] = 1.0 if g.can_take_initiative(p) else 0.0
    gf[6] = len(me.hand) / 3.0
    gf[7] = len(opp.hand) / 3.0
    gf[8] = len(me.bag) / 10.0
    gf[9] = len(opp.bag) / 10.0
    gf[10] = (len(me.disc_up) + len(me.disc_down)) / 10.0
    gf[11] = len(opp.disc_down) / 5.0
    gf[12] = len(opp.disc_up) / 10.0
    gf[13] = 1.0 if ROYAL in me.hand else 0.0
    gf[14] = 1.0 if ROYAL in me.bag else 0.0
    gf[15] = 1.0 if (ROYAL in me.disc_down or ROYAL in me.disc_up) else 0.0
    gf[16] = 1.0 if ROYAL in (opp.bag + opp.hand + opp.disc_down) else 0.0
    gf[17] = 1.0 if ROYAL in opp.disc_up else 0.0
    gf[18] = min(g.round, 100) / 50.0
    gf[19] = g.round / g.max_rounds
    gf[20] = 1.0 if g.current == p else 0.0
    gf[21] = sum(1 for loc in SPEC.locations if g.control[loc] is None) / 10.0
    if dr is not None:
        gf[22] = 1.0
        gf[23] = dr.step / len(dr.pool)
        gf[24] = draft_picks_left(g) / 2.0
    gf[25] = 1.0 if g.first_player == p else 0.0
    return {"cell_i": cell_i, "cell_f": cell_f, "unit_i": unit_i, "unit_f": unit_f,
            "glob_i": glob_i, "glob_f": glob_f}


def draft_picks_left(g: Game) -> int:
    """Cartes que le joueur au trait choisit encore dans ce tour de draft (1 ou 2)."""
    from ..units import DRAFT_ORDER
    order, s = DRAFT_ORDER[g.mode], g.draft.step
    n = 1
    while s + n < len(order) and order[s + n] == order[s]:
        n += 1
    return n


def encode_action(a: Action, perm: list[int]) -> tuple[int, ...]:
    cells = [perm[c] for c in a.cells[:3]]
    cells += [NONE_CELL] * (3 - len(cells))
    return (KIND_ID[a.kind],
            COIN_ID[a.coin] if a.coin else 0,
            COIN_ID[a.unit] if a.unit else 0,
            COIN_ID[a.extra] if a.extra else 0,
            *cells)


def encode_actions(g: Game, legal: list[Action]) -> np.ndarray:
    perm = PERM[g.team(g.to_move)]
    return np.array([encode_action(a, perm) for a in legal], np.int64).reshape(-1, ACT_F)


STATE_KEYS = ("cell_i", "cell_f", "unit_i", "unit_f", "glob_i", "glob_f")


def collate(states: list[dict], actions: list[np.ndarray], a_max: int | None = None):
    """Empile des observations et des listes d'actions de tailles variables.

    Renvoie (dict d'entrées numpy, masque (B, A)). Les actions manquantes
    sont remplies avec des zéros et masquées.
    """
    b = len(states)
    out = {k: np.stack([s[k] for s in states]) for k in STATE_KEYS}
    a_len = max(len(a) for a in actions)
    if a_max is not None:
        a_len = max(a_len, a_max)
    acts = np.zeros((b, a_len, ACT_F), np.int64)
    acts[:, :, 4:] = NONE_CELL
    mask = np.zeros((b, a_len), bool)
    for i, a in enumerate(actions):
        acts[i, :len(a)] = a
        mask[i, :len(a)] = True
    out["acts"] = acts
    return out, mask
