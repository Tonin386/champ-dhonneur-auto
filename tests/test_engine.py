import random

import pytest

from champ_dhonneur.bots import GreedyBot, RandomBot
from champ_dhonneur.engine import (ATTACK, DEPLOY, RG_RESERVE, SKIP, TACTIC, Action, Game, Unit)
from champ_dhonneur.notation import action_str, export_record, find_action, import_record, play_token
from champ_dhonneur.units import UNITS


def conserved(g: Game) -> None:
    for pl in g.players:
        on_board = sum(u.coins for u in g.board.values() if u.owner == pl.idx)
        limbo = sum(1 for pd in g.pending if pd.kind == "priest" and pd.player == pl.idx and pd.coin)
        total = (pl.total_coins() + on_board + sum(pl.reserve.values()) + sum(pl.lost.values()) + limbo)
        assert total == sum(UNITS[u].count for u in pl.units) + 1
        assert all(v >= 0 for v in pl.reserve.values())
    assert all(u.coins >= 1 for u in g.board.values())
    for t in range(2):
        placed = sum(1 for v in g.control.values() if v == t)
        assert placed + g.markers_left[t] == (6 if g.n_players == 2 else 8)


def play_random(seed: int, mode: str = "2J") -> Game:
    g = Game(mode, seed=seed)
    rng = random.Random(seed)
    while not g.done:
        legal = g.legal_actions()
        assert legal, "aucune action légale"
        g.apply(rng.choice(legal))
        conserved(g)
    return g


@pytest.mark.parametrize("seed", range(40))
def test_random_games_2p(seed):
    play_random(seed)


@pytest.mark.parametrize("seed", range(10))
def test_random_games_4p(seed):
    play_random(seed, "4J")


@pytest.mark.parametrize("seed", range(15))
def test_record_roundtrip(seed):
    g = play_random(seed)
    rec = export_record(g)
    g2 = import_record(rec)
    assert g2.log == g.log or export_record(g2) == rec
    assert g2.board.keys() == g.board.keys()
    assert g2.control == g.control
    assert g2.result_label() == g.result_label()


def empty_game(units, first=0):
    g = Game("2J", units, seed=0, first=first)
    return g


def put(g, cell, owner, u, coins=1):
    g.board[g.spec.index[cell]] = Unit(owner, u, coins)
    g.players[owner].reserve[u] -= coins


def names(g, acts):
    return {action_str(g, a) for a in acts}


def with_hand(g, p, coins):
    pl = g.players[p]
    pl.bag += pl.hand
    pl.hand = []
    for c in coins:
        pl.bag.remove(c)
        pl.hand.append(c)


def test_knight_needs_bolstered_attacker():
    g = empty_game([["S", "P", "X", "H"], ["N", "C", "L", "E"]])
    put(g, "d3", 0, "S")
    put(g, "d4", 1, "N")
    with_hand(g, 0, ["S"])
    assert "Sd3xd4" not in names(g, g.legal_actions())
    g.board[g.spec.index["d3"]].coins = 2
    assert "Sd3xd4" in names(g, g.legal_actions())


@pytest.mark.parametrize("coins,allowed", [(2, False), (3, True)])
def test_berserk_extra_maneuver_knight_after_payment(coins, allowed):
    # la pièce du Berserk est défaussée avant sa manœuvre supplémentaire : il doit rester renforcé
    g = empty_game([["B", "P", "X", "H"], ["N", "C", "L", "E"]])
    put(g, "d2", 0, "B", coins)
    put(g, "d4", 1, "N")
    with_hand(g, 0, ["B"])
    d2, d3, d4 = (g.spec.index[c] for c in ("d2", "d3", "d4"))
    g.apply(next(a for a in g.legal_actions() if a.kind == "move" and a.cells == (d2, d3)))
    assert g.pending and g.pending[-1].kind == "berserk"
    attaque = [a for a in g.legal_actions() if a.kind == ATTACK and a.cells == (d3, d4)]
    assert bool(attaque) == allowed
    if allowed:
        g.apply(attaque[0])
        assert g.board[d3].coins == coins - 1 and d4 not in g.board


def test_archer_only_tactic_and_range():
    g = empty_game([["A", "P", "X", "H"], ["S", "C", "L", "E"]])
    put(g, "d3", 0, "A")
    put(g, "d4", 1, "S")
    put(g, "d5", 1, "C")
    with_hand(g, 0, ["A"])
    n = names(g, g.legal_actions())
    assert "Ad3xd4" not in n          # pas d'attaque adjacente
    assert "Ad3:xd5" in n             # à 2 cases, même par-dessus une unité
    assert "Ad3:xd4" not in n


def test_crossbow_line_blocked():
    g = empty_game([["X", "P", "A", "H"], ["S", "C", "L", "E"]])
    put(g, "d3", 0, "X")
    put(g, "d4", 1, "S")
    put(g, "d5", 1, "C")
    with_hand(g, 0, ["X"])
    n = names(g, g.legal_actions())
    assert "Xd3:xd5" not in n and "Xd3xd4" in n


def test_pikeman_retaliates():
    g = empty_game([["S", "A", "X", "H"], ["P", "C", "L", "E"]])
    put(g, "d3", 0, "S", 2)
    put(g, "d4", 1, "P")
    with_hand(g, 0, ["S"])
    g.apply(find_action(g, "Sd3xd4"))
    assert g.board[g.spec.index["d3"]].coins == 1
    assert g.spec.index["d4"] not in g.board


def test_royal_guard_reserve_defense_and_seal_tactic():
    g = empty_game([["S", "A", "X", "H"], ["G", "C", "L", "E"]])
    put(g, "d3", 0, "S")
    put(g, "d4", 1, "G")
    with_hand(g, 0, ["S"])
    g.apply(find_action(g, "Sd3xd4"))
    assert g.to_move == 1 and g.pending[-1].kind == "rg"
    before = g.players[1].reserve["G"]
    g.apply(Action(RG_RESERVE))
    assert g.players[1].reserve["G"] == before - 1
    assert g.board[g.spec.index["d4"]].coins == 1


def test_berserk_chain_and_soldier_move():
    g = empty_game([["B", "S", "X", "H"], ["A", "C", "L", "E"]])
    put(g, "d3", 0, "B", 3)
    put(g, "d5", 1, "A", 2)
    with_hand(g, 0, ["B"])
    play_token(g, "Bd3-d4>Bd4xd5>Bd4xd5")
    assert g.spec.index["d5"] not in g.board
    assert g.board[g.spec.index["d4"]].coins == 1


def test_warrior_priest_draws():
    g = empty_game([["R", "S", "X", "H"], ["A", "C", "L", "E"]])
    put(g, "b1", 0, "R")
    g.control[g.spec.index["b1"]] = None
    g.markers_left[0] += 1
    with_hand(g, 0, ["R"])
    g.apply(find_action(g, "Rb1^"))
    assert g.pending and g.pending[-1].kind == "priest"
    assert all(a.kind != SKIP for a in g.legal_actions())


def test_footman_two_units_and_tactic():
    g = empty_game([["F", "S", "X", "H"], ["A", "C", "L", "E"]])
    put(g, "b1", 0, "F")
    with_hand(g, 0, ["F"])
    assert any(a.kind == DEPLOY for a in g.legal_actions())  # second Fantassin
    put(g, "e1", 0, "F")
    n = names(g, g.legal_actions())
    assert "F:" in n
    play_token(g, "F:>Fb1-b2>Fe1-e2")
    assert {g.spec.names[p] for p in g.units_of(0, "F")} == {"b2", "e2"}


def test_lancer_straight_line():
    g = empty_game([["L", "S", "X", "H"], ["A", "C", "P", "E"]])
    put(g, "d1", 0, "L")
    put(g, "d4", 1, "A")
    with_hand(g, 0, ["L"])
    n = names(g, g.legal_actions())
    assert "Ld1:d3xd4" in n
    assert not any(s.startswith("Ld1x") for s in n)


def test_scout_deploy_adjacent():
    g = empty_game([["E", "S", "X", "H"], ["A", "C", "L", "P"]])
    put(g, "d4", 0, "S")
    with_hand(g, 0, ["E"])
    n = names(g, g.legal_actions())
    assert "E@d5" in n and "E@b1" in n


def test_initiative_once_per_round():
    g = Game("2J", "premiere", seed=5)
    assert g.to_move == 0
    assert "I" not in {action_str(g, a, hidden=False) for a in g.legal_actions()}
    g.apply(g.legal_actions()[0])          # Blanc passe
    acts = g.legal_actions()
    ini = [a for a in acts if a.kind == "initiative"]
    assert ini
    g.apply(ini[0])
    assert g.initiative == 1
    assert not [a for a in g.legal_actions() if a.kind == "initiative"]


def test_bots_play_full_game():
    g = Game("2J", seed=11)
    bots = [GreedyBot(1), RandomBot(2)]
    while not g.done:
        g.apply(bots[g.team(g.to_move)].choose(g))
    assert g.done
