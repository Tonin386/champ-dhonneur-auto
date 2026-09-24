"""Moteur de règles de Champ d'honneur (2 et 4 joueurs).

Architecture :
  * `Game` contient l'état complet (information parfaite, côté arbitre).
  * `legal_actions()` renvoie les décisions possibles pour `to_move`.
  * `apply(action)` applique une décision.

Les effets en chaîne (Berserk, Soldat, Moine soldat, Mercenaire, tactique du
Fantassin, défense de la Garde royale) sont gérés par une pile de décisions
en attente (`pending`). Chaque décision est donc atomique et petite, ce qui
simplifie l'interface humaine, la notation et l'apprentissage par renforcement.
"""
from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

from .board import BoardSpec, get_board
from .units import FIRST_GAME, ROYAL, UNITS

# Types d'actions
DEPLOY, BOLSTER = "deploy", "bolster"
MOVE, CONTROL, ATTACK, TACTIC = "move", "control", "attack", "tactic"
INITIATIVE, RECRUIT, PASS = "initiative", "recruit", "pass"
SKIP, RG_RESERVE, RG_UNIT = "skip", "rg_reserve", "rg_unit"

MANEUVERS = (MOVE, CONTROL, ATTACK, TACTIC)
FACE_DOWN = (INITIATIVE, RECRUIT, PASS)

TEAM_NAMES = ["Blanc", "Noir"]


class Action(NamedTuple):
    """Une décision. Tuple nommé : hachage et comparaison en C (très sollicités par la recherche)."""
    kind: str
    coin: str | None = None        # pièce jouée (None = action gratuite / de suivi)
    unit: str | None = None        # type de l'unité qui agit
    cells: tuple[int, ...] = ()    # cases concernées
    extra: str | None = None       # type recruté, ou type de l'unité alliée (Capitaine, Porte étendard)


class Unit:
    __slots__ = ("owner", "utype", "coins")

    def __init__(self, owner: int, utype: str, coins: int = 1):
        self.owner, self.utype, self.coins = owner, utype, coins

    def copy(self) -> "Unit":
        return Unit(self.owner, self.utype, self.coins)

    def __repr__(self) -> str:
        return f"Unit({self.owner},{self.utype},{self.coins})"


class Player:
    __slots__ = ("idx", "team", "units", "bag", "hand", "disc_up", "disc_down", "reserve", "lost")

    def __init__(self, idx: int, team: int, units: list[str]):
        self.idx, self.team, self.units = idx, team, list(units)
        self.bag: list[str] = []
        self.hand: list[str] = []
        self.disc_up: list[str] = []     # défausse face visible (publique)
        self.disc_down: list[str] = []   # défausse face cachée (privée)
        self.reserve: dict[str, int] = {}
        self.lost: dict[str, int] = {u: 0 for u in units}  # pièces remises dans la boîte

    def copy(self) -> "Player":
        p = Player.__new__(Player)
        p.idx, p.team, p.units = self.idx, self.team, self.units
        p.bag, p.hand = self.bag[:], self.hand[:]
        p.disc_up, p.disc_down = self.disc_up[:], self.disc_down[:]
        p.reserve, p.lost = dict(self.reserve), dict(self.lost)
        return p

    def total_coins(self) -> int:
        return len(self.bag) + len(self.hand) + len(self.disc_up) + len(self.disc_down)


class Pending:
    """Décision en attente, empilée par une capacité ou une tactique."""
    __slots__ = ("kind", "player", "pos", "coin", "drawn", "positions")

    def __init__(self, kind: str, player: int, pos: int = -1, positions: tuple[int, ...] = ()):
        self.kind, self.player, self.pos = kind, player, pos
        self.coin: str | None = None
        self.drawn = False
        self.positions = positions

    def copy(self) -> "Pending":
        p = Pending(self.kind, self.player, self.pos, self.positions)
        p.coin, p.drawn = self.coin, self.drawn
        return p


class IllegalAction(Exception):
    pass


class Game:
    def __init__(self, mode: str = "2J", units: list[list[str]] | str | None = None,
                 seed: int | None = None, first: int | None = None, max_rounds: int = 150):
        self.mode = mode
        self.spec: BoardSpec = get_board(mode)
        self.n_players = 2 if mode == "2J" else 4
        self.seed = seed if seed is not None else random.randrange(2**31)
        setup_rng = random.Random(self.seed)
        self.rng = random.Random(self.seed * 7919 + 17)   # pioches uniquement
        self.max_rounds = max_rounds
        per_player = 4 if self.n_players == 2 else 3
        if units == "premiere":
            units = [list(u) for u in FIRST_GAME]
            if first is None:
                first = 0
        if units is None or units == "aleatoire":
            deck = sorted(UNITS)
            setup_rng.shuffle(deck)
            units = [sorted(deck[i * per_player:(i + 1) * per_player]) for i in range(self.n_players)]
        assert len(units) == self.n_players
        self.players = [Player(i, i % 2, units[i]) for i in range(self.n_players)]
        for pl in self.players:
            for u in pl.units:
                pl.bag += [u, u]
                pl.reserve[u] = UNITS[u].count - 2
            pl.bag.append(ROYAL)
        self.board: dict[int, Unit] = {}
        self.loc_set = frozenset(self.spec.locations)
        self.control: dict[int, int | None] = {loc: None for loc in self.spec.locations}
        for team, names in enumerate(self.spec.starts):
            for n in names:
                self.control[self.spec.index[n]] = team
        total_markers = 6 if self.n_players == 2 else 8
        self.markers_left = [total_markers - len(self.spec.starts[t]) for t in range(2)]
        if first is None:
            first = setup_rng.randrange(self.n_players)
        self.first_player = first
        self.initiative = first
        self.initiative_moved = False
        self.round_first = first
        self.round = 0
        self.current = first
        self.pending: list[Pending] = []
        self.winner: int | None = None     # équipe gagnante
        self.done = False
        self.forced: dict[int, deque] = {}  # pioches imposées (rejouer une partie physique)
        self.log: list[tuple[int, int, Action]] = []   # (manche, joueur, action)
        self._start_round()

    # ------------------------------------------------------------------ copie
    def copy(self, rng: random.Random | None = None, log: bool = True) -> "Game":
        """Copie profonde. `rng` : générateur à utiliser tel quel (sinon copie de l'état) ;
        `log=False` : ne pas copier l'historique (copies jetables des recherches)."""
        g = Game.__new__(Game)
        g.mode, g.spec, g.n_players, g.seed = self.mode, self.spec, self.n_players, self.seed
        if rng is None:
            g.rng = random.Random()
            g.rng.setstate(self.rng.getstate())
        else:
            g.rng = rng
        g.max_rounds = self.max_rounds
        g.players = [p.copy() for p in self.players]
        g.board = {k: u.copy() for k, u in self.board.items()}
        g.loc_set = self.loc_set
        g.control = dict(self.control)
        g.markers_left = self.markers_left[:]
        g.first_player, g.initiative = self.first_player, self.initiative
        g.initiative_moved, g.round_first = self.initiative_moved, self.round_first
        g.round, g.current = self.round, self.current
        g.pending = [p.copy() for p in self.pending]
        g.winner, g.done = self.winner, self.done
        g.forced = {k: deque(v) for k, v in self.forced.items()}
        g.log = self.log[:] if log else []
        return g

    # ------------------------------------------------------------ accesseurs
    def team(self, p: int) -> int:
        return self.players[p].team

    @property
    def to_move(self) -> int:
        return self.pending[-1].player if self.pending else self.current

    def units_of(self, p: int, utype: str) -> list[int]:
        return [pos for pos, u in self.board.items() if u.owner == p and u.utype == utype]

    def is_enemy(self, p: int, unit: Unit) -> bool:
        return self.players[unit.owner].team != self.players[p].team

    # ---------------------------------------------------------------- pioche
    def force_draws(self, p: int, coins: list[str]) -> None:
        self.forced.setdefault(p, deque()).extend(coins)

    def _draw(self, p: int) -> str | None:
        pl = self.players[p]
        if not pl.bag:
            pl.bag = pl.disc_up + pl.disc_down
            pl.disc_up, pl.disc_down = [], []
        if not pl.bag:
            return None
        fq = self.forced.get(p)
        if fq:
            c = fq.popleft()
            if c not in pl.bag:
                raise IllegalAction(f"Pioche imposée impossible : {c} n'est pas dans le sac")
            pl.bag.remove(c)
            return c
        i = self.rng.randrange(len(pl.bag))
        pl.bag[i], pl.bag[-1] = pl.bag[-1], pl.bag[i]
        return pl.bag.pop()

    def _start_round(self) -> None:
        self.round += 1
        if self.round > self.max_rounds:
            self.done, self.winner = True, None
            return
        self.initiative_moved = False
        self.round_first = self.initiative
        for k in range(self.n_players):
            p = (self.round_first + k) % self.n_players
            for _ in range(3):
                c = self._draw(p)
                if c is None:
                    break
                self.players[p].hand.append(c)
        self.current = self.round_first

    def _advance(self) -> None:
        for k in range(1, self.n_players + 1):
            p = (self.current + k) % self.n_players
            if self.players[p].hand:
                self.current = p
                return
        self._start_round()

    # ------------------------------------------------------- actions légales
    def legal_actions(self) -> list[Action]:
        if self.done:
            return []
        if self.pending:
            return self._pending_actions(self.pending[-1])
        p = self.current
        out: list[Action] = []
        for coin in sorted(set(self.players[p].hand)):
            out += self._coin_actions(p, coin)
        return out

    def can_take_initiative(self, p: int) -> bool:
        return (not self.initiative_moved and self.round_first != p
                and self.team(self.initiative) != self.team(p))

    def _coin_actions(self, p: int, coin: str) -> list[Action]:
        pl = self.players[p]
        acts = [Action(PASS, coin)]
        for u in pl.units:
            if pl.reserve.get(u, 0) > 0:
                acts.append(Action(RECRUIT, coin, extra=u))
        if self.can_take_initiative(p):
            acts.append(Action(INITIATIVE, coin))
        if coin == ROYAL:
            for pos in self.units_of(p, "G"):
                for nb in self.spec.neighbors[pos]:
                    if nb not in self.board:
                        acts.append(Action(TACTIC, ROYAL, "G", (pos, nb)))
            return acts
        u = coin
        positions = self.units_of(p, u)
        if len(positions) < UNITS[u].max_units:
            for c in sorted(self._deploy_cells(p, u)):
                acts.append(Action(DEPLOY, coin, u, (c,)))
        for pos in positions:
            acts.append(Action(BOLSTER, coin, u, (pos,)))
            acts += self._maneuvers(p, pos, coin)
            acts += self._tactics(p, pos, coin)
        if u == "F" and len(positions) == 2:
            acts.append(Action(TACTIC, coin, "F", ()))
        return acts

    def _deploy_cells(self, p: int, u: str) -> set[int]:
        team = self.team(p)
        cells = {loc for loc in self.spec.locations
                 if self.control[loc] == team and loc not in self.board}
        if u == "E":
            for pos, unit in self.board.items():
                if self.team(unit.owner) == team:
                    cells.update(nb for nb in self.spec.neighbors[pos] if nb not in self.board)
        return cells

    def _can_attack(self, attacker: Unit, target: Unit) -> bool:
        return not (target.utype == "N" and attacker.coins < 2)

    def _enemy_at(self, p: int, cell: int) -> Unit | None:
        t = self.board.get(cell)
        return t if t is not None and self.is_enemy(p, t) else None

    def _maneuvers(self, p: int, pos: int, coin: str | None, moves_only: bool = False) -> list[Action]:
        unit = self.board[pos]
        u = unit.utype
        out = [Action(MOVE, coin, u, (pos, nb)) for nb in self.spec.neighbors[pos] if nb not in self.board]
        if moves_only:
            return out
        team = self.team(p)
        if pos in self.loc_set and self.control[pos] != team and self.markers_left[team] > 0:
            out.append(Action(CONTROL, coin, u, (pos,)))
        if UNITS[u].normal_attack:
            for nb in self.spec.neighbors[pos]:
                t = self._enemy_at(p, nb)
                if t is not None and self._can_attack(unit, t):
                    out.append(Action(ATTACK, coin, u, (pos, nb)))
        return out

    def _tactics(self, p: int, pos: int, coin: str) -> list[Action]:
        unit = self.board[pos]
        u = unit.utype
        sp = self.spec
        out: list[Action] = []
        if u == "C":
            for mid in sp.neighbors[pos]:
                if mid in self.board:
                    continue
                for tgt in sp.neighbors[mid]:
                    t = self._enemy_at(p, tgt)
                    if t is not None and self._can_attack(unit, t):
                        out.append(Action(TACTIC, coin, u, (pos, mid, tgt)))
        elif u == "H":
            dests = set()
            for mid in sp.neighbors[pos]:
                if mid in self.board:
                    continue
                for d in sp.neighbors[mid]:
                    if d not in self.board and sp.dist[pos][d] == 2:
                        dests.add(d)
            out += [Action(TACTIC, coin, u, (pos, d)) for d in sorted(dests)]
        elif u == "A":
            for tgt in sp.ring2[pos]:
                t = self._enemy_at(p, tgt)
                if t is not None and self._can_attack(unit, t):
                    out.append(Action(TACTIC, coin, u, (pos, tgt)))
        elif u == "X":
            for mid, tgt in sp.lines2[pos]:
                if mid in self.board:
                    continue
                t = self._enemy_at(p, tgt)
                if t is not None and self._can_attack(unit, t):
                    out.append(Action(TACTIC, coin, u, (pos, tgt)))
        elif u == "L":
            for ray in sp.rays[pos]:
                if not ray or ray[0] in self.board:
                    continue
                for dist in (1, 2):
                    if len(ray) <= dist:
                        break
                    if dist == 2 and ray[1] in self.board:
                        break
                    t = self._enemy_at(p, ray[dist])
                    if t is not None and self._can_attack(unit, t):
                        out.append(Action(TACTIC, coin, u, (pos, ray[dist - 1], ray[dist])))
        elif u == "K":
            team = self.team(p)
            for ally_pos in sp.within2[pos]:
                ally = self.board.get(ally_pos)
                if ally is None or self.team(ally.owner) != team or not UNITS[ally.utype].normal_attack:
                    continue
                for tgt in sp.neighbors[ally_pos]:
                    t = self._enemy_at(p, tgt)
                    if t is not None and self._can_attack(ally, t):
                        out.append(Action(TACTIC, coin, u, (pos, ally_pos, tgt), ally.utype))
        elif u == "D":
            team = self.team(p)
            for ally_pos in sp.within2[pos]:
                ally = self.board.get(ally_pos)
                if ally is None or self.team(ally.owner) != team:
                    continue
                for to in sp.neighbors[ally_pos]:
                    if to not in self.board and sp.dist[pos][to] <= 2:
                        out.append(Action(TACTIC, coin, u, (pos, ally_pos, to), ally.utype))
        return out

    def _pending_actions(self, pd: Pending) -> list[Action]:
        k = pd.kind
        if k == "rg":
            return [Action(RG_UNIT), Action(RG_RESERVE)]
        if k == "priest":
            return self._coin_actions(pd.player, pd.coin)
        skip = [Action(SKIP)]
        if k == "berserk" or k == "merc":
            return skip + self._maneuvers(pd.player, pd.pos, None)
        if k == "soldat":
            return skip + self._maneuvers(pd.player, pd.pos, None, moves_only=True)
        if k == "footman":
            out = skip
            for pos in pd.positions:
                out = out + self._maneuvers(pd.player, pos, None)
            return out
        raise RuntimeError(k)

    # ----------------------------------------------------------- application
    def apply(self, a: Action) -> None:
        if self.done:
            raise IllegalAction("La partie est terminée")
        p = self.to_move
        self.log.append((self.round, p, a))
        if self.pending:
            self._apply_pending(self.pending.pop(), a)
        else:
            self._apply_main(p, a, from_hand=True)
        self._settle()

    def apply_checked(self, a: Action) -> None:
        if a not in self.legal_actions():
            raise IllegalAction(f"Action illégale : {a}")
        self.apply(a)

    def _settle(self) -> None:
        while not self.done and self.pending:
            pd = self.pending[-1]
            if self._pending_valid(pd):
                return
            self.pending.pop()
        if not self.done and not self.pending:
            self._advance()

    def _pending_valid(self, pd: Pending) -> bool:
        k = pd.kind
        if k == "rg":
            return True
        if k == "priest":
            if not pd.drawn:
                pd.coin = self._draw(pd.player)
                pd.drawn = True
            return pd.coin is not None
        if k == "footman":
            pd.positions = tuple(pos for pos in pd.positions
                                 if (u := self.board.get(pos)) and u.utype == "F" and u.owner == pd.player)
            return len(self._pending_actions(pd)) > 1
        unit = self.board.get(pd.pos)
        want = {"berserk": "B", "soldat": "S", "merc": "M"}[k]
        if unit is None or unit.utype != want or unit.owner != pd.player:
            return False
        if k == "berserk" and unit.coins < 2:
            return False
        return len(self._pending_actions(pd)) > 1

    def _apply_main(self, p: int, a: Action, from_hand: bool) -> None:
        pl = self.players[p]
        coin = a.coin
        if from_hand:
            pl.hand.remove(coin)
        k = a.kind
        if k == PASS:
            pl.disc_down.append(coin)
        elif k == INITIATIVE:
            pl.disc_down.append(coin)
            self.initiative = p
            self.initiative_moved = True
        elif k == RECRUIT:
            pl.disc_down.append(coin)
            pl.reserve[a.extra] -= 1
            pl.disc_up.append(a.extra)
            if a.extra == "M":
                for pos in self.units_of(p, "M"):
                    self.pending.append(Pending("merc", p, pos))
        elif k == DEPLOY:
            self.board[a.cells[0]] = Unit(p, a.unit, 1)
        elif k == BOLSTER:
            self.board[a.cells[0]].coins += 1
        elif k in MANEUVERS:
            pl.disc_up.append(coin)
            self._do_maneuver(p, a)
        else:
            raise IllegalAction(str(a))

    def _move(self, frm: int, to: int) -> None:
        self.board[to] = self.board.pop(frm)

    def _trigger(self, pos: int, kind: str) -> None:
        """Empile les capacités déclenchées par une manœuvre de l'unité en `pos`."""
        unit = self.board.get(pos)
        if unit is None:
            return
        if unit.utype == "B":
            self.pending.append(Pending("berserk", unit.owner, pos))
        elif unit.utype == "S" and kind == ATTACK:
            self.pending.append(Pending("soldat", unit.owner, pos))
        elif unit.utype == "R" and kind in (ATTACK, CONTROL):
            self.pending.append(Pending("priest", unit.owner))

    def _do_maneuver(self, p: int, a: Action) -> None:
        k, c = a.kind, a.cells
        if k == MOVE:
            self._move(c[0], c[1])
            self._trigger(c[1], MOVE)
        elif k == CONTROL:
            self._trigger(c[0], CONTROL)
            self._control(p, c[0])
        elif k == ATTACK:
            self._trigger(c[0], ATTACK)
            self._attack(c[0], c[1], adjacent=True)
        elif k == TACTIC:
            u = a.unit
            if u == "C":
                self._move(c[0], c[1])
                self._trigger(c[1], ATTACK)
                self._attack(c[1], c[2], adjacent=True)
            elif u in ("H", "G"):
                self._move(c[0], c[1])
                self._trigger(c[1], MOVE)
            elif u in ("A", "X"):
                self._trigger(c[0], ATTACK)
                self._attack(c[0], c[1], adjacent=False)
            elif u == "L":
                self._move(c[0], c[1])
                self._trigger(c[1], ATTACK)
                self._attack(c[1], c[2], adjacent=True)
            elif u == "K":
                self._trigger(c[1], ATTACK)
                self._attack(c[1], c[2], adjacent=True)
            elif u == "D":
                self._move(c[1], c[2])
                self._trigger(c[2], MOVE)
            elif u == "F":
                self.pending.append(Pending("footman", p, positions=tuple(self.units_of(p, "F"))))
            else:
                raise IllegalAction(str(a))
        else:
            raise IllegalAction(str(a))

    def _control(self, p: int, pos: int) -> None:
        team = self.team(p)
        prev = self.control[pos]
        if prev is not None:
            self.markers_left[prev] += 1
        self.control[pos] = team
        self.markers_left[team] -= 1
        if self.markers_left[team] == 0:
            self.done, self.winner = True, team
            self.pending.clear()

    def _remove_coin(self, pos: int) -> None:
        unit = self.board[pos]
        unit.coins -= 1
        self.players[unit.owner].lost[unit.utype] += 1
        if unit.coins <= 0:
            del self.board[pos]

    def _attack(self, att_pos: int, tgt_pos: int, adjacent: bool) -> None:
        tgt = self.board[tgt_pos]
        if adjacent and tgt.utype == "P":
            self._remove_coin(att_pos)
        if tgt.utype == "G" and self.players[tgt.owner].reserve.get("G", 0) > 0:
            self.pending.append(Pending("rg", tgt.owner, tgt_pos))
        else:
            self._remove_coin(tgt_pos)

    def _apply_pending(self, pd: Pending, a: Action) -> None:
        k = pd.kind
        if k == "rg":
            if a.kind == RG_RESERVE:
                pl = self.players[pd.player]
                pl.reserve["G"] -= 1
                pl.lost["G"] += 1
            elif a.kind == RG_UNIT:
                self._remove_coin(pd.pos)
            else:
                raise IllegalAction(str(a))
            return
        if k == "priest":
            self._apply_main(pd.player, a, from_hand=False)
            return
        if a.kind == SKIP:
            return
        if k == "berserk":
            unit = self.board[pd.pos]
            unit.coins -= 1
            self.players[pd.player].disc_up.append("B")
            self._do_maneuver(pd.player, a)
        elif k == "soldat":
            self._move(a.cells[0], a.cells[1])
        elif k == "merc":
            self._do_maneuver(pd.player, a)
        elif k == "footman":
            rest = tuple(x for x in pd.positions if x != a.cells[0])
            if rest:
                self.pending.append(Pending("footman", pd.player, positions=rest))
            self._do_maneuver(pd.player, a)

    # ---------------------------------------------------------- informations
    def pending_label(self) -> str:
        if not self.pending:
            return ""
        return {
            "berserk": "Berserk : défausser une pièce de sa pile pour manœuvrer à nouveau ?",
            "soldat": "Soldat : se déplacer d'une case après l'attaque ?",
            "merc": "Mercenaire : manœuvre gratuite ?",
            "footman": "Fantassin : manœuvre du Fantassin suivant",
            "priest": "Moine soldat : utilisez la pièce piochée",
            "rg": "Garde royale attaquée : retirer une pièce de la pile ou de la réserve ?",
        }[self.pending[-1].kind]

    def result_label(self) -> str:
        if not self.done:
            return "*"
        if self.winner is None:
            return "1/2-1/2"
        return "1-0" if self.winner == 0 else "0-1"

    def determinize(self, observer: int, rng: random.Random | None = None,
                    rapide: bool = False) -> "Game":
        """Copie où l'information cachée à `observer` est ré-échantillonnée.

        Pour chaque adversaire, l'ensemble {sac + main + défausse cachée} est
        connu en tant que multiensemble (tout le reste est public) : on le
        redistribue au hasard en conservant la taille de chaque zone. Les
        propres sacs de l'observateur sont déjà aléatoires ; on réinitialise
        seulement le générateur de pioche.
        """
        rng = rng or random.Random()
        if rapide:
            # copie jetable pour la recherche : pioches tirées directement dans `rng`
            # (évite l'initialisation coûteuse d'un générateur), sans historique
            g = self.copy(rng=rng, log=False)
        else:
            g = self.copy()
            g.rng = random.Random(rng.randrange(2**62))
        g.forced = {}
        for pl in g.players:
            if pl.idx == observer:
                continue
            pool = pl.bag + pl.hand + pl.disc_down
            rng.shuffle(pool)
            nh, nd = len(pl.hand), len(pl.disc_down)
            pl.hand = pool[:nh]
            pl.disc_down = pool[nh:nh + nd]
            pl.bag = pool[nh + nd:]
        return g
