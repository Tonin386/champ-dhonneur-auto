"""Notation Champ d'Honneur (NCH), inspirée de la notation algébrique des échecs.

Cases : lettre de colonne + rang dans la colonne (ex. « c3 »).
Unités : une majuscule (A Archer, B Berserk, C Cavalerie, D Porte étendard,
E Éclaireur, F Fantassin, G Garde royale, H Cavalerie légère, K Capitaine,
L Lancier, M Mercenaire, N Chevalier, P Piquier, R Moine soldat, S Soldat,
X Arbalétrier) ; « * » = Sceau royal.

    Déployer            S@b1        (comme le parachutage au shogi / crazyhouse)
    Renforcer           S+b1
    Déplacer            Sb1-b2
    Attaquer            Sb2xb3
    Contrôler           Sb2^        (on « plante le drapeau »)
    Tactique            <unité><case>:<détail>   ex. Cb2:b3xb4, Hb2:b4, Ab2:xc4,
                        Lb2:b4xb5, Kb2:Sc3xc4, Db2:Sc3-c4, F:, *:Gb2-b3
    Prendre l'initiative I
    Recruter            $P          (recrute un Piquier)
    Passer              --
    Pièce cachée        {X}  ajouté aux actions face cachée dans un relevé complet
                        (I{S}, $P{X}, --{*}) ; supprimé dans le relevé public.
    Suite de coup       >…   effet enchaîné : Berserk, Soldat, Mercenaire,
                        Moine soldat, Fantassin (ex. Bc3-c4>Bc4xc5)
    Garde royale        (R)  la défense retire une pièce de la réserve
    Refus d'une option  0    (omis dans les relevés)
"""
from __future__ import annotations

import re

from .engine import (ATTACK, BOLSTER, CONTROL, DEPLOY, INITIATIVE, MOVE, PASS, RECRUIT,
                     RG_RESERVE, RG_UNIT, SKIP, TACTIC, TEAM_NAMES, Action, Game, IllegalAction)
from .units import ROYAL


def action_str(game: Game, a: Action, hidden: bool = True) -> str:
    n = game.spec.names
    c = [n[i] for i in a.cells]
    k = a.kind
    tag = f"{{{a.coin}}}" if hidden and a.coin else ""
    if k == PASS:
        return "--" + tag
    if k == INITIATIVE:
        return "I" + tag
    if k == RECRUIT:
        return f"${a.extra}" + tag
    if k == DEPLOY:
        return f"{a.unit}@{c[0]}"
    if k == BOLSTER:
        return f"{a.unit}+{c[0]}"
    if k == MOVE:
        return f"{a.unit}{c[0]}-{c[1]}"
    if k == ATTACK:
        return f"{a.unit}{c[0]}x{c[1]}"
    if k == CONTROL:
        return f"{a.unit}{c[0]}^"
    if k == SKIP:
        return "0"
    if k == RG_RESERVE:
        return "(R)"
    if k == RG_UNIT:
        return "(U)"
    if k == TACTIC:
        u = a.unit
        if u == "C" or u == "L":
            return f"{u}{c[0]}:{c[1]}x{c[2]}"
        if u == "H":
            return f"H{c[0]}:{c[1]}"
        if u in ("A", "X"):
            return f"{u}{c[0]}:x{c[1]}"
        if u == "K":
            return f"K{c[0]}:{a.extra}{c[1]}x{c[2]}"
        if u == "D":
            return f"D{c[0]}:{a.extra}{c[1]}-{c[2]}"
        if u == "F":
            return "F:"
        if u == "G":
            return f"{ROYAL}:G{c[0]}-{c[1]}"
    raise ValueError(a)


def describe(game: Game, a: Action) -> str:
    """Traduction en français courant d'une action."""
    from .units import unit_name
    n = game.spec.names
    c = [n[i] for i in a.cells]
    k = a.kind
    un = unit_name(a.unit) if a.unit else ""
    coin = f" (pièce {unit_name(a.coin)})" if a.coin else ""
    return {
        PASS: lambda: f"Passer{coin}",
        INITIATIVE: lambda: f"Prendre l'initiative{coin}",
        RECRUIT: lambda: f"Recruter : {unit_name(a.extra or '')}{coin}",
        DEPLOY: lambda: f"Déployer {un} en {c[0]}",
        BOLSTER: lambda: f"Renforcer {un} en {c[0]}",
        MOVE: lambda: f"{un} : {c[0]} → {c[1]}",
        ATTACK: lambda: f"{un} en {c[0]} attaque {c[1]}",
        CONTROL: lambda: f"{un} prend le contrôle de {c[0]}",
        SKIP: lambda: "Ne rien faire",
        RG_RESERVE: lambda: "Retirer une pièce de la réserve",
        RG_UNIT: lambda: "Retirer une pièce de la Garde royale",
        TACTIC: lambda: f"Tactique {un} : {action_str(game, a)}",
    }[k]()


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lstrip(">")


class AmbiguousNotation(Exception):
    def __init__(self, text: str, candidates: list[Action]):
        super().__init__(text)
        self.candidates = candidates


def find_action(game: Game, text: str) -> Action:
    """Retrouve l'action légale correspondant à une chaîne de notation.

    Le relevé complet (avec {pièce}) est exact ; la forme publique est
    acceptée si elle n'est pas ambiguë.
    """
    t = normalize(text)
    legal = game.legal_actions()
    for a in legal:
        if action_str(game, a, hidden=True) == t:
            return a
    matches = [a for a in legal if action_str(game, a, hidden=False) == t]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise AmbiguousNotation(t, matches)
    raise IllegalAction(f"Coup illégal ou inconnu : {text}")


# ------------------------------------------------------------ relevés de partie
def export_record(game: Game, hidden: bool = True, headers: dict | None = None) -> str:
    h = {
        "Jeu": "Champ d'honneur",
        "Mode": game.mode,
        "Graine": str(game.seed),
        "Initiative": str(game.first_player),
    }
    for p in game.players:
        h[f"Unites{p.idx}"] = " ".join(p.units)
    if game.max_rounds != 150:
        h["MaxManches"] = str(game.max_rounds)
    h["Resultat"] = game.result_label()
    if headers:
        h.update(headers)
    lines = [f'[{k} "{v}"]' for k, v in h.items()]
    lines.append("")
    tokens: list[str] = []
    cur_round = 0
    replay = Game(game.mode, [p.units for p in game.players], seed=game.seed,
                  first=game.first_player, max_rounds=game.max_rounds)
    body: list[str] = []
    for rnd, _p, a in game.log:
        is_main = not replay.pending
        s = action_str(replay, a, hidden=hidden)
        if is_main:
            if rnd != cur_round:
                if tokens:
                    body.append(" ".join(tokens))
                tokens = [f"{rnd}."]
                cur_round = rnd
            tokens.append(s)
        elif a.kind in (SKIP, RG_UNIT):
            pass
        elif a.kind == RG_RESERVE:
            tokens[-1] += "(R)"
        else:
            tokens[-1] += ">" + s
        replay.apply(a)
    if tokens:
        body.append(" ".join(tokens))
    body.append(game.result_label())
    return "\n".join(lines + body) + "\n"


def parse_headers(text: str) -> dict[str, str]:
    return dict(re.findall(r'\[(\w+)\s+"([^"]*)"\]', text))


def import_record(text: str) -> Game:
    h = parse_headers(text)
    mode = h.get("Mode", "2J")
    n = 2 if mode == "2J" else 4
    units = [h[f"Unites{i}"].split() for i in range(n)]
    game = Game(mode, units, seed=int(h["Graine"]), first=int(h.get("Initiative", 0)),
                max_rounds=int(h.get("MaxManches", 150)))
    body = re.sub(r"\[[^\]]*\]", " ", text)
    for tok in body.split():
        if re.fullmatch(r"\d+\.", tok) or tok in ("1-0", "0-1", "1/2-1/2", "*"):
            continue
        play_token(game, tok)
    return game


def _flush_optional(game: Game) -> None:
    """Refuse les options en attente (omises dans les relevés)."""
    while game.pending and not game.done:
        legal = game.legal_actions()
        if Action(SKIP) in legal:
            game.apply(Action(SKIP))
        elif Action(RG_UNIT) in legal:
            game.apply(Action(RG_UNIT))
        else:
            return


def play_token(game: Game, token: str) -> list[Action]:
    """Joue un coup complet (action principale et suites « > »)."""
    played = []
    parts = token.split(">")
    for i, part in enumerate(parts):
        if i == 0:
            _flush_optional(game)
        elif game.pending and game.pending[-1].kind == "rg":
            game.apply(Action(RG_UNIT))
        rg = part.endswith("(R)") or part.endswith("(U)")
        choice = part[-3:] if rg else None
        core = part[:-3] if rg else part
        if core:
            try:
                a = find_action(game, core)
            except IllegalAction:
                if Action(SKIP) in game.legal_actions():
                    game.apply(Action(SKIP))
                    a = find_action(game, core)
                else:
                    raise
            game.apply(a)
            played.append(a)
        if choice:
            a = Action(RG_RESERVE if choice == "(R)" else RG_UNIT)
            game.apply_checked(a)
            played.append(a)
    if game.pending and game.pending[-1].kind == "rg":
        game.apply(Action(RG_UNIT))
    return played


def team_name(t: int | None) -> str:
    return "Nulle" if t is None else TEAM_NAMES[t]
