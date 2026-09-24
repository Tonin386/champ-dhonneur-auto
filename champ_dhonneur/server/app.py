"""Serveur web : API JSON, spectateur de l'entraînement (/) et partie contre les bots (/jouer).

L'interface du spectateur est une application React compilée dans server/web (voir web/ à la
racine du dépôt, « make front ») ; la page de jeu est un fichier statique unique."""
from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..bots import make_bot
from ..engine import Game, IllegalAction
from ..notation import AmbiguousNotation, action_str, describe, export_record, find_action
from ..units import UNITS
from . import spectateur

STATIC = Path(__file__).parent / "static"
WEB = Path(__file__).parent / "web"          # spectateur compilé (make front)
app = FastAPI(title="Champ d'honneur")
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.include_router(spectateur.router)
app.mount("/img", StaticFiles(directory=STATIC / "img"), name="img")
if (WEB / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")


class Session:
    def __init__(self, game: Game, roles: list[str]):
        self.game = game
        self.roles = roles
        self.bots = {i: make_bot(r, seed=i) for i, r in enumerate(roles) if r != "humain"}
        self.lock = threading.Lock()
        self.last: list[str] = []
        self.last_cells: list[int] = []

    @property
    def humans(self) -> set[int]:
        return {i for i in range(len(self.roles)) if i not in self.bots}


SESSIONS: dict[str, Session] = {}


class NewGame(BaseModel):
    mode: str = "2J"
    unites: str = "aleatoire"
    roles: list[str] = ["humain", "mcts"]
    graine: int | None = None


class Play(BaseModel):
    notation: str | None = None
    index: int | None = None


def parse_units(s: str):
    if s in ("aleatoire", "premiere"):
        return s
    return [list(g) for g in s.upper().split("/")]


def state_json(s: Session, reveal: bool = False) -> dict:
    g = s.game
    sp = g.spec
    tm = g.to_move
    humans = s.humans
    if reveal or not humans:
        visible = set(range(g.n_players))
    elif len(humans) == 1:
        visible = set(humans)
    else:
        visible = {tm} if tm in humans else set()
    cells = []
    for i, (x, y2) in enumerate(sp.xy2):
        u = g.board.get(i)
        cells.append({
            "i": i, "name": sp.names[i], "x": x, "y2": y2,
            "loc": i in g.loc_set, "control": g.control.get(i),
            "unit": None if u is None else {"owner": u.owner, "team": g.team(u.owner),
                                             "type": u.utype, "coins": u.coins},
        })
    players = []
    for pl in g.players:
        players.append({
            "idx": pl.idx, "team": pl.team, "role": s.roles[pl.idx],
            "units": [{"letter": u, "name": UNITS[u].name, "count": UNITS[u].count,
                       "tactic": UNITS[u].tactic, "ability": UNITS[u].ability} for u in pl.units],
            "hand": sorted(pl.hand) if pl.idx in visible else None,
            "hand_count": len(pl.hand), "bag": len(pl.bag),
            "disc_up": sorted(pl.disc_up), "disc_down": len(pl.disc_down),
            "reserve": pl.reserve, "lost": pl.lost,
            "initiative": g.initiative == pl.idx,
        })
    legal = []
    human_turn = not g.done and tm in humans
    if human_turn:
        pending_coin = g.pending[-1].coin if g.pending and g.pending[-1].kind == "priest" else None
        for k, a in enumerate(g.legal_actions()):
            legal.append({"i": k, "n": action_str(g, a), "pub": action_str(g, a, hidden=False),
                          "desc": describe(g, a), "coin": a.coin, "kind": a.kind,
                          "cells": list(a.cells)})
    else:
        pending_coin = None
    rec = export_record(g, hidden=False)
    moves = "\n".join(l for l in rec.splitlines() if l and not l.startswith("["))
    return {
        "mode": g.mode, "round": g.round, "to_move": tm, "human_turn": human_turn,
        "pending": g.pending_label(), "pending_coin": pending_coin,
        "done": g.done, "result": g.result_label(), "winner": g.winner,
        "markers_left": g.markers_left, "cells": cells, "players": players,
        "legal": legal, "moves": moves, "last": s.last, "last_cells": s.last_cells, "seed": g.seed,
        "cols": len(sp.col_lengths), "max_y2": max(y for _, y in sp.xy2),
    }


def get_session(gid: str) -> Session:
    s = SESSIONS.get(gid)
    if s is None:
        raise HTTPException(404, "Partie introuvable")
    return s


@app.get("/")
def index():
    if (WEB / "index.html").exists():
        return FileResponse(WEB / "index.html")
    return RedirectResponse("/jouer")   # interface du spectateur non compilée


@app.get("/jouer")
def jouer():
    return FileResponse(STATIC / "index.html")


@app.get("/api/bots")
def bots():
    """Bots disponibles (le bot IA n'apparaît que si un modèle entraîné est présent)."""
    from ..bots.neural import modele_par_defaut
    out = [["mcts", "Bot MCTS"], ["heur", "Bot Gumbel heuristique"], ["glouton", "Bot glouton"],
           ["aleatoire", "Bot aléatoire"]]
    import importlib.util
    modele = modele_par_defaut() if importlib.util.find_spec("torch") else None
    if modele:
        out.insert(0, ["ia", "Bot IA (réseau)"])
    return {"bots": out, "modele": modele}


@app.post("/api/games")
def new_game(req: NewGame):
    n = 2 if req.mode == "2J" else 4
    roles = (req.roles + ["mcts"] * n)[:n]
    try:
        g = Game(req.mode, parse_units(req.unites), seed=req.graine)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Mise en place invalide : {e}")
    if n == 4 and any(r.split(":")[0] in ("ia", "heur") for r in roles):
        raise HTTPException(400, "Le bot IA n'est disponible qu'en 2 joueurs pour l'instant")
    gid = uuid.uuid4().hex[:10]
    try:
        SESSIONS[gid] = Session(g, roles)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))
    return {"id": gid, "state": state_json(SESSIONS[gid])}


@app.get("/api/games/{gid}")
def get_state(gid: str, reveal: bool = False):
    return state_json(get_session(gid), reveal)


@app.post("/api/games/{gid}/play")
def play(gid: str, req: Play, reveal: bool = False):
    s = get_session(gid)
    with s.lock:
        g = s.game
        if g.done:
            raise HTTPException(400, "La partie est terminée")
        if g.to_move not in s.humans:
            raise HTTPException(400, "Ce n'est pas au tour d'un humain")
        legal = g.legal_actions()
        try:
            if req.index is not None:
                a = legal[req.index]
            else:
                a = find_action(g, req.notation or "")
        except AmbiguousNotation as e:
            raise HTTPException(400, "Précisez la pièce : " + ", ".join(action_str(g, c) for c in e.candidates))
        except (IllegalAction, IndexError) as e:
            raise HTTPException(400, str(e))
        s.last = [action_str(g, a, hidden=False)]
        s.last_cells = list(a.cells)
        g.apply(a)
        return state_json(s, reveal)


@app.post("/api/games/{gid}/bot")
def bot_step(gid: str, reveal: bool = False):
    """Fait jouer les bots jusqu'au prochain tour humain (ou fin de partie)."""
    s = get_session(gid)
    with s.lock:
        g = s.game
        played = []
        while not g.done and g.to_move in s.bots:
            p = g.to_move
            a = s.bots[p].choose(g)
            played.append({"player": p, "n": action_str(g, a, hidden=False), "cells": list(a.cells)})
            g.apply(a)
            if not s.humans and len(played) >= 1:
                break  # bot contre bot : un coup à la fois pour l'animation
        s.last = [m["n"] for m in played]
        s.last_cells = sorted({c for m in played for c in m["cells"]})
        st = state_json(s, reveal)
        st["bot_moves"] = played
        return st


_ANALYSTE: dict = {}


@app.post("/api/games/{gid}/analyse")
def analyse(gid: str, simulations: int = 200):
    """Conseil de l'IA pour le joueur humain au trait (n'utilise que son information)."""
    from ..ia.analyse import analyser
    s = get_session(gid)
    with s.lock:
        g = s.game
        if g.mode != "2J":
            raise HTTPException(400, "L'analyse IA n'est disponible qu'en 2 joueurs")
        if g.to_move not in s.humans:
            raise HTTPException(400, "L'analyse est réservée au joueur humain au trait")
        try:
            from ..bots.neural import modele_par_defaut
            modele = modele_par_defaut()
            # la date du modèle fait partie de la clé : un nouveau meilleur.pt est pris en compte
            cle = (simulations, modele, modele and os.path.getmtime(modele))
            if cle not in _ANALYSTE:
                _ANALYSTE.clear()
                _ANALYSTE[cle] = make_bot(f"ia:sims={simulations}", seed=0)
            return analyser(g.copy(), _ANALYSTE[cle])
        except (FileNotFoundError, ImportError) as e:
            raise HTTPException(400, f"Bot IA indisponible : {e}")


@app.post("/api/games/{gid}/undo")
def undo(gid: str, reveal: bool = False):
    s = get_session(gid)
    with s.lock:
        g = s.game
        idx = [i for i, (_, p, _) in enumerate(g.log) if p in s.humans]
        if not idx:
            raise HTTPException(400, "Rien à annuler")
        n = idx[-1]
        ng = Game(g.mode, [p.units for p in g.players], seed=g.seed, first=g.first_player,
                  max_rounds=g.max_rounds)
        for _, _, a in g.log[:n]:
            ng.apply(a)
        s.game = ng
        s.last = []
        s.last_cells = []
        return state_json(s, reveal)


@app.get("/api/games/{gid}/record", response_class=PlainTextResponse)
def record(gid: str, complet: bool = False):
    s = get_session(gid)
    roles = {f"Role{i}": r for i, r in enumerate(s.roles)}
    return export_record(s.game, hidden=complet, headers=roles)
