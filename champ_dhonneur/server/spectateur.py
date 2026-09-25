"""Spectateur de l'entraînement : tableau de bord, parties en direct et rediffusions.

Une partie est rejouée une seule fois côté serveur puis envoyée au navigateur sous forme
d'« images » compactes (une par décision) : lecture, retour arrière et changement de vitesse
se font ensuite dans le navigateur, sans aller-retour réseau.

Sources lues dans runs/<nom>/ :
  journal.jsonl, etat.json, elo.json, config.json  tableau de bord
  parties/*.nch                                    rediffusions (parties terminées)
  direct/t*.json, direct/phase.json                parties en cours des processus d'auto-jeu
                                                   (écrits par ia/direct.py, facultatifs)

Le flux /api/entrainements/<nom>/flux (Server-Sent Events) pousse le tableau de bord quand
il change et, pour chaque table en direct, les nouvelles images au fil des coups.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..engine import Action, Game
from ..notation import action_str, describe, import_record, parse_headers
from ..units import UNITS

RUNS = Path(os.environ.get("CHAMP_RUNS", "runs"))
router = APIRouter()


# ------------------------------------------------------------------ images d'une partie
class Film:
    """Images successives d'une partie ; `ajouter` la prolonge d'une décision (direct).

    Chaque unité reçoit un identifiant stable d'une image à l'autre : le navigateur peut
    ainsi animer ses déplacements.
    """

    def __init__(self, mode: str, unites: list[list[str]], graine: int, initiative: int,
                 max_manches: int, entetes: dict | None = None, setup: dict | None = None):
        game = Game(**setup) if setup else Game(mode, unites, seed=graine, first=initiative,
                                                max_rounds=max_manches)
        self._demarrer(game, entetes)

    def _demarrer(self, game: Game, entetes: dict | None) -> None:
        self.game = game
        self.entetes = entetes or {}
        self._ids: dict[int, int] = {}      # case -> identifiant de l'unité qui l'occupe
        self._cle: dict[int, tuple[int, str]] = {}   # identifiant -> (joueur, type)
        self._suivant = 0
        self._suivre(None)
        self.images = [self._image(None)]

    @classmethod
    def depuis_partie(cls, game: Game, entetes: dict | None = None) -> "Film":
        """Film qui part d'une position quelconque (copiée), par exemple issue de l'éditeur."""
        f = cls.__new__(cls)
        f._demarrer(game.copy(), entetes)
        return f

    @classmethod
    def depuis_releve(cls, texte: str) -> "Film":
        complete = import_record(texte)
        f = cls.__new__(cls)
        f._demarrer(complete.neuve(), parse_headers(texte))
        for _, _, a in complete.log:
            f.ajouter(a)
        return f

    def ajouter(self, a: Action) -> None:
        g = self.game
        info = {"n": action_str(g, a, hidden=False), "d": describe(g, a), "j": g.to_move,
                "k": a.kind, "c": list(a.cells), "u": a.unit, "pc": a.coin}
        g.apply(a)
        self._suivre(a)
        self.images.append(self._image(info))

    def _suivre(self, a: Action | None) -> None:
        """Met à jour les identifiants : une unité qui disparaît d'une case et réapparaît
        ailleurs (même joueur, même type) est la même unité qui s'est déplacée."""
        g = self.game
        avant = self._ids
        apres: dict[int, int] = {}
        nouvelles = []
        for c, u in g.board.items():
            ancien = avant.get(c)
            if ancien is not None and self._cle.get(ancien) == (u.owner, u.utype):
                apres[c] = ancien
            else:
                nouvelles.append(c)
        libres = [i for c, i in avant.items() if i not in apres.values()]
        for c in nouvelles:
            u = g.board[c]
            cand = [i for i in libres if self._cle.get(i) == (u.owner, u.utype)]
            if a is not None and a.cells:
                depart = avant.get(a.cells[0])
                if depart in cand:
                    cand.remove(depart)
                    cand.insert(0, depart)
            if cand:
                apres[c] = cand[0]
                libres.remove(cand[0])
            else:
                apres[c] = self._suivant
                self._cle[self._suivant] = (u.owner, u.utype)
                self._suivant += 1
        self._ids = apres

    def _image(self, info: dict | None) -> dict:
        g = self.game
        img = {
            "r": g.round, "t": g.to_move, "i": g.initiative, "m": list(g.markers_left),
            "u": sorted([self._ids[c], c, u.owner, u.utype, u.coins] for c, u in g.board.items()),
            "c": sorted([c, t] for c, t in g.control.items() if t is not None),
            "j": [{"h": sorted(p.hand), "s": len(p.bag), "d": sorted(p.disc_up),
                   "x": len(p.disc_down), "v": dict(p.reserve),
                   "l": sum(p.lost.values())} for p in g.players],
            "a": info,
        }
        if g.draft is not None:   # mise en place avancée : armées en cours de constitution
            img["arm"] = [list(p.units) for p in g.players]
            if g.in_draft:
                img["tir"] = {"dispo": list(g.draft.available), "etape": g.draft.step}
        if (g.pending or g.in_draft) and not g.done:
            img["p"] = g.pending_label()
        if g.done:
            img["f"] = {"g": g.winner, "r": g.result_label()}
        return img

    def decor(self) -> dict:
        """Ce qui ne change pas pendant la partie : plateau, unités, en-têtes."""
        g = self.game
        sp = g.spec
        lettres = sorted({u for p in g.players for u in p.units} | set(g.draft.pool if g.draft else ()))
        return {
            "mode": g.mode, "cols": len(sp.col_lengths), "max_y2": max(y for _, y in sp.xy2),
            "cases": [[sp.names[i], x, y2, i in g.loc_set] for i, (x, y2) in enumerate(sp.xy2)],
            "equipes": [p.team for p in g.players],
            "unites": [list(p.units) for p in g.players],
            "cartes": {u: {"nom": UNITS[u].name, "pieces": UNITS[u].count,
                           "tactique": UNITS[u].tactic, "capacite": UNITS[u].ability}
                       for u in lettres},
            "graine": g.seed, "entetes": self.entetes,
            **({"draft": {"cartes": list(g.draft.pool), "premier": g.draft.first}} if g.draft else {}),
        }


# ------------------------------------------------------------------ dossiers
def _lire_json(chemin: Path, defaut):
    try:
        return json.loads(chemin.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return defaut


def _mtime(chemin: Path) -> float:
    try:
        return chemin.stat().st_mtime
    except OSError:
        return 0.0


def dossier_run(nom: str) -> Path:
    d = RUNS / nom
    if not re.fullmatch(r"[\w.-]+", nom) or not (d / "config.json").exists():
        raise HTTPException(404, "Entraînement introuvable")
    return d


def _journal(d: Path, nom: str = "journal.jsonl") -> list[dict]:
    lignes = []
    try:
        with open(d / nom, encoding="utf-8") as f:
            for l in f:
                try:
                    lignes.append(json.loads(l))
                except ValueError:
                    pass   # ligne en cours d'écriture
    except OSError:
        pass
    return lignes


def _resume_partie(f: Path) -> dict:
    texte = f.read_text(encoding="utf-8")
    h = parse_headers(texte)
    manches = re.findall(r"(?m)^(\d+)\.", texte)
    return {"fichier": f.stem, "type": h.get("Type", "?"), "iteration": int(h.get("Iteration", 0)),
            "blanc": h.get("Blanc", "?"), "noir": h.get("Noir", "?"), "resultat": h.get("Resultat", "*"),
            "unites": [h.get("Unites0", "").replace(" ", ""), h.get("Unites1", "").replace(" ", "")],
            "manches": int(manches[-1]) if manches else 0, "date": f.stat().st_mtime}


_RESUMES: dict[tuple[str, float], dict] = {}


def _parties(d: Path, n: int) -> list[dict]:
    out = []
    for f in sorted((d / "parties").glob("*.nch"), reverse=True)[:n]:
        try:
            cle = (str(f), f.stat().st_mtime)
            if cle not in _RESUMES:
                _RESUMES[cle] = _resume_partie(f)
            out.append(_RESUMES[cle])
        except OSError:
            pass   # fichier purgé entre-temps
    if len(_RESUMES) > 4000:
        _RESUMES.clear()
    return out


def tableau(d: Path, parties: int = 150) -> dict:
    """Tableau de bord : état, séries par itération (colonnes), courbe d'Elo, parties."""
    cfg = _lire_json(d / "config.json", {})
    etat = _lire_json(d / "etat.json", {})
    elo = _lire_json(d / "elo.json", {}).get("elo", {})
    cols: dict[str, list] = {k: [] for k in (
        "iteration", "parties", "parties_par_heure", "manches", "nulles", "victoires_blanc",
        "exemples", "fenetre", "perte_politique", "perte_valeur", "entropie", "precision",
        "val_precision", "val_premier_coup", "val_kl", "t_autojeu", "t_apprentissage",
        "t_evaluation", "amorce")}
    courbe = []
    for l in _journal(d):
        st, ap, va, t = l.get("autojeu", {}), l.get("apprentissage", {}), l.get("validation") or {}, l.get("temps", {})
        n = max(st.get("parties", 0), 1)
        for k, v in (("iteration", l.get("iteration")), ("parties", st.get("parties", 0)),
                     ("parties_par_heure", st.get("parties_par_heure")),
                     ("manches", round(st.get("manches", 0) / n, 2)),
                     ("nulles", round(st.get("nulles", 0) / n, 4)),
                     ("victoires_blanc", round(st.get("victoires_blanc", 0) / n, 4)),
                     ("exemples", l.get("exemples")), ("fenetre", l.get("fenetre")),
                     ("perte_politique", ap.get("perte_politique")), ("perte_valeur", ap.get("perte_valeur")),
                     ("entropie", ap.get("entropie")), ("precision", ap.get("precision_valeur")),
                     ("val_precision", va.get("precision_valeur")), ("val_premier_coup", va.get("premier_coup")),
                     ("val_kl", va.get("kl")), ("t_autojeu", t.get("autojeu")),
                     ("t_apprentissage", t.get("apprentissage")), ("t_evaluation", t.get("evaluation")),
                     ("amorce", bool(st.get("amorce")))):
            cols[k].append(round(v, 4) if isinstance(v, float) else v)
        if "evaluation" in l:
            e = l["evaluation"]
            # Elo réajusté sur tous les résultats (elo.json) ; à défaut, celui du jour de l'évaluation
            nom_modele = f"iter_{l['iteration']:04d}"
            courbe.append({"iteration": l["iteration"], "elo": round(elo.get(nom_modele, e["elo"]), 1),
                           "elo_initial": e["elo"], "meilleur": e.get("meilleur"),
                           "matchs": {k: v["score"] for k, v in e["matchs"].items()}})
    m = cfg.get("modele", {})
    phase = _lire_json(d / "direct" / "phase.json", None)
    return {
        "nom": d.name, "maintenant": time.time(),
        "iteration": etat.get("iteration", 0), "meilleur": etat.get("meilleur"),
        "elo_meilleur": elo.get(etat.get("meilleur") or ""),
        "ancres": {k: round(v) for k, v in elo.items() if not k.startswith("iter_")},
        "config": {"iterations": cfg.get("iterations", 0), "modele": m,
                   "simulations": cfg.get("autojeu", {}).get("simulations"),
                   "parties_par_iteration": cfg.get("parties_par_iteration"),
                   "travailleurs": cfg.get("travailleurs"), "eval_tous": cfg.get("eval_tous")},
        "phase": phase, "controle": _lire_json(d / "controle.json", {}), "series": cols, "courbe": courbe, "parties": _parties(d, parties),
        "totaux": {"parties": sum(cols["parties"]),
                   "secondes": round(sum((a or 0) + (b or 0) + (c or 0) for a, b, c in
                                         zip(cols["t_autojeu"], cols["t_apprentissage"], cols["t_evaluation"])))},
        "maj": _mtime(d / "journal.jsonl") or None,
    }


def _signature(d: Path) -> tuple:
    return tuple(_mtime(d / f) for f in ("journal.jsonl", "etat.json", "elo.json", "direct/phase.json",
                                         "controle.json", "parties"))


def unites(d: Path) -> dict:
    """Valeur dynamique des unités à chaque itération (unites.jsonl, écrit par ia/valeurs.py) et
    bilan des drafts de l'auto-jeu (journal.jsonl : victoires du premier à choisir)."""
    draft = []
    for l in _journal(d):
        st = l.get("autojeu", {})
        if st.get("parties_draft"):
            draft.append({"iteration": l.get("iteration"), "parties_draft": st["parties_draft"],
                          "victoires_choisit": st.get("victoires_choisit", 0)})
    return {"historique": _journal(d, "unites.jsonl"), "draft": draft}


# ------------------------------------------------------------------ rediffusions
_FILMS: "OrderedDict[tuple[str, float], dict]" = OrderedDict()
_FILMS_VERROU = threading.Lock()


def film_partie(d: Path, fichier: str) -> dict:
    f = d / "parties" / f"{fichier}.nch"
    if not re.fullmatch(r"[\w.-]+", fichier) or not f.exists():
        raise HTTPException(404, "Partie introuvable (peut-être purgée : seules les plus récentes sont gardées)")
    cle = (str(f), _mtime(f))
    with _FILMS_VERROU:
        if cle in _FILMS:
            _FILMS.move_to_end(cle)
            return _FILMS[cle]
    try:
        film = Film.depuis_releve(f.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Relevé illisible : {e}")
    out = {"id": fichier, "decor": film.decor(), "images": film.images, "fini": True}
    with _FILMS_VERROU:
        _FILMS[cle] = out
        while len(_FILMS) > 48:
            _FILMS.popitem(last=False)
    return out


# ------------------------------------------------------------------ direct
class Table:
    """Partie en cours d'un processus d'auto-jeu (direct/tNN.json), suivie incrémentalement."""

    def __init__(self, chemin: Path):
        self.chemin = chemin
        self.nom = chemin.stem
        self.mtime = 0.0
        self.id: str | None = None
        self.film: Film | None = None
        self.meta: dict = {}
        self.verrou = threading.Lock()

    def actualiser(self, phase: dict | None) -> None:
        m = _mtime(self.chemin)
        if m == self.mtime:
            return
        data = _lire_json(self.chemin, None)
        if not data:
            return
        with self.verrou:
            self.mtime = m
            actions = [Action(k, c, u, tuple(cells), x) for k, c, u, cells, x in data["actions"]]
            if data["id"] != self.id or self.film is None or len(actions) < len(self.film.images) - 1:
                ph = phase or {}
                entetes = {"Type": "direct", "Iteration": str(data.get("iteration", ph.get("iteration", ""))),
                           "Blanc": data.get("joueur", ph.get("joueur", "?")),
                           "Noir": data.get("joueur", ph.get("joueur", "?"))}
                self.film = Film(data.get("mode", "2J"), data["unites"], data["graine"],
                                 data["initiative"], data.get("max_manches", 150), entetes,
                                 setup=data.get("setup"))
                self.id = data["id"]
            for a in actions[len(self.film.images) - 1:]:
                self.film.ajouter(a)
            self.meta = {"parties": data.get("parties", 0), "total": data.get("total", 0),
                         "maj": data.get("maj", m), "fini": bool(data.get("fini"))}

    def depuis(self, n: int) -> tuple[list[dict], int]:
        with self.verrou:
            imgs = self.film.images if self.film else []
            return imgs[n:], len(imgs)


_TABLES: dict[str, Table] = {}
_TABLES_VERROU = threading.Lock()


def tables(d: Path) -> list[Table]:
    rep = d / "direct"
    phase = _lire_json(rep / "phase.json", None)
    out = []
    for f in sorted(rep.glob("t*.json")) if rep.is_dir() else []:
        with _TABLES_VERROU:
            t = _TABLES.setdefault(str(f), Table(f))
        try:
            t.actualiser(phase)
        except Exception:  # noqa: BLE001 — fichier d'un autre format : table ignorée
            continue
        if t.film is not None:
            out.append(t)
    return out


def _evenement(nom: str, data) -> str:
    return f"event: {nom}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


# ------------------------------------------------------------------ routes
@router.get("/api/entrainements")
def entrainements():
    out = []
    if RUNS.is_dir():
        for d in RUNS.iterdir():
            if not (d / "config.json").exists():
                continue
            etat = _lire_json(d / "etat.json", {})
            out.append({"nom": d.name, "iteration": etat.get("iteration", 0),
                        "meilleur": etat.get("meilleur"),
                        "maj": max(_mtime(d / "journal.jsonl"), _mtime(d / "config.json"),
                                   _mtime(d / "direct" / "phase.json"))})
    out.sort(key=lambda r: -r["maj"])
    return {"entrainements": out}


class Controle(BaseModel):
    pause: bool


@router.post("/api/entrainements/{nom}/pause")
def pause(nom: str, req: Controle):
    """Pause de l'entraînement après l'itération en cours (controle.json, lu par ia/entrainement.py),
    ou reprise."""
    d = dossier_run(nom)
    try:
        tmp = d / "controle.json.tmp"
        tmp.write_text(json.dumps({"pause": req.pause, "demande": time.time()}), encoding="utf-8")
        os.replace(tmp, d / "controle.json")
    except OSError as e:
        raise HTTPException(503, f"Impossible d'écrire dans {d} ({e.strerror}) : le dossier runs doit être "
                                 "monté en écriture")
    return {"controle": _lire_json(d / "controle.json", {})}


@router.get("/api/entrainements/{nom}")
def entrainement(nom: str, parties: int = 150):
    return tableau(dossier_run(nom), parties)


@router.get("/api/entrainements/{nom}/unites")
def valeurs_unites(nom: str):
    return unites(dossier_run(nom))


@router.get("/api/entrainements/{nom}/parties/{fichier}")
def partie(nom: str, fichier: str):
    """Partie enregistrée, rejouée : décor + une image par décision."""
    return film_partie(dossier_run(nom), fichier)


@router.get("/api/entrainements/{nom}/flux")
async def flux(nom: str, request: Request):
    """Server-Sent Events : `tableau` quand il change, `table` à chaque coup joué en direct."""
    d = dossier_run(nom)

    async def generer():
        signature = None
        vus: dict[str, tuple[str | None, int, bool]] = {}   # table -> (id, images envoyées, fini)
        tour = 0
        dernier_envoi = time.time()
        yield "retry: 2000\n\n"
        while not await request.is_disconnected():
            if tour % 4 == 0:
                sig = await asyncio.to_thread(_signature, d)
                if sig != signature:
                    signature = sig
                    yield _evenement("tableau", await asyncio.to_thread(tableau, d))
                    dernier_envoi = time.time()
            presentes = await asyncio.to_thread(tables, d)
            for nom_t in set(vus) - {t.nom for t in presentes}:
                del vus[nom_t]
                yield _evenement("table", {"table": nom_t, "retire": True})
            for t in presentes:
                ident, n, fini = vus.get(t.nom, (None, 0, False))
                if t.id != ident:
                    n = 0
                imgs, total = t.depuis(n)
                if imgs or t.id != ident or t.meta.get("fini") != fini:
                    msg = {"table": t.nom, "id": t.id, "debut": n, "images": imgs, **t.meta}
                    if n == 0:
                        msg["decor"] = t.film.decor()
                    vus[t.nom] = (t.id, total, t.meta.get("fini", False))
                    yield _evenement("table", msg)
                    dernier_envoi = time.time()
            if time.time() - dernier_envoi > 15:
                yield ": attente\n\n"   # garde la connexion ouverte (proxys)
                dernier_envoi = time.time()
            tour += 1
            await asyncio.sleep(0.25)

    return StreamingResponse(generer(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
