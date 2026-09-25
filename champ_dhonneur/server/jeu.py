"""Page « Jouer » : parties Humain / IA entraînée, analyse de position et éditeur.

Trois mises en place :
  * draft (par défaut) : mise en place avancée du livret, 8 cartes tirées au hasard ou choisies ;
  * libre : les deux armées de 4 unités sont composées avant la partie ;
  * position : une position composée dans l'éditeur.

Une partie est un `Film` (le même format d'images que le spectateur) prolongé coup par coup ;
le navigateur ne reçoit que les images qui lui manquent. Revenir en arrière rejoue la partie
depuis sa position de départ (position initiale ou position de l'éditeur).

Information cachée : face à une IA, la main de l'IA et ses pièces jouées face cachée sont
masquées (sauf « mains visibles » ou partie terminée) ; l'analyse n'utilise alors que
l'information du joueur humain.
"""
from __future__ import annotations

import importlib.util
import os
import re
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from ..engine import FACE_DOWN, Action, Game
from ..ia.conseil import conseil_draft, dossier_valeurs, lire_valeurs
from ..notation import action_str, describe, export_record
from ..position import depuis_position, position
from ..units import BATTLES, DRAFT_POOL, FIRST_GAME, UNITS
from .spectateur import RUNS, Film

router = APIRouter(prefix="/api/jeu")

# armées toutes faites du mode libre (livret p.5 et p.14-15)
MODELES_ARMEES = {"Première partie": FIRST_GAME, "Gaugamèles": BATTLES["gaugameles"],
                  "Bannockburn": BATTLES["bannockburn"], "Crécy": BATTLES["crecy"]}
NIVEAUX = {64: "Rapide", 200: "Normal", 800: "Fort"}


# ------------------------------------------------------------------ modèles
def torch_present() -> bool:
    return importlib.util.find_spec("torch") is not None


def modeles() -> list[dict]:
    """Modèles entraînés disponibles : meilleur modèle de chaque entraînement, puis itérations."""
    out: list[dict] = []
    vus: set[str] = set()

    from ..bots.neural import modele_compatible

    def ajouter(chemin: Path, nom: str) -> None:
        reel = str(chemin.resolve())
        if chemin.exists() and reel not in vus and modele_compatible(chemin):
            vus.add(reel)
            out.append({"chemin": str(chemin), "nom": nom})

    env = os.environ.get("CHAMP_MODELE")
    if env:
        ajouter(Path(env), "Meilleur modèle")
    ajouter(Path("modeles/meilleur.pt"), "Meilleur modèle (modeles/)")
    runs = sorted((d for d in RUNS.iterdir() if (d / "modeles").is_dir()), key=lambda d: -d.stat().st_mtime) \
        if RUNS.is_dir() else []
    for d in runs:
        ajouter(d / "modeles" / "meilleur.pt", f"Meilleur modèle ({d.name})")
    for d in runs:
        for f in sorted((d / "modeles").glob("iter_*.pt"), reverse=True):
            it = int(re.sub(r"\D", "", f.stem) or 0)
            ajouter(f, f"{d.name} · itération {it}")
    return out


def _modele_valide(chemin: str | None) -> str | None:
    if chemin is None:
        return None
    if chemin not in {m["chemin"] for m in modeles()}:
        raise HTTPException(400, "Modèle inconnu")
    return chemin


# ------------------------------------------------------------------ sessions
class Joueur(BaseModel):
    type: str = "humain"            # humain | ia
    niveau: int = 200               # simulations par décision
    modele: str | None = None


class Nouvelle(BaseModel):
    mode: str = "draft"                     # draft | libre (ignoré si `position`)
    cartes: list[str] | None = None         # draft : les 8 cartes (sinon tirées au hasard)
    armees: list[list[str]] | None = None   # libre : 4 unités par armée (sinon au hasard)
    premier: int | None = None              # draft : premier à choisir ; libre : Initiative
    graine: int | None = None
    joueurs: list[Joueur] = [Joueur(), Joueur(type="ia")]
    position: dict | None = None    # position de l'éditeur
    mains_visibles: bool = False


class Jouer(BaseModel):
    index: int


class Revenir(BaseModel):
    pos: int


class Reglages(BaseModel):
    joueurs: list[Joueur] | None = None
    mains_visibles: bool | None = None


class Session:
    def __init__(self, depart: Game, joueurs: list[Joueur], mise: dict, mains_visibles: bool):
        self.depart = depart.copy()
        self.mise = mise                    # mise en place : {"mode", "libelle"}
        self.edite = mise["mode"] == "position"
        self.mains_visibles = mains_visibles
        self.actions: list[Action] = []
        self.version = 0
        self.lock = threading.Lock()
        self.bots: dict[int, object] = {}
        self.regler(joueurs)
        self.film = Film.depuis_partie(self.depart, self._entetes())

    @property
    def game(self) -> Game:
        return self.film.game

    def regler(self, joueurs: list[Joueur]) -> None:
        if len(joueurs) != 2 or any(j.type not in ("humain", "ia") for j in joueurs):
            raise HTTPException(400, "Deux joueurs attendus, humains ou IA")
        ia = [j for j in joueurs if j.type == "ia"]
        if ia and not torch_present():
            raise HTTPException(400, "IA indisponible : PyTorch n'est pas installé (image Docker « ia »)")
        bots = {}
        for i, j in enumerate(joueurs):
            if j.type != "ia":
                continue
            j.modele = _modele_valide(j.modele)
            j.niveau = max(16, min(3200, j.niveau))
            ancien = self.bots.get(i)
            if ancien is not None and getattr(ancien, "_spec", None) == (j.niveau, j.modele):
                bots[i] = ancien
                continue
            from ..bots.neural import NeuralBot
            try:
                b = NeuralBot(modele=j.modele, simulations=j.niveau, seed=i)
            except (FileNotFoundError, ValueError) as e:
                raise HTTPException(400, f"IA indisponible : {e}")
            b._spec = (j.niveau, j.modele)
            bots[i] = b
        self.joueurs = joueurs
        self.bots = bots

    def _entetes(self) -> dict:
        noms = [nom_joueur(j) for j in self.joueurs]
        return {"Type": "partie", "Blanc": noms[0], "Noir": noms[1]}

    def masques(self) -> set[int]:
        """Joueurs dont la main est cachée à l'écran."""
        humains = [i for i, j in enumerate(self.joueurs) if j.type == "humain"]
        if self.mains_visibles or len(humains) != 1:
            return set()
        return {i for i in range(2) if i not in humains}

    def revenir(self, n: int) -> None:
        """Garde les n premières décisions."""
        n = max(0, min(n, len(self.actions)))
        film = Film.depuis_partie(self.depart, self._entetes())
        for a in self.actions[:n]:
            film.ajouter(a)
        self.actions = self.actions[:n]
        self.film = film
        self.version += 1

    def jeu_a(self, pos: int) -> Game:
        """Copie de la partie après `pos` décisions."""
        if pos >= len(self.actions):
            return self.game.copy()
        g = self.depart.copy()
        for a in self.actions[:pos]:
            g.apply(a)
        return g


SESSIONS: dict[str, Session] = {}


def nom_joueur(j: Joueur) -> str:
    if j.type == "humain":
        return "Humain"
    niveau = NIVEAUX.get(j.niveau, f"{j.niveau} simulations")
    m = next((x["nom"] for x in modeles() if x["chemin"] == j.modele), None) if j.modele else None
    return f"IA {niveau.lower()}" + (f" — {m}" if m else "")


def session(gid: str) -> Session:
    s = SESSIONS.get(gid)
    if s is None:
        raise HTTPException(404, "Partie introuvable (le serveur a peut-être redémarré)")
    return s


# ------------------------------------------------------------------ état envoyé au navigateur
_PIECE = re.compile(r" \(pièce [^)]*\)$")


def _masquer(img: dict, masques: set[int]) -> dict:
    if not masques:
        return img
    img = dict(img)
    img["j"] = [dict(p, h=["?"] * len(p["h"])) if i in masques else p for i, p in enumerate(img["j"])]
    a = img.get("a")
    if a and a["j"] in masques and a["k"] in FACE_DOWN:
        img["a"] = dict(a, d=_PIECE.sub("", a["d"]), pc=None)
    return img


def etat(s: Session, depuis: int = 0, version: int = -1) -> dict:
    g = s.game
    imgs = s.film.images
    masques = set() if g.done else s.masques()
    debut = min(depuis, len(imgs)) if version == s.version else 0
    tm = g.to_move
    humain = not g.done and s.joueurs[tm].type == "humain"
    legal = []
    if humain:
        for k, a in enumerate(g.legal_actions()):
            legal.append({"i": k, "n": action_str(g, a, hidden=False), "d": describe(g, a),
                          "coin": a.coin, "kind": a.kind, "cells": list(a.cells)})
    out = {
        "id": s.id, "version": s.version, "debut": debut, "n": len(imgs),
        "images": [_masquer(i, masques) for i in imgs[debut:]],
        "joueurs": [dict(j.model_dump(), nom=nom_joueur(j)) for j in s.joueurs],
        "trait": tm, "humain": humain, "ia": not g.done and not humain,
        "legal": legal, "attente": g.pending_label() if g.pending and not g.done else "",
        "piece_attente": g.pending[-1].coin if humain and g.pending and g.pending[-1].kind == "priest" else None,
        "fini": g.done, "resultat": g.result_label(), "edite": s.edite, "mise": s.mise,
        "mains_visibles": s.mains_visibles, "masques": sorted(masques),
    }
    if g.in_draft:
        d = dossier_valeurs([j.modele for j in s.joueurs] + [os.environ.get("CHAMP_MODELE")], RUNS)
        out["conseil"] = conseil_draft(g, lire_valeurs(d) if d else None)
    if debut == 0:
        out["decor"] = s.film.decor()
    return out


# ------------------------------------------------------------------ routes
@router.get("/regles")
def regles():
    """Ce que l'éditeur et l'écran de mise en place doivent connaître."""
    g = Game("2J", "premiere", seed=0)
    decor = Film.depuis_partie(g).decor()
    return {
        "cases": decor["cases"], "cols": decor["cols"], "max_y2": decor["max_y2"],
        "departs": [[g.spec.names[g.spec.index[n]] for n in l] for l in g.spec.starts],
        "unites": {u: {"nom": d.name, "pieces": d.count, "max": d.max_units, "tactique": d.tactic,
                       "capacite": d.ability} for u, d in sorted(UNITS.items())},
        "premiere": FIRST_GAME, "armees": MODELES_ARMEES, "draft": DRAFT_POOL["2J"], "niveaux": NIVEAUX,
    }


@router.get("/ia")
def ia():
    if not torch_present():
        return {"disponible": False, "raison": "PyTorch n'est pas installé sur ce serveur (image Docker « ia »)",
                "modeles": []}
    m = modeles()
    if not m:
        return {"disponible": False, "raison": "Aucun modèle entraîné trouvé (runs/*/modeles, modeles/)",
                "modeles": []}
    return {"disponible": True, "modeles": m}


def _lettres(l: list[str], n: int, quoi: str) -> list[str]:
    l = [u.upper() for u in l]
    if len(l) != n or len(set(l)) != n or not set(l) <= set(UNITS):
        raise HTTPException(400, f"{quoi} : {n} unités différentes attendues")
    return sorted(l)


def mise_en_place(req: Nouvelle) -> tuple[Game, dict]:
    """Partie de départ et description de sa mise en place."""
    if req.premier not in (None, 0, 1):
        raise HTTPException(400, "Premier joueur : 0 (Or) ou 1 (Argent)")
    if req.position is not None:
        try:
            return depuis_position(req.position, req.graine), {"mode": "position", "libelle": "Position de l'éditeur"}
        except ValueError as e:
            raise HTTPException(400, str(e))
    if req.mode == "draft":
        cartes = None if req.cartes is None else _lettres(req.cartes, DRAFT_POOL["2J"], "Draft")
        g = Game("2J", "draft", seed=req.graine, pool=cartes, draft_first=req.premier)
        return g, {"mode": "draft", "libelle": "Draft · " + ("cartes choisies" if cartes else "cartes tirées au hasard")}
    if req.mode == "libre":
        if req.armees is None:
            armees, nom = "aleatoire", "armées au hasard"
        else:
            if len(req.armees) != 2:
                raise HTTPException(400, "Libre : deux armées attendues")
            armees = [_lettres(a, 4, f"Armée {('Or', 'Argent')[k]}") for k, a in enumerate(req.armees)]
            if set(armees[0]) & set(armees[1]):
                raise HTTPException(400, "Libre : une unité ne peut servir que dans une armée")
            nom = next((n for n, a in MODELES_ARMEES.items() if [sorted(x) for x in a] == armees),
                       "armées choisies")
        g = Game("2J", armees, seed=req.graine, first=req.premier)
        return g, {"mode": "libre", "libelle": f"Libre · {nom}"}
    raise HTTPException(400, "Mode de jeu : « draft » ou « libre »")


@router.post("")
def nouvelle(req: Nouvelle):
    g, mise = mise_en_place(req)
    s = Session(g, req.joueurs, mise, req.mains_visibles)
    s.id = uuid.uuid4().hex[:10]
    SESSIONS[s.id] = s
    while len(SESSIONS) > 200:
        SESSIONS.pop(next(iter(SESSIONS)))
    return etat(s)


@router.get("/{gid}")
def lire(gid: str, depuis: int = 0, version: int = -1):
    return etat(session(gid), depuis, version)


@router.post("/{gid}/jouer")
def jouer(gid: str, req: Jouer, depuis: int = 0, version: int = -1):
    s = session(gid)
    with s.lock:
        g = s.game
        if g.done:
            raise HTTPException(400, "La partie est terminée")
        if s.joueurs[g.to_move].type != "humain":
            raise HTTPException(400, "C'est au tour de l'IA")
        legal = g.legal_actions()
        if not 0 <= req.index < len(legal):
            raise HTTPException(400, "Coup inconnu (la position a changé ?)")
        a = legal[req.index]
        s.film.ajouter(a)
        s.actions.append(a)
        return etat(s, depuis, version)


@router.post("/{gid}/ia")
def coup_ia(gid: str, depuis: int = 0, version: int = -1):
    """L'IA au trait joue une décision (le navigateur rappelle tant que l'IA a le trait)."""
    s = session(gid)
    with s.lock:
        g = s.game
        if not g.done and s.joueurs[g.to_move].type == "ia":
            a = s.bots[g.to_move].choose(g)
            s.film.ajouter(a)
            s.actions.append(a)
        return etat(s, depuis, version)


@router.post("/{gid}/revenir")
def revenir(gid: str, req: Revenir):
    """Reprend la partie après `pos` décisions (les suivantes sont effacées)."""
    s = session(gid)
    with s.lock:
        s.revenir(req.pos)
        return etat(s)


@router.post("/{gid}/annuler")
def annuler(gid: str):
    """Annule le dernier coup humain (et les réponses de l'IA qui ont suivi)."""
    s = session(gid)
    with s.lock:
        g = s.depart.copy()
        dernier = None
        for k, a in enumerate(s.actions):
            if s.joueurs[g.to_move].type == "humain" and g.turn_start:
                dernier = k
            g.apply(a)
        if dernier is None:
            dernier = max(0, len(s.actions) - 1)
        s.revenir(dernier)
        return etat(s)


@router.post("/{gid}/reglages")
def reglages(gid: str, req: Reglages):
    s = session(gid)
    with s.lock:
        if req.joueurs is not None:
            s.regler(req.joueurs)
            s.film.entetes = s._entetes()
        if req.mains_visibles is not None:
            s.mains_visibles = req.mains_visibles
        s.version += 1
        return etat(s)


@router.get("/{gid}/position")
def lire_position(gid: str, pos: int | None = None):
    s = session(gid)
    with s.lock:
        g = s.jeu_a(len(s.actions) if pos is None else pos)
    masques = s.masques()
    if masques and not g.done:
        # information cachée de l'IA : répartition tirée au hasard, compatible avec ce que sait l'humain
        humain = next(i for i in range(2) if i not in masques)
        return dict(position(g.determinize(humain)), hasard=True)
    return position(g)


@router.get("/{gid}/releve", response_class=PlainTextResponse)
def releve(gid: str):
    s = session(gid)
    if s.edite:
        raise HTTPException(400, "Partie issue de l'éditeur : pas de relevé rejouable depuis la mise en place")
    noms = [nom_joueur(j) for j in s.joueurs]
    return export_record(s.game, hidden=True, headers={"Blanc": noms[0], "Noir": noms[1]})


# ------------------------------------------------------------------ analyse
_ANALYSTE: dict = {}
_ANALYSTE_VERROU = threading.Lock()


def _analyste(simulations: int, modele: str | None):
    from ..bots.neural import modele_par_defaut
    from ..ia.analyse import bot_analyse
    chemin = modele or modele_par_defaut()
    cle = (simulations, chemin, chemin and os.path.getmtime(chemin))
    if cle not in _ANALYSTE:
        _ANALYSTE.clear()
        _ANALYSTE[cle] = bot_analyse(simulations, chemin)
    return _ANALYSTE[cle]


@router.post("/{gid}/analyse")
def analyse(gid: str, pos: int | None = None, simulations: int = 400, modele: str | None = None):
    """Évaluation (10 = un bastion d'avance, + pour Or), victoire forcée et meilleurs coups."""
    from ..score import Solveur, appreciation, bastions, texte_mat, texte_score
    s = session(gid)
    with s.lock:
        n = len(s.actions) if pos is None else max(0, min(pos, len(s.actions)))
        g = s.jeu_a(n)
        masques = set() if g.done else s.masques()
    base = {"pos": n, "trait": g.to_move, "bastions": bastions(g)}
    if g.done:
        return dict(base, fini=g.result_label(), texte=g.result_label().replace("1/2-1/2", "½-½"), coups=[],
                    score=0.0 if g.winner is None else (99.9 if g.winner == 0 else -99.9),
                    gain_or=0.5 if g.winner is None else float(g.winner == 0))
    if g.to_move in masques:
        return dict(base, indisponible="L'IA réfléchit : l'analyse reprend à votre tour "
                                       "(elle n'utilise que ce que vous savez).")
    observateur = g.to_move if masques else None
    modele = _modele_valide(modele)
    ia_ok = torch_present() and (modele or modeles())
    if not ia_ok:
        mat = Solveur(observateur).chercher(g)
        b = bastions(g)
        score = 10.0 * (b[0] - b[1])
        out = dict(base, source="materiel", score=score, texte=texte_score(score),
                   appreciation=appreciation(score), coups=[], mat=None,
                   indisponible="Réseau indisponible : score au seul décompte des bastions")
        if mat:
            out.update(texte=texte_mat(mat["equipe"], mat["coups"]),
                       mat={"equipe": mat["equipe"], "coups": mat["coups"],
                            "coup": mat["action"] and action_str(g, mat["action"], hidden=False)})
        return _gain(out)
    from ..ia.analyse import analyser
    simulations = max(32, min(3200, simulations))
    with _ANALYSTE_VERROU:
        try:
            bot = _analyste(simulations, modele)
        except (FileNotFoundError, ImportError) as e:
            raise HTTPException(400, f"IA indisponible : {e}")
        r = analyser(g, bot, top=6, observateur=observateur)
    legal = g.legal_actions()
    for c in r["coups"]:
        action = c.pop("action")
        c["cases"] = list(action.cells)
        c["i"] = legal.index(action)            # index du coup pour le jouer depuis l'analyse
    return _gain(dict(base, source="reseau", **{k: v for k, v in r.items() if k != "bastions"}))


def _gain(out: dict) -> dict:
    """Part de la barre d'évaluation revenant à Or (espérance de gain ramenée dans [0, 1])."""
    from ..score import vers_valeur
    if out.get("mat"):
        out["gain_or"] = 1.0 if out["mat"]["equipe"] == 0 else 0.0
    elif out.get("score") is not None:
        out["gain_or"] = round((1 + vers_valeur(out["score"], out.get("k"))) / 2, 4)
    return out
