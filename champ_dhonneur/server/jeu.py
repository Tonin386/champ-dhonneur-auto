"""Page « Jouer » : parties Humain / IA entraînée, analyse de position et éditeur.

Trois mises en place :
  * draft (par défaut) : mise en place avancée du livret, 8 cartes tirées au hasard ou choisies ;
  * libre : les deux armées de 4 unités sont composées avant la partie ;
  * position : une position composée dans l'éditeur.

Partie hybride (`hybride`, docs/HYBRIDE.md) : la partie se joue sur un vrai plateau contre l'IA.
On saisit les pioches de l'IA (pièces tirées de son sac) et les coups du joueur plateau, sans sa
main : ses coups sont « libres » (toute pièce qu'il pourrait avoir) et sa main reste fictive.
L'historique est alors une suite d'étapes (coup concret, pioches saisies de l'IA).

Historique : chaque partie jouée (au moins une décision) est enregistrée dans $CHAMP_PARTIES
(par défaut ./parties), un fichier JSON par partie réécrit à chaque coup : mise en place,
joueurs, décisions et pioches saisies. Une partie absente de la mémoire (serveur redémarré,
page rechargée) est relue depuis ce fichier et rejouée à l'identique.

Une partie est un `Film` (le même format d'images que le spectateur) prolongé coup par coup ;
le navigateur ne reçoit que les images qui lui manquent. Revenir en arrière rejoue la partie
depuis sa position de départ (position initiale ou position de l'éditeur).

Information cachée : face à une IA, la main de l'IA et ses pièces jouées face cachée sont
masquées (sauf « mains visibles » ou partie terminée) ; l'analyse n'utilise alors que
l'information du joueur humain.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from ..engine import FACE_DOWN, Action, Game, IllegalAction
from ..hybride import (CACHEE, TirageRequis, coups_libres, concretiser, essayer, pieces_possibles,
                       piocher_initial, retirer_main)
from ..ia.conseil import conseil_draft, dossier_valeurs, lire_valeurs
from ..notation import action_str, describe, export_record
from ..position import depuis_position, position
from ..units import BATTLES, DRAFT_POOL, FIRST_GAME, UNITS
from .spectateur import RUNS, Film

router = APIRouter(prefix="/api/jeu")
PARTIES = Path(os.environ.get("CHAMP_PARTIES", "parties"))
FORMAT_PARTIE = 1
_journal = logging.getLogger(__name__)

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
    hybride: bool = False           # partie sur un vrai plateau : une IA et un joueur plateau


class Jouer(BaseModel):
    index: int


class Revenir(BaseModel):
    pos: int
    copie: bool = False             # reprendre dans une nouvelle partie (variante), l'originale intacte


class Piocher(BaseModel):
    piece: str


class Reglages(BaseModel):
    joueurs: list[Joueur] | None = None
    mains_visibles: bool | None = None


class Session:
    def __init__(self, depart: Game, joueurs: list[Joueur], mise: dict, mains_visibles: bool,
                 hybride: bool = False, recette: dict | None = None, tolerant: bool = False):
        self.id = uuid.uuid4().hex[:10]
        self.cree = time.time()
        # comment reconstruire la position de départ : {"setup": …} ou {"position": …, "graine": …}
        self.recette = recette or {"setup": depart.setup}
        self.mise = mise                    # mise en place : {"mode", "libelle"}
        self.edite = mise["mode"] == "position"
        self.mains_visibles = mains_visibles
        self.hybride = hybride
        self.actions: list[Action] = []
        self.tirages: list[list[str]] = []  # hybride : pioches saisies de l'IA à chaque étape
        self.attente: dict | None = None    # hybride : coup en attente des pioches de l'IA
        self.version = 0
        self.lock = threading.Lock()
        self.bots: dict[int, object] = {}
        self.regler(joueurs, tolerant)
        self._depart = depart.copy()
        self.initial: list[str] | None = None   # hybride : première main de l'IA, saisie
        if hybride and not depart.in_draft and not self.edite:
            retirer_main(self._depart, self.ia)
            self.initial = []
        self.film = Film.depuis_partie(self.depart(), self._entetes())

    @property
    def game(self) -> Game:
        return self.film.game

    def depart(self) -> Game:
        """Position de départ (avec la première main saisie de l'IA)."""
        g = self._depart.copy()
        if self.initial:
            piocher_initial(g, self.ia, self.initial)
        return g

    def regler(self, joueurs: list[Joueur], tolerant: bool = False) -> None:
        """Joueurs et bots IA. `tolerant` (partie relue de l'historique) : une IA impossible à
        recréer (modèle disparu, PyTorch absent) laisse la partie consultable, sans bot."""
        if len(joueurs) != 2 or any(j.type not in ("humain", "ia") for j in joueurs):
            raise HTTPException(400, "Deux joueurs attendus, humains ou IA")
        if self.hybride:
            if sorted(j.type for j in joueurs) != ["humain", "ia"]:
                raise HTTPException(400, "Partie hybride : une IA et un joueur plateau")
            if hasattr(self, "ia") and joueurs[self.ia].type != "ia":
                raise HTTPException(400, "Partie hybride : l'IA ne change pas de camp en cours de partie")
            self.ia = next(i for i, j in enumerate(joueurs) if j.type == "ia")
            self.plateau = 1 - self.ia
        ia = [j for j in joueurs if j.type == "ia"]
        if ia and not torch_present() and not tolerant:
            raise HTTPException(400, "IA indisponible : PyTorch n'est pas installé (image Docker « ia »)")
        bots = {}
        for i, j in enumerate(joueurs):
            if j.type != "ia":
                continue
            j.niveau = max(16, min(3200, j.niveau))
            try:
                if not torch_present():
                    raise HTTPException(400, "PyTorch absent")
                try:
                    j.modele = _modele_valide(j.modele)
                except HTTPException:
                    if not tolerant:
                        raise
                    j.modele = None               # modèle disparu : meilleur modèle actuel
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
            except HTTPException:
                if not tolerant:
                    raise
        self.joueurs = joueurs
        self.bots = bots

    def nom(self, j: Joueur) -> str:
        return "Joueur plateau" if self.hybride and j.type == "humain" else nom_joueur(j)

    def _entetes(self) -> dict:
        noms = [self.nom(j) for j in self.joueurs]
        return {"Type": "partie", "Blanc": noms[0], "Noir": noms[1]}

    def masques(self) -> set[int]:
        """Joueurs dont la main est cachée à l'écran (hybride : celle du joueur plateau, fictive)."""
        if self.hybride:
            return {self.plateau}
        humains = [i for i, j in enumerate(self.joueurs) if j.type == "humain"]
        if self.mains_visibles or len(humains) != 1:
            return set()
        return {i for i in range(2) if i not in humains}

    # ---- étapes : coup concret et pioches de l'IA
    def _preparer(self, g: Game, a: Action, tirages: list[str]) -> None:
        if self.hybride:
            if g.to_move == self.plateau:
                concretiser(g, self.plateau, a)
            if tirages:
                g.force_draws(self.ia, tirages)

    def _ajouter(self, film: Film, a: Action, tirages: list[str]) -> None:
        self._preparer(film.game, a, tirages)
        film.ajouter(a)
        if tirages:
            film.images[-1]["a"]["ti"] = list(tirages)

    def _rejouer(self, n: int) -> Film:
        film = Film.depuis_partie(self.depart(), self._entetes())
        for a, t in zip(self.actions[:n], self.tirages[:n]):
            self._ajouter(film, a, t)
        return film

    def _valider(self, a: Action, tirages: list[str]) -> None:
        self._ajouter(self.film, a, tirages)
        self.actions.append(a)
        self.tirages.append(tirages)
        self.attente = None

    def jouer_etape(self, a: Action) -> None:
        """Joue un coup concret ; en hybride, attend d'abord les pioches de l'IA qu'il déclenche."""
        if not self.hybride:
            self._valider(a, [])
            return
        joueur = self.game.to_move
        vu = a._replace(coin=CACHEE) if joueur == self.plateau and a.coin and a.kind in FACE_DOWN else a
        texte = {"n": action_str(self.game, vu, hidden=False), "d": _decrire(self.game, vu), "j": joueur}
        self._poursuivre({"action": a, "tirages": [], "joueur": joueur, "coup": texte})

    def _poursuivre(self, att: dict) -> None:
        """Essaie le coup en attente avec les pioches saisies ; complète les pioches sans choix."""
        a = att["action"]
        while True:
            h = self.game.copy()
            if h.to_move == self.plateau:
                concretiser(h, self.plateau, a)
            try:
                essayer(h, a, self.ia, att["tirages"])
            except TirageRequis as r:
                if len(set(r.sac)) == 1:        # une seule pièce possible : pas de choix
                    att["tirages"] = att["tirages"] + [r.sac[0]]
                    continue
                att["requis"] = r
                self.attente = att
                return
            except IllegalAction as e:
                raise HTTPException(400, str(e))
            self._valider(a, att["tirages"])
            return

    # ---- première main de l'IA (hybride, hors draft et éditeur)
    def _n_initial(self) -> int:
        return min(3, len(self._depart.players[self.ia].bag))

    def _sac_initial(self) -> list[str]:
        sac = list(self._depart.players[self.ia].bag)
        for c in self.initial:
            sac.remove(c)
        return sorted(sac)

    def piocher(self, piece: str) -> None:
        if self.initial is not None and len(self.initial) < self._n_initial():
            if piece not in self._sac_initial():
                raise HTTPException(400, f"{piece} n'est pas dans le sac de l'IA")
            self.initial.append(piece)
            while len(self.initial) < self._n_initial() and len(set(self._sac_initial())) == 1:
                self.initial.append(self._sac_initial()[0])
            self.film = Film.depuis_partie(self.depart(), self._entetes())
            self.version += 1
            return
        if self.attente is None:
            raise HTTPException(400, "Aucune pioche attendue")
        r = self.attente["requis"]
        if piece not in r.sac:
            raise HTTPException(400, f"{piece} n'est pas dans le sac de l'IA")
        self._poursuivre(dict(self.attente, tirages=self.attente["tirages"] + [piece]))

    def recommencer_pioche(self) -> None:
        if self.initial is not None and not self.actions:
            self.initial = []
            self.film = Film.depuis_partie(self.depart(), self._entetes())
            self.version += 1
        elif self.attente is not None:
            self._poursuivre(dict(self.attente, tirages=[]))

    def requis(self) -> dict | None:
        """Pioche de l'IA attendue (hybride) : contexte, pièces du sac, rang."""
        if self.initial is not None and len(self.initial) < self._n_initial():
            return {"contexte": "initial", "sac": dict(Counter(self._sac_initial())), "melange": False,
                    "rang": len(self.initial) + 1, "total": self._n_initial(), "choisies": list(self.initial),
                    "coup": None}
        if self.attente is None:
            return None
        r = self.attente["requis"]
        return {"contexte": "moine" if r.moine else "manche", "sac": dict(Counter(r.sac)),
                "melange": r.melange, "rang": r.rang, "total": r.total,
                "choisies": list(self.attente["tirages"]), "coup": self.attente["coup"]}

    def revenir(self, n: int) -> None:
        """Garde les n premières décisions."""
        n = max(0, min(n, len(self.actions)))
        self.film = self._rejouer(n)
        self.actions = self.actions[:n]
        self.tirages = self.tirages[:n]
        self.attente = None
        self.version += 1

    def jeu_a(self, pos: int) -> Game:
        """Copie de la partie après `pos` décisions."""
        if pos >= len(self.actions):
            return self.game.copy()
        g = self.depart()
        for a, t in zip(self.actions[:pos], self.tirages[:pos]):
            self._preparer(g, a, t)
            g.apply(a)
        return g


    # ---- historique des parties (fichier JSON par partie)
    def donnees(self) -> dict:
        g = self.game
        att = None
        if self.attente is not None:
            att = {"action": _action_json(self.attente["action"]), "tirages": self.attente["tirages"],
                   "joueur": self.attente["joueur"], "coup": self.attente["coup"]}
        return {
            "format": FORMAT_PARTIE, "id": self.id, "cree": self.cree, "maj": time.time(),
            "depart": self.recette, "joueurs": [j.model_dump() for j in self.joueurs],
            "mise": self.mise, "hybride": self.hybride, "mains_visibles": self.mains_visibles,
            "initial": self.initial, "actions": [_action_json(a) for a in self.actions],
            "tirages": self.tirages, "attente": att,
            "resume": {"noms": [self.nom(j) for j in self.joueurs], "types": [j.type for j in self.joueurs],
                       "fini": g.done, "resultat": g.result_label(), "manche": g.round,
                       "decisions": len(self.actions), "trait": g.to_move},
        }

    def enregistrer(self) -> None:
        """Réécrit le fichier de la partie (dès sa première décision). Sans effet si le dossier
        n'est pas accessible en écriture (la partie continue, seulement en mémoire)."""
        if not self.actions and not self.initial:
            return
        try:
            PARTIES.mkdir(parents=True, exist_ok=True)
            tmp = PARTIES / f".{self.id}.json.tmp"
            tmp.write_text(json.dumps(self.donnees(), ensure_ascii=False), encoding="utf-8")
            tmp.replace(PARTIES / f"{self.id}.json")
        except OSError as e:
            _journal.warning("Historique des parties : écriture impossible dans %s (%s)", PARTIES, e)

    @classmethod
    def restaurer(cls, d: dict) -> "Session":
        """Partie relue de l'historique : rejoue ses décisions depuis la même position de départ."""
        r = d["depart"]
        g = depuis_position(r["position"], r["graine"]) if "position" in r else Game(**r["setup"])
        s = cls(g, [Joueur(**j) for j in d["joueurs"]], d["mise"], d.get("mains_visibles", False),
                d.get("hybride", False), recette=r, tolerant=True)
        s.id, s.cree = d["id"], d.get("cree", time.time())
        if d.get("initial") is not None:
            s.initial = list(d["initial"])
        s.actions = [_action(a) for a in d["actions"]]
        s.tirages = [list(t) for t in d.get("tirages") or [[] for _ in s.actions]]
        s.film = s._rejouer(len(s.actions))
        att = d.get("attente")
        if att is not None:
            try:
                s._poursuivre({"action": _action(att["action"]), "tirages": list(att["tirages"]),
                               "joueur": att["joueur"], "coup": att["coup"]})
            except HTTPException:
                s.attente = None
        return s

    def copie(self, n: int) -> "Session":
        """Nouvelle partie reprenant les n premières décisions (variante) ; celle-ci reste intacte."""
        d = self.donnees()
        d.update(id=uuid.uuid4().hex[:10], cree=time.time(), attente=None,
                 actions=d["actions"][:n], tirages=d["tirages"][:n],
                 mise=dict(self.mise, libelle=self.mise["libelle"].removesuffix(" · variante") + " · variante"))
        return Session.restaurer(d)


def _action_json(a: Action) -> list:
    return [a.kind, a.coin, a.unit, list(a.cells), a.extra]


def _action(l: list) -> Action:
    return Action(l[0], l[1], l[2], tuple(l[3]), l[4])


def _fichier(gid: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{10}", gid):
        raise HTTPException(404, "Partie introuvable")
    return PARTIES / f"{gid}.json"


def _decrire(g: Game, a: Action) -> str:
    if a.coin == CACHEE:
        return describe(g, a._replace(coin=None)) + " (pièce cachée)"
    return describe(g, a)


SESSIONS: dict[str, Session] = {}


def nom_joueur(j: Joueur) -> str:
    if j.type == "humain":
        return "Humain"
    niveau = NIVEAUX.get(j.niveau, f"{j.niveau} simulations")
    m = next((x["nom"] for x in modeles() if x["chemin"] == j.modele), None) if j.modele else None
    return f"IA {niveau.lower()}" + (f" — {m}" if m else "")


def session(gid: str) -> Session:
    """Partie en mémoire, sinon relue de l'historique."""
    s = SESSIONS.get(gid)
    if s is None:
        f = _fichier(gid)
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise HTTPException(404, "Partie introuvable")
        try:
            s = Session.restaurer(d)
        except Exception as e:  # noqa: BLE001 — fichier d'une version incompatible ou abîmé
            _journal.exception("Partie %s illisible", gid)
            raise HTTPException(400, f"Partie enregistrée illisible : {e}")
        _memoriser(s)
    return s


def _memoriser(s: Session) -> None:
    SESSIONS[s.id] = s
    while len(SESSIONS) > 200:
        SESSIONS.pop(next(iter(SESSIONS)))


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
    # hybride : la main du joueur plateau est fictive, même en fin de partie
    masques = s.masques() if s.hybride or not g.done else set()
    debut = min(depuis, len(imgs)) if version == s.version else 0
    tm = g.to_move
    requis = s.requis() if s.hybride else None
    libre = not g.done and requis is None
    humain = libre and s.joueurs[tm].type == "humain"
    legal = []
    if humain:
        coups = coups_libres(g, tm) if s.hybride else g.legal_actions()
        for k, a in enumerate(coups):
            legal.append({"i": k, "n": action_str(g, a, hidden=False), "d": _decrire(g, a),
                          "coin": a.coin, "kind": a.kind, "cells": list(a.cells)})
    moine = bool(humain and g.pending and g.pending[-1].kind == "priest")
    attente = g.pending_label() if g.pending and not g.done else ""
    if moine and s.hybride:
        attente = "Moine soldat : pièce piochée inconnue, choisissez celle qu'il a jouée"
    out = {
        "id": s.id, "version": s.version, "debut": debut, "n": len(imgs),
        "images": [_masquer(i, masques) for i in imgs[debut:]],
        "joueurs": [dict(j.model_dump(), nom=s.nom(j)) for j in s.joueurs],
        "trait": tm, "humain": humain, "ia": libre and not humain,
        "legal": legal, "attente": attente,
        "piece_attente": g.pending[-1].coin if moine and not s.hybride else None,
        "fini": g.done, "resultat": g.result_label(), "edite": s.edite, "mise": s.mise,
        "mains_visibles": s.mains_visibles, "masques": sorted(masques),
        "hybride": {"ia": s.ia, "plateau": s.plateau} if s.hybride else None,
        "tirage": requis,
        # hybride : pièces que le joueur plateau pourrait jouer (information publique)
        "possibles": dict(pieces_possibles(g, tm)) if humain and s.hybride and not g.in_draft
        and (not g.pending or moine) else None,
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
    if req.hybride:
        mise = dict(mise, libelle="Plateau réel · " + mise["libelle"])
    recette = {"position": req.position, "graine": g.seed} if req.position is not None else None
    s = Session(g, req.joueurs, mise, req.mains_visibles, req.hybride, recette)
    _memoriser(s)
    return etat(s)


@router.get("/parties")
def parties():
    """Historique : parties enregistrées (au moins une décision), les plus récentes d'abord."""
    out = []
    if PARTIES.is_dir():
        for f in PARTIES.glob("*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({"id": d["id"], "cree": d.get("cree"), "maj": d.get("maj"), "mise": d["mise"]["libelle"],
                        "hybride": d.get("hybride", False), **d.get("resume", {})})
    out.sort(key=lambda p: -(p.get("maj") or 0))
    return {"parties": out, "dossier": str(PARTIES.resolve()),
            "ecriture": os.access(PARTIES if PARTIES.is_dir() else PARTIES.parent, os.W_OK)}


@router.delete("/parties/{gid}")
def supprimer_partie(gid: str):
    f = _fichier(gid)
    SESSIONS.pop(gid, None)
    try:
        f.unlink()
    except FileNotFoundError:
        raise HTTPException(404, "Partie introuvable")
    return {"ok": True}


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
        if s.hybride and s.requis():
            raise HTTPException(400, "Saisissez d'abord la pioche de l'IA")
        if s.joueurs[g.to_move].type != "humain":
            raise HTTPException(400, "C'est au tour de l'IA")
        legal = coups_libres(g, g.to_move) if s.hybride else g.legal_actions()
        if not 0 <= req.index < len(legal):
            raise HTTPException(400, "Coup inconnu (la position a changé ?)")
        a = legal[req.index]
        if s.hybride:
            a = concretiser(g.copy(), g.to_move, a)   # pièce fictive, choisie sans toucher la partie
        s.jouer_etape(a)
        s.enregistrer()
        return etat(s, depuis, version)


@router.post("/{gid}/ia")
def coup_ia(gid: str, depuis: int = 0, version: int = -1):
    """L'IA au trait joue une décision (le navigateur rappelle tant que l'IA a le trait)."""
    s = session(gid)
    with s.lock:
        g = s.game
        if not g.done and s.joueurs[g.to_move].type == "ia" and not (s.hybride and s.requis()):
            if g.to_move not in s.bots:
                raise HTTPException(400, "IA indisponible sur ce serveur : la partie reste consultable")
            s.jouer_etape(s.bots[g.to_move].choose(g))
            s.enregistrer()
        return etat(s, depuis, version)


@router.post("/{gid}/tirage")
def tirage(gid: str, req: Piocher, depuis: int = 0, version: int = -1):
    """Hybride : pièce tirée du sac de l'IA sur la table (une par une, dans l'ordre)."""
    s = session(gid)
    if not s.hybride:
        raise HTTPException(400, "Pioches saisies : partie hybride seulement")
    with s.lock:
        s.piocher(req.piece)
        s.enregistrer()
        return etat(s, depuis, version)


@router.post("/{gid}/tirage/annuler")
def tirage_annuler(gid: str, depuis: int = 0, version: int = -1):
    """Hybride : recommence la pioche en cours (le coup qui l'a déclenchée est conservé)."""
    s = session(gid)
    with s.lock:
        s.recommencer_pioche()
        s.enregistrer()
        return etat(s, depuis, version)


@router.post("/{gid}/revenir")
def revenir(gid: str, req: Revenir):
    """Reprend la partie après `pos` décisions (les suivantes sont effacées)."""
    s = session(gid)
    with s.lock:
        if req.copie:
            v = s.copie(max(0, min(req.pos, len(s.actions))))
            _memoriser(v)
            v.enregistrer()
            return etat(v)
        s.revenir(req.pos)
        s.enregistrer()
        return etat(s)


@router.post("/{gid}/annuler")
def annuler(gid: str):
    """Annule le dernier coup humain (et les réponses de l'IA qui ont suivi)."""
    s = session(gid)
    with s.lock:
        if s.attente is not None:
            att, s.attente = s.attente, None
            if att["joueur"] != s.ia:           # coup du joueur plateau pas encore validé : abandonné
                s.enregistrer()
                return etat(s)
        if s.hybride and not s.actions and s.initial:
            s.recommencer_pioche()               # première main de l'IA à saisir de nouveau
            s.enregistrer()
            return etat(s)
        g = s.depart()
        dernier = None
        for k, (a, t) in enumerate(zip(s.actions, s.tirages)):
            if s.joueurs[g.to_move].type == "humain" and g.turn_start:
                dernier = k
            s._preparer(g, a, t)
            g.apply(a)
        if dernier is None:
            dernier = max(0, len(s.actions) - 1)
        s.revenir(dernier)
        s.enregistrer()
        return etat(s)


@router.post("/{gid}/reglages")
def reglages(gid: str, req: Reglages):
    s = session(gid)
    with s.lock:
        if req.joueurs is not None:
            s.regler(req.joueurs)
            s.film.entetes = s._entetes()
        if req.mains_visibles is not None and not s.hybride:
            s.mains_visibles = req.mains_visibles
        s.version += 1
        s.enregistrer()
        return etat(s)


@router.get("/{gid}/position")
def lire_position(gid: str, pos: int | None = None):
    s = session(gid)
    with s.lock:
        g = s.jeu_a(len(s.actions) if pos is None else pos)
    masques = s.masques()
    if masques and not g.done:
        # information cachée : répartition tirée au hasard, compatible avec ce que sait l'autre joueur
        voit = next(i for i in range(2) if i not in masques)
        qui = "du joueur plateau" if s.hybride else "de l'IA"
        return dict(position(g.determinize(voit)),
                    hasard=f"Pièces cachées {qui} (main, sac, défausse cachée) réparties au hasard")
    return position(g)


@router.get("/{gid}/releve", response_class=PlainTextResponse)
def releve(gid: str):
    s = session(gid)
    if s.edite:
        raise HTTPException(400, "Partie issue de l'éditeur : pas de relevé rejouable depuis la mise en place")
    if s.hybride:
        raise HTTPException(400, "Partie hybride : les pioches saisies ne figurent pas dans un relevé")
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
        base["coups"] = []
        if s.hybride:
            return dict(base, indisponible="Main du joueur plateau inconnue : l'analyse, vue par l'IA, "
                                           "reprend à son tour.")
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
