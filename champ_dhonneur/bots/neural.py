"""Bot IA : réseau entraîné par auto-jeu + recherche Gumbel IS-MCTS.

Spécification (via make_bot) :
    ia                       modèle par défaut, 200 simulations
    ia:800                   800 simulations
    ia:t=2                   2 secondes par décision (recherche progressive)
    ia:modele=runs/x/modeles/meilleur.pt,sims=400,dispositif=cuda
    ia:moteur=python         recherche du moteur Python (défaut : Rust si le module est compilé)
    heur:200                 même recherche, évaluée par l'heuristique (sans réseau)

Avec le moteur Rust (`champ_rs`), la partie est reconstruite dans le moteur Rust à chaque coup
et cherchée par vagues de `PARALLELE_RUST` simulations : bien plus de simulations par seconde
que la recherche Python (voir docs/IA.md). L'analyse (lignes de jeu) reste en Python.

Le modèle par défaut est cherché dans $CHAMP_MODELE, puis modeles/meilleur.pt,
puis runs/continu/modeles/meilleur.pt et runs/principal/modeles/meilleur.pt.
Le dispositif par défaut est $CHAMP_DISPOSITIF (cpu si absent).
"""
from __future__ import annotations

import os
from pathlib import Path

from ..engine import Action, Game
from .base import Bot

DEFAUTS = ["modeles/meilleur.pt", "runs/continu/modeles/meilleur.pt", "runs/principal/modeles/meilleur.pt"]
PARALLELE_RUST = 16   # simulations par vague (perte virtuelle) de la recherche Rust du bot
_CACHE: dict[tuple, object] = {}
_VERSIONS: dict[tuple, bool] = {}


def modele_compatible(chemin: str | Path) -> bool:
    """Le modèle a-t-il été entraîné avec l'encodage actuel ? (mis en cache par date du fichier)"""
    try:
        cle = (str(chemin), os.path.getmtime(chemin))
    except OSError:
        return False
    if cle not in _VERSIONS:
        try:
            import torch

            from ..ia.encodage import VERSION
            ck = torch.load(chemin, map_location="cpu", weights_only=False, mmap=True)
            _VERSIONS[cle] = ck.get("version_encodage", 1) == VERSION
        except Exception:  # noqa: BLE001 — fichier illisible ou PyTorch absent
            _VERSIONS[cle] = False
    return _VERSIONS[cle]


def modele_par_defaut() -> str | None:
    env = os.environ.get("CHAMP_MODELE")
    if env and Path(env).exists() and modele_compatible(env):
        return env
    for c in DEFAUTS:
        if Path(c).exists() and modele_compatible(c):
            return c
    return None


class NeuralBot(Bot):
    name = "ia"

    def __init__(self, modele: str | None = None, simulations: int = 200,
                 temps: float | None = None, dispositif: str | None = None, heuristique: bool = False,
                 seed: int | None = None, moteur: str = "auto"):
        super().__init__(seed)
        from ..ia import rs
        from ..ia.recherche import RechercheGumbel
        self.simulations, self.temps = simulations, temps
        if moteur == "rust" and not rs.disponible():
            raise RuntimeError("moteur Rust demandé mais indisponible (module champ_rs absent ?)")
        self.rust = moteur != "python" and not heuristique and rs.disponible()
        dispositif = dispositif or os.environ.get("CHAMP_DISPOSITIF", "cpu")
        if heuristique:
            from ..ia.evaluateurs import EvaluateurHeuristique
            self.ev = EvaluateurHeuristique()
            self.name = "heur"
            par = 1
        else:
            chemin = modele or modele_par_defaut()
            if chemin is None:
                raise FileNotFoundError(
                    "Aucun modèle entraîné trouvé : lancez « champ entrainer » ou "
                    "définissez CHAMP_MODELE=chemin/vers/meilleur.pt")
            # la date du fichier fait partie de la clé : un modèle remplacé sur disque (entraînement
            # continu qui publie un nouveau meilleur.pt) est rechargé à la partie suivante
            chemin_abs = str(Path(chemin).resolve())
            cle = (chemin_abs, dispositif, os.path.getmtime(chemin))
            if cle not in _CACHE:
                from ..ia.evaluateurs import EvaluateurReseau
                for ancienne in [k for k in _CACHE if k[:2] == cle[:2]]:
                    del _CACHE[ancienne]
                _CACHE[cle] = EvaluateurReseau.depuis_fichier(chemin, dispositif)
            self.ev = _CACHE[cle]
            par = 8   # vagues de 8 simulations : lots plus efficaces pour le réseau
        self.recherche = RechercheGumbel(params_jeu(simulations, par), seed=seed)
        self.derniere = None
        self.mat = None

    def choose(self, game: Game) -> Action:
        """Meilleur coup : victoire forcée s'il y en a une (solveur exact, avec la seule information
        du joueur), sinon celui de la recherche (budget de simulations, ou durée : `temps`)."""
        from ..ia.recherche import Limite, executer
        legal = game.legal_actions()
        self.derniere = self.mat = None
        if len(legal) == 1:
            return legal[0]
        if not game.in_draft:
            from ..score import Solveur
            mat = Solveur(game.to_move).chercher(game)
            if mat and mat["equipe"] == game.team(game.to_move) and mat["action"] is not None:
                self.mat = mat
                return mat["action"]
        if self.rust:
            from ..ia import rs
            agent = {"parallele": PARALLELE_RUST}
            graine = self.rng.randrange(2**62)
            res = (rs.rechercher_temps(game, self.ev, self.temps, agent, graine) if self.temps
                   else rs.rechercher(game, self.ev, self.simulations, agent, graine))
            self.derniere = res
            return res.action
        if self.temps:
            gen = self.recherche.approfondir(game, Limite(secondes=self.temps))
        else:
            gen = self.recherche.generateur(game, self.simulations)
        res = executer([gen], self.ev)[0]
        self.derniere = res
        return res.action

    def analyse(self, top: int = 5) -> list[tuple[Action, float, float, float]]:
        """(action, probabilité π', Q, visites) des meilleures actions de la dernière recherche."""
        r = self.derniere
        if r is None:
            return []
        order = sorted(range(len(r.legal)), key=lambda i: -r.politique[i])[:top]
        return [(r.legal[i], float(r.politique[i]), float(r.q[i]), float(r.visites[i])) for i in order]


def params_jeu(simulations: int, parallele: int = 8):
    """Recherche du bot de jeu et de l'analyse : sans bruit, 32 candidats à la racine."""
    from ..ia.recherche import ParamsRecherche
    return ParamsRecherche(simulations=simulations, m=32, bruit=False, parallele=parallele)


def depuis_spec(arg: str, seed: int | None, heuristique: bool = False) -> NeuralBot:
    kw: dict = {}
    for part in filter(None, arg.split(",")):
        if "=" not in part:
            kw["simulations"] = int(part)
            continue
        k, v = part.split("=", 1)
        if k in ("sims", "simulations"):
            kw["simulations"] = int(v)
        elif k in ("t", "temps"):
            kw["temps"] = float(v)
        elif k in ("modele", "model"):
            kw["modele"] = v
        elif k in ("dispositif", "device"):
            kw["dispositif"] = v
        elif k == "moteur":
            kw["moteur"] = v
        else:
            raise ValueError(f"Option de bot IA inconnue : {k}")
    return NeuralBot(heuristique=heuristique, seed=seed, **kw)
