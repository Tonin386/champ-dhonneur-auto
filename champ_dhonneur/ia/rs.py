"""Pont vers le moteur Rust (module `champ_rs`, compilé depuis rust/).

Le moteur Rust joue les parties d'auto-jeu entières (règles, déterminisations, recherche
Gumbel IS-MCTS, encodage) ; Python n'évalue que les lots de positions avec le réseau. Les
règles sont identiques à celles du moteur Python, vérifiées décision par décision par
tests/test_rust.py, et les graines donnent les mêmes pioches : chaque relevé produit par Rust
est rejoué et exporté par le moteur Python, ce qui contrôle aussi la cohérence en production.
"""
from __future__ import annotations

import time

import numpy as np

from ..engine import Action, Game
from ..notation import export_record
from .encodage import ALL_LETTERS, CELL_F, GLOB_F, UNIT_F

try:
    import champ_rs
except ImportError:   # module non compilé : l'auto-jeu reste en Python
    champ_rs = None

KINDS = list(champ_rs.NOMS_TYPES) if champ_rs else []
LETTRE = {i + 1: u for i, u in enumerate(ALL_LETTERS)} | {17: "*"}
CODE = {u: i for i, u in LETTRE.items()}


def disponible() -> bool:
    return champ_rs is not None


def action_python(t) -> Action:
    genre, piece, unite, extra, cases = t
    return Action(KINDS[genre], LETTRE.get(piece), LETTRE.get(unite), tuple(cases), LETTRE.get(extra))


def action_rust(a: Action) -> tuple:
    return (KINDS.index(a.kind), CODE.get(a.coin, 0), CODE.get(a.unit, 0), CODE.get(a.extra, 0),
            list(a.cells))


class EvaluateurLotUniforme:
    """Politique uniforme, valeur nulle (tests)."""

    def evaluer_lot(self, x: dict) -> tuple[np.ndarray, np.ndarray]:
        b, a = x["acts"].shape[:2]
        return np.zeros((b, a), np.float32), np.zeros(b, np.float32)


def lot_numpy(lot: dict) -> dict[str, np.ndarray]:
    """Lot renvoyé par champ_rs (bytes) -> tableaux numpy (entrées du réseau)."""
    b, a_len = lot["b"], lot["a_len"]
    f = lambda k, dt, *forme: np.frombuffer(lot[k], dt).reshape(b, *forme)  # noqa: E731
    return {"cell_i": f("cell_i", np.int64, 37, 4), "cell_f": f("cell_f", np.float32, 37, CELL_F),
            "unit_i": f("unit_i", np.int64, 8, 2), "unit_f": f("unit_f", np.float32, 8, UNIT_F),
            "glob_i": f("glob_i", np.int64, 2), "glob_f": f("glob_f", np.float32, GLOB_F),
            "acts": f("acts", np.int64, a_len, 7)}


def _donnees(r: dict) -> dict[str, np.ndarray]:
    n = r["n"]
    if n == 0:
        return {"z": np.zeros(0, np.float32)}
    g = lambda k, dt, *forme: np.frombuffer(r[k], dt).reshape(n, *forme)  # noqa: E731
    return {"cell_i": g("cell_i", np.int8, 37, 4), "cell_f": g("cell_f", np.float32, 37, CELL_F).astype(np.float16),
            "unit_i": g("unit_i", np.int8, 8, 2), "unit_f": g("unit_f", np.float32, 8, UNIT_F).astype(np.float16),
            "glob_i": g("glob_i", np.int8, 2), "glob_f": g("glob_f", np.float32, GLOB_F).astype(np.float16),
            "acts": g("acts", np.int8, 64, 7).copy(), "pi": g("pi", np.float32, 64).astype(np.float16),
            "n_act": g("n_act", np.int16).copy(), "z": g("z", np.float32).copy(),
            "q": g("q", np.float32).copy()}


def releve(graine: int, actions: list, resultat: str, max_manches: int, entetes: dict) -> str:
    """Rejoue une partie du moteur Rust dans le moteur Python et exporte son relevé."""
    g = Game("2J", seed=graine)
    g.max_rounds = max_manches
    for t in actions:
        g.apply(action_python(t))
    if g.result_label() != resultat:
        raise RuntimeError(f"moteurs divergents (graine {graine}) : {g.result_label()} contre {resultat}")
    return export_record(g, headers=entetes)


class DiffuseurRs:
    """Diffusion en direct (même fichier que direct.Diffuseur) d'une partie du moteur Rust."""

    def __init__(self, chemin: str, total: int, periode: float = 0.25):
        from pathlib import Path
        self.chemin, self.total, self.periode = Path(chemin), total, periode
        self.n, self.t, self.graine = -1, 0.0, None

    def observer(self, aj, max_manches: int, forcer: bool = False) -> None:
        if not forcer and time.time() - self.t < self.periode:
            return
        s = aj.suivie()
        if s is None:
            return
        graine, premier, armees, actions, fini, finies = s
        if graine == self.graine and len(actions) == self.n and not fini:
            return
        self.graine, self.n, self.t = graine, len(actions), time.time()
        from .direct import _ecrire
        try:
            _ecrire(self.chemin, {
                "id": f"{graine}", "mode": "2J", "unites": [list(a) for a in armees],
                "graine": graine, "initiative": premier, "max_manches": max_manches,
                "actions": [[a.kind, a.coin, a.unit, list(a.cells), a.extra]
                            for a in map(action_python, actions)],
                "fini": fini, "parties": finies, "total": self.total, "maj": self.t})
        except OSError:
            pass   # la diffusion ne doit jamais interrompre l'auto-jeu


def jouer_parties_rs(evaluateur, P, seed: int | None = None) -> tuple[dict, dict]:
    """Même contrat que autojeu.jouer_parties, avec le moteur Rust.

    `evaluateur.evaluer_lot(x)` reçoit les entrées numpy du réseau et renvoie
    (logits (B, A) float32, valeurs (B,) float32 dans [-1, 1]).
    """
    t0 = time.time()
    params = {k: getattr(P, k) for k in ("parties", "simultanees", "simulations", "simulations_rapides",
                                         "p_complete", "max_manches", "m", "parallele", "c_visit",
                                         "c_scale", "releves")}
    aj = champ_rs.AutoJeu(params, int(seed if seed is not None else time.time_ns()) % 2**63)
    diffuseur = DiffuseurRs(P.direct, P.parties) if getattr(P, "direct", "") else None
    lot = aj.etape()
    while lot is not None:
        if diffuseur is not None:
            diffuseur.observer(aj, P.max_manches)
        if P.duree_max and time.time() - t0 > P.duree_max:
            break   # mesure de débit : on abandonne les parties en cours
        logits, valeurs = evaluateur.evaluer_lot(lot_numpy(lot))
        lot = aj.etape(np.ascontiguousarray(logits, np.float32).tobytes(), lot["a_len"],
                       np.ascontiguousarray(valeurs, np.float32).tobytes())
    if diffuseur is not None:
        diffuseur.observer(aj, P.max_manches, forcer=True)   # position finale de la partie suivie
    r = aj.resultats()
    stats = dict(r["stats"])
    stats["secondes"] = time.time() - t0
    stats["releves"] = [releve(gr, acts, res, P.max_manches, {"Type": "autojeu"})
                        for gr, acts, res in r["releves"]]
    return _donnees(r), stats
