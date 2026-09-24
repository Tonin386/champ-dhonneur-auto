"""Diffusion en direct de l'entraînement pour le spectateur web (server/spectateur.py).

  * Chaque processus d'auto-jeu suit une de ses parties en cours et l'écrit dans
    runs/<nom>/direct/tNN.json : graine, unités et liste exacte des décisions, de quoi la
    rejouer à l'identique. Écriture atomique, au plus `periode` fois par seconde : quelques
    kilo-octets sérialisés, sans relecture de la partie (coût négligeable devant la recherche).
    Quand la partie suivie se termine, sa position finale est écrite puis le processus suit
    une autre partie.
  * L'entraîneur écrit la phase en cours (auto-jeu, apprentissage, évaluation) dans
    direct/phase.json.

Rien n'est lu par l'entraînement : supprimer le dossier direct/ est sans conséquence.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def _ecrire(chemin: Path, data: dict) -> None:
    tmp = chemin.with_name(chemin.name + ".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, chemin)


def ecrire_phase(dossier: Path, phase: str, iteration: int, **extra) -> None:
    rep = Path(dossier) / "direct"
    rep.mkdir(parents=True, exist_ok=True)
    _ecrire(rep / "phase.json", {"phase": phase, "iteration": iteration, "debut": time.time(), **extra})


def preparer_tables(dossier: Path, n: int) -> list[str]:
    """Chemins des n tables de l'itération ; les tables d'une itération précédente sont effacées."""
    rep = Path(dossier) / "direct"
    rep.mkdir(parents=True, exist_ok=True)
    for f in rep.glob("t*.json"):
        f.unlink(missing_ok=True)
    return [str(rep / f"t{i:02d}.json") for i in range(n)]


class Diffuseur:
    """Suit une partie parmi celles d'un processus d'auto-jeu ; `observer` après chaque vague."""

    def __init__(self, chemin: str, total: int, periode: float = 0.25):
        self.chemin = Path(chemin)
        self.total = total
        self.periode = periode
        self.g = None
        self.n = -1
        self.t = 0.0

    def observer(self, jeux: list, parties_finies: int) -> None:
        if self.g is not None and all(j is not self.g for j in jeux):
            self._ecrire(parties_finies, fini=True)   # partie suivie terminée : position finale
            self.g = None
        if self.g is None and jeux:
            self.g, self.n = jeux[0], -1
        if self.g is None:
            return
        if len(self.g.log) != self.n and time.time() - self.t >= self.periode:
            self._ecrire(parties_finies)

    def _ecrire(self, parties_finies: int, fini: bool = False) -> None:
        g = self.g
        self.n, self.t = len(g.log), time.time()
        try:
            _ecrire(self.chemin, {
                "id": f"{g.seed}", "mode": g.mode, "unites": [p.units for p in g.players],
                "setup": g.setup,
                "graine": g.seed, "initiative": g.first_player, "max_manches": g.max_rounds,
                "actions": [[a.kind, a.coin, a.unit, list(a.cells), a.extra] for _, _, a in g.log],
                "fini": fini or g.done, "parties": parties_finies, "total": self.total,
                "maj": self.t})
        except OSError:
            pass   # la diffusion ne doit jamais interrompre l'auto-jeu
