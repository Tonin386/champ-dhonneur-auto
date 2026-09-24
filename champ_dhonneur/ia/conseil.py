"""Conseil de draft : valeur de chaque carte encore disponible pendant la mise en place avancée.

La valeur vient de la dernière mesure de l'entraînement (`runs/<nom>/unites.json`, écrite par
`ia/valeurs.py`), sur l'échelle du scoreur (10 points = un bastion d'avance). Le modèle d'armée
de cette mesure donne, pour le premier à choisir (A) face au second (B, qui commence la partie) :

    V = Σ_{u∈A} a_u − Σ_{w∈B} a_w + Σ_{paires⊂A} s − Σ_{paires⊂B} s + Σ_{u∈A, w∈B} c(u, w) − commencer

La valeur d'une carte est celle du draft si on la prend maintenant, les choix suivants des deux
joueurs étant optimaux pour ce même modèle (minimax exact sur les cartes restantes). Python pur :
utilisable par le serveur web sans PyTorch ni NumPy.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ..units import DRAFT_ORDER

_CACHE: dict[str, tuple[float, dict | None]] = {}


def lire_valeurs(dossier: Path) -> dict | None:
    """Dernière mesure `unites.json` d'un entraînement (mise en cache selon la date du fichier)."""
    f = Path(dossier) / "unites.json"
    try:
        m = f.stat().st_mtime
    except OSError:
        return None
    cle = str(f)
    if cle in _CACHE and _CACHE[cle][0] == m:
        return _CACHE[cle][1]
    try:
        v = json.loads(f.read_text(encoding="utf-8"))
        v = v if isinstance(v, dict) and isinstance(v.get("unites"), dict) else None
    except (OSError, ValueError):
        v = None
    if v is not None:
        v["source"] = Path(dossier).name
    _CACHE[cle] = (m, v)
    return v


def dossier_valeurs(modeles: list[str | None], runs: Path) -> Path | None:
    """Entraînement dont viennent les valeurs : celui du premier modèle qui en a (modèle
    `runs/<nom>/modeles/*.pt`), sinon l'entraînement le plus récent qui en a mesuré."""
    for m in modeles:
        if m:
            d = Path(m).resolve().parent.parent
            if (d / "unites.json").exists():
                return d
    try:
        candidats = [d for d in Path(runs).iterdir() if (d / "unites.json").exists()]
    except OSError:
        return None
    return max(candidats, key=lambda d: (d / "unites.json").stat().st_mtime, default=None)


def _paire(table: dict, u: str, w: str) -> float:
    return float(table.get(u + w if u < w else w + u, 0.0))


def contre(v: dict, u: str, w: str) -> float:
    """c(u, w) en points : > 0 quand u l'emporte sur w (table antisymétrique)."""
    c = v.get("contres", {})
    return float(c.get(u + w, 0.0)) if u < w else -float(c.get(w + u, 0.0))


def valeur_armees(v: dict, a: list[str], b: list[str]) -> float:
    """Avantage de l'armée `a` sur l'armée `b` en points (hors avantage de commencer)."""
    un = v["unites"]
    s = v.get("synergies", {})
    x = sum(float(un[u]["points"]) for u in a) - sum(float(un[w]["points"]) for w in b)
    x += sum(_paire(s, a[i], a[j]) for i in range(len(a)) for j in range(i + 1, len(a)))
    x -= sum(_paire(s, b[i], b[j]) for i in range(len(b)) for j in range(i + 1, len(b)))
    x += sum(contre(v, u, w) for u in a for w in b)
    return x


def conseil_draft(g, v: dict | None) -> dict | None:
    """Valeur de chaque carte disponible pour le joueur qui choisit, ou None hors draft.

    `valeur` : points du draft pour ce joueur s'il prend la carte (minimax, avantage de commencer
    compris) ; `propre`, `synergie`, `contre` : décomposition de l'effet immédiat de la carte
    (valeur propre, synergies avec ses cartes déjà prises, contres face aux cartes adverses)."""
    if v is None or not g.in_draft:
        return None
    d = g.draft
    un = v["unites"]
    if not all(u in un for u in d.pool):
        return None
    ordre = DRAFT_ORDER[g.mode]
    a0 = d.first
    armees = {0: [u for p, u in d.picks if p == a0], 1: [u for p, u in d.picks if p != a0]}
    commencer = float(v.get("commencer", 0.0))
    memo: dict[tuple[frozenset, frozenset], float] = {}

    def minimax(a: frozenset, b: frozenset, etape: int) -> float:
        """Valeur du draft pour A, choix restants optimaux."""
        if etape == len(ordre):
            return valeur_armees(v, sorted(a), sorted(b)) - commencer
        cle = (a, b)
        if cle not in memo:
            libres = [u for u in d.pool if u not in a and u not in b]
            if ordre[etape] == 0:
                memo[cle] = max(minimax(a | {u}, b, etape + 1) for u in libres)
            else:
                memo[cle] = min(minimax(a, b | {u}, etape + 1) for u in libres)
        return memo[cle]

    cote = ordre[d.step]                        # 0 : A choisit, 1 : B choisit
    signe = 1.0 if cote == 0 else -1.0
    miens, siens = armees[cote], armees[1 - cote]
    a, b = frozenset(armees[0]), frozenset(armees[1])
    cartes = {}
    for u in d.available:
        apres = minimax(a | {u}, b, d.step + 1) if cote == 0 else minimax(a, b | {u}, d.step + 1)
        cartes[u] = {
            "valeur": round(signe * apres, 2),
            "propre": round(float(un[u]["points"]), 2),
            "ic95": round(float(un[u].get("ic95", 0.0)), 2),
            "synergie": round(sum(_paire(v.get("synergies", {}), u, x) for x in miens), 2),
            "contre": round(sum(contre(v, u, w) for w in siens), 2),
        }
    if not cartes:
        return None
    meilleure = max(cartes, key=lambda u: cartes[u]["valeur"])
    k = v.get("K")
    return {"cartes": cartes, "meilleure": meilleure, "iteration": v.get("iteration"),
            "source": v.get("source"), "K": k if isinstance(k, (int, float)) and math.isfinite(k) else None,
            "joueur": g.to_move, "premier": cote == 0}
