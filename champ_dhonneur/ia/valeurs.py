"""Valeur dynamique des unités, sur l'échelle du scoreur (10 points = un bastion d'avance).

Calculée à chaque itération avec le réseau courant, elle évolue avec l'apprentissage.

* **Échelle.** Le réseau estime l'espérance de gain v ; le scoreur la convertit en points par
  10·atanh(v)/K. K est recalibré à chaque itération sur la fenêtre d'exemples (espérance de gain
  observée à d bastions d'avance ≈ tanh(K·d)) et écrit dans `echelle.json`.
* **Sondes fixes.** `pools` tirages de 8 cartes (toujours les mêmes, graine fixe) ; pour chacun,
  les 70 répartitions en deux armées de 4 : A (le premier à choisir) reçoit S, B le reste et
  commence la partie, comme après un draft. Chaque répartition est évaluée par le réseau à la
  première décision de la manche 1, moyennée sur `pioches` tirages de sacs communs à toutes les
  répartitions d'un même tirage : L(S) = atanh(v_A).
* **Valeur d'armée (a_u).** Régression ridge sur les sondes :
      L = c0 + Σ_{u∈A} a_u − Σ_{w∈B} a_w + Σ_{paires⊂A} s − Σ_{paires⊂B} s + Σ_{u∈A, w∈B} c_uw
  a_u : gain, en points, à remplacer une unité moyenne par u ; s : synergies entre alliés ;
  c : contres (c_uw > 0 : u l'emporte sur w). Intervalle de confiance par bootstrap sur les tirages.
* **Draft optimal.** Minimax exact des sondes dans l'ordre A1 B2 A2 B2 A1 : fréquence à laquelle
  chaque unité est le premier choix optimal, regret (en points) si A ne la prend pas en premier
  alors qu'elle est disponible, avantage du premier à choisir.
* **Préférences de l'auto-jeu.** Probabilité de choix (cible π' de la recherche) de chaque carte
  quand elle est disponible, rapportée au choix uniforme (1 = neutre).

Ces valeurs sont une mesure : elles ne servent ni de cible ni d'a priori à l'apprentissage.
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np

from ..units import ALL_LETTERS, DRAFT_ORDER
from .encodage import COIN_ID, PENDING_ID

LETTRES = ALL_LETTERS
IDX = {u: i for i, u in enumerate(LETTRES)}
PAIRES = list(combinations(range(len(LETTRES)), 2))
PIDX = {p: k for k, p in enumerate(PAIRES)}
REPARTITIONS = list(combinations(range(8), 4))      # 70 sous-ensembles S (indices dans le tirage)


# ------------------------------------------------------------------ échelle
def calibrer_k(fenetre: list[dict], maximum: int = 400_000) -> float | None:
    """K tel que l'espérance de gain à d bastions d'avance ≈ tanh(K·d), sur les exemples de jeu."""
    d_all, z_all = [], []
    for data in fenetre:
        if "cell_i" not in data or not len(data["z"]):
            continue
        jeu = data["glob_i"][:, 0] != PENDING_ID["draft"]
        ctrl = data["cell_i"][jeu, :, 2]
        d_all.append((ctrl == 2).sum(1) - (ctrl == 3).sum(1))
        z_all.append(data["z"][jeu])
    if not d_all:
        return None
    d = np.concatenate(d_all).astype(np.float64)
    z = np.concatenate(z_all).astype(np.float64)
    if len(d) > maximum:
        idx = np.random.default_rng(0).choice(len(d), maximum, replace=False)
        d, z = d[idx], z[idx]
    y = (1 + z) / 2
    ks = np.arange(0.05, 3.0, 0.01)
    p = np.clip((1 + np.tanh(ks[:, None] * d[None])) / 2, 1e-6, 1 - 1e-6)
    ll = (y * np.log(p) + (1 - y) * np.log(1 - p)).sum(1)
    return float(ks[int(ll.argmax())])


def points(theta: np.ndarray | float, k: float) -> np.ndarray | float:
    """Logit de gain (atanh v) -> points du scoreur."""
    return 10.0 * np.asarray(theta) / k


# ------------------------------------------------------------------ sondes
def tirages(n: int, graine: int = 12345) -> list[tuple[str, ...]]:
    rng = np.random.default_rng(graine)
    return [tuple(sorted(rng.choice(LETTRES, 8, replace=False).tolist())) for _ in range(n)]


def evaluer_sondes(evaluateur, pools: list[tuple[str, ...]], pioches: int = 2, graine: int = 777,
                   lot: int = 2048) -> np.ndarray:
    """L[t, r] = atanh(v_A) de la répartition r du tirage t (moyenne sur les pioches)."""
    from ..engine import Game
    rng = np.random.default_rng(graine)
    graines = rng.integers(0, 2**31, size=(len(pools), pioches))
    requetes, cles = [], []
    L = np.zeros((len(pools), len(REPARTITIONS)))
    for t, pool in enumerate(pools):
        for r, s in enumerate(REPARTITIONS):
            a = [pool[i] for i in s]
            b = [u for i, u in enumerate(pool) if i not in s]
            for k in range(pioches):
                g = Game("2J", [a, b], seed=int(graines[t, k]), first=1)   # B commence
                requetes.append((g, g.legal_actions()))
                cles.append((t, r))
    somme = np.zeros_like(L)
    for i in range(0, len(requetes), lot):
        reps = evaluateur.evaluer(requetes[i:i + lot])
        for (t, r), (_, v) in zip(cles[i:i + lot], reps):
            somme[t, r] += -v                       # v : point de vue de B (au trait)
    v_a = np.clip(somme / pioches, -0.999, 0.999)
    return np.arctanh(v_a)


# ------------------------------------------------------------------ modèle d'armée
def _caracteristiques(pool: tuple[str, ...], s: tuple[int, ...]) -> np.ndarray:
    n_u, n_p = len(LETTRES), len(PAIRES)
    x = np.zeros(1 + n_u + 2 * n_p)
    x[0] = 1.0
    a = sorted(IDX[pool[i]] for i in s)
    b = sorted(IDX[u] for i, u in enumerate(pool) if i not in s)
    for u in a:
        x[1 + u] += 1
    for w in b:
        x[1 + w] -= 1
    for p in combinations(a, 2):
        x[1 + n_u + PIDX[p]] += 1
    for p in combinations(b, 2):
        x[1 + n_u + PIDX[p]] -= 1
    for u in a:
        for w in b:
            x[1 + n_u + n_p + PIDX[(min(u, w), max(u, w))]] += 1 if u < w else -1
    return x


def matrice(pools) -> np.ndarray:
    """(tirages, 70, caractéristiques) du modèle d'armée."""
    return np.array([[_caracteristiques(p, s) for s in REPARTITIONS] for p in pools])


def ajuster(pools, L: np.ndarray, ridge: float = 1.0, X: np.ndarray | None = None) -> dict:
    """Ridge : effets propres (faible pénalité), synergies et contres (pénalité `ridge`)."""
    X = (matrice(pools) if X is None else X).reshape(-1, 1 + len(LETTRES) + 2 * len(PAIRES))
    y = L.reshape(-1)
    n_u = len(LETTRES)
    pen = np.full(X.shape[1], ridge)
    pen[0] = 0.0
    pen[1:1 + n_u] = 1e-3
    w = np.linalg.solve(X.T @ X + np.diag(pen), X.T @ y)
    pred = X @ w
    r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    Xm = X[:, :1 + n_u]
    wm = np.linalg.lstsq(Xm, y, rcond=None)[0]
    r2m = 1 - ((y - Xm @ wm) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    a = w[1:1 + n_u]
    a = a - a.mean()                                  # Σ a = 0 : unité « moyenne » = 0
    n_p = len(PAIRES)
    return {"c0": float(w[0]), "a": a, "s": w[1 + n_u:1 + n_u + n_p], "c": w[1 + n_u + n_p:],
            "r2": float(r2), "r2_effets_propres": float(r2m)}


# ------------------------------------------------------------------ draft optimal
def minimax_draft(L_tirage: np.ndarray) -> tuple[float, np.ndarray]:
    """Valeur du draft (point de vue de A) et valeur selon le premier choix de A (8 cartes)."""
    feuille = {sum(1 << i for i in s): L_tirage[r] for r, s in enumerate(REPARTITIONS)}
    ordre = DRAFT_ORDER["2J"]
    memo: dict[tuple[int, int], float] = {}

    def val(ma: int, mb: int, etape: int) -> float:
        if etape == 8:
            return feuille[ma]
        cle = (ma, mb)
        if cle in memo:
            return memo[cle]
        libres = [i for i in range(8) if not (ma | mb) >> i & 1]
        if ordre[etape] == 0:
            v = max(val(ma | 1 << i, mb, etape + 1) for i in libres)
        else:
            v = min(val(ma, mb | 1 << i, etape + 1) for i in libres)
        memo[cle] = v
        return v

    premiers = np.array([val(1 << i, 0, 1) for i in range(8)])
    return float(premiers.max()), premiers


def priorites(pools, L: np.ndarray, k: float) -> dict:
    n = len(LETTRES)
    premier = np.zeros(n)
    dispo = np.zeros(n)
    regret = np.zeros(n)
    valeurs = []
    for pool, lt in zip(pools, L):
        d, prem = minimax_draft(lt)
        valeurs.append(d)
        best = int(prem.argmax())
        for i, u in enumerate(pool):
            dispo[IDX[u]] += 1
            regret[IDX[u]] += d - prem[i]
        premier[IDX[pool[best]]] += 1
    dispo = np.maximum(dispo, 1)
    return {"premier_choix": premier / dispo, "regret_premier": points(regret / dispo, k),
            "avantage_premier_choix": float(points(np.mean(valeurs), k))}


# ------------------------------------------------------------------ préférences de l'auto-jeu
def preferences_autojeu(data: dict) -> dict | None:
    """π' moyen de chaque carte disponible, rapporté au choix uniforme (1 = neutre)."""
    if not data or "glob_i" not in data:
        return None
    dr = data["glob_i"][:, 0] == PENDING_ID["draft"]
    if not dr.any():
        return None
    acts, pi, na = data["acts"][dr], data["pi"][dr].astype(np.float64), data["n_act"][dr]
    lettre = {COIN_ID[u]: u for u in LETTRES}
    somme, nb = np.zeros(len(LETTRES)), np.zeros(len(LETTRES))
    somme1, nb1 = np.zeros(len(LETTRES)), np.zeros(len(LETTRES))
    premier = data["glob_f"][dr, 23] == 0
    for j in range(len(na)):
        n = int(na[j])
        tot = pi[j, :n].sum() or 1.0
        for a in range(n):
            u = IDX[lettre[int(acts[j, a, 2])]]
            r = pi[j, a] / tot * n
            somme[u] += r
            nb[u] += 1
            if premier[j]:
                somme1[u] += r
                nb1[u] += 1
    return {"choix": somme / np.maximum(nb, 1), "premier": somme1 / np.maximum(nb1, 1),
            "decisions": int(dr.sum())}


# ------------------------------------------------------------------ étape d'itération
def etape(dossier: Path, it: int, evaluateur, fenetre: list[dict], nouvelles: dict, cfg: dict,
          k_defaut: float) -> dict:
    """Calcule, enregistre (unites.jsonl, unites.json, echelle.json) et renvoie le résumé."""
    pools_n = int(cfg.get("pools", 256))
    pioches = int(cfg.get("pioches", 2))
    k = calibrer_k(fenetre) or k_defaut
    pools = tirages(pools_n, int(cfg.get("graine", 12345)))
    L = evaluer_sondes(evaluateur, pools, pioches)
    X = matrice(pools)
    ridge = float(cfg.get("ridge", 1.0))
    fit = ajuster(pools, L, ridge, X)
    rng = np.random.default_rng(it)
    boot = []
    for _ in range(int(cfg.get("bootstrap", 30))):
        idx = rng.integers(0, len(pools), len(pools))
        boot.append(ajuster(None, L[idx], ridge, X[idx])["a"])
    et = np.std(boot, axis=0) if boot else np.zeros(len(LETTRES))
    pr = priorites(pools, L, k)
    pref = preferences_autojeu(nouvelles)
    n_u = len(LETTRES)

    def paire(vec):
        return {f"{LETTRES[i]}{LETTRES[j]}": round(float(points(vec[k_], k)), 2)
                for k_, (i, j) in enumerate(PAIRES)}

    unites = {}
    for i, u in enumerate(LETTRES):
        unites[u] = {"points": round(float(points(fit["a"][i], k)), 2),
                     "ic95": round(float(1.96 * points(et[i], k)), 2),
                     "premier_choix": round(float(pr["premier_choix"][i]), 3),
                     "regret_premier": round(float(pr["regret_premier"][i]), 2)}
        if pref is not None:
            unites[u]["choix_autojeu"] = round(float(pref["choix"][i]), 3)
            unites[u]["premier_autojeu"] = round(float(pref["premier"][i]), 3)
    ligne = {"iteration": it, "K": round(k, 3), "unites": unites,
             "synergies": paire(fit["s"]), "contres": paire(fit["c"]),
             "commencer": round(float(points(-fit["c0"], k)), 2),
             "avantage_premier_choix": round(pr["avantage_premier_choix"], 2),
             "r2": round(fit["r2"], 3), "r2_effets_propres": round(fit["r2_effets_propres"], 3),
             "sondes": len(pools) * len(REPARTITIONS) * pioches,
             "decisions_draft": pref["decisions"] if pref else 0}
    assert n_u == 16
    from .entrainement import ecrire_atomique
    with open(dossier / "unites.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
    ecrire_atomique(dossier / "unites.json", json.dumps(ligne, indent=1, ensure_ascii=False))
    ecrire_atomique(dossier / "echelle.json", json.dumps({"iteration": it, "K": k}))
    return {"K": ligne["K"], "top": sorted(unites, key=lambda u: -unites[u]["points"])[:4],
            "avantage_premier_choix": ligne["avantage_premier_choix"], "r2": ligne["r2"]}


def lire_k(chemin_modele: str | None, defaut: float) -> float:
    """K de l'entraînement d'un modèle (runs/<nom>/echelle.json), sinon `defaut`."""
    if not chemin_modele:
        return defaut
    f = Path(chemin_modele).resolve().parent.parent / "echelle.json"
    try:
        k = float(json.loads(f.read_text())["K"])
        return k if math.isfinite(k) and k > 0 else defaut
    except (OSError, ValueError, KeyError):
        return defaut
