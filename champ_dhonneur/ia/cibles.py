"""Cibles de valeur à variance réduite : retours TD(λ) le long de chaque partie.

Le résultat final z d'une partie est une cible très bruitée dans ce jeu où le hasard des
pioches pèse lourd : la précision de la valeur de recherche sur le résultat n'est que de 0,56
en début de partie. Comme la valeur « à court terme » de KataGo, le retour λ d'une position est
la moyenne, à poids géométriques, des valeurs de recherche des positions suivantes de la même
partie (ramenées à son point de vue), puis du résultat final :

    G_t = (1 − λ) q_t + λ · σ_t · G_{t+1},      G après la dernière position = z

où σ_t = ±1 selon que la position suivante est jouée par la même équipe ou l'adversaire.
λ = 0 donne la valeur de recherche seule, λ = 1 le résultat final seul.

Les exemples d'une partie sont contigus et dans l'ordre (auto-jeu Python et Rust). Le début de
chaque partie est donné par la colonne `debut` quand elle existe ; sinon il est déduit de la
manche, qui ne décroît jamais au cours d'une partie. Les équipes se déduisent de z (même signe =
même équipe) ; les parties nulles (z = 0) gardent leur valeur de recherche.
"""
from __future__ import annotations

import numpy as np


def manches(data: dict) -> np.ndarray:
    """Manche de chaque exemple (caractéristique globale 18 = min(manche, 100) / 50)."""
    return np.rint(data["glob_f"][:, 18].astype(np.float32) * 50).astype(np.int32)


def debuts(data: dict) -> np.ndarray:
    """Vrai au premier exemple de chaque partie."""
    if "debut" in data:
        return data["debut"].astype(bool)
    m = manches(data)
    d = np.ones(len(m), bool)
    d[1:] = m[1:] < m[:-1]
    return d


def retours_lambda(data: dict, lam: float) -> np.ndarray:
    """Retour TD(λ) de chaque exemple, du point de vue de son joueur, dans [-1, 1]."""
    z = data["z"].astype(np.float64)
    q = data["q"].astype(np.float64)
    n = len(z)
    out = q.copy()
    if n == 0:
        return out.astype(np.float32)
    bornes = np.flatnonzero(debuts(data)).tolist() + [n]
    for a, b in zip(bornes[:-1], bornes[1:]):
        zs = z[a:b]
        if np.any(zs == 0) or not np.all(np.abs(zs) == 1):
            continue                    # partie nulle (ou frontière douteuse) : valeur de recherche
        g = zs[-1]                      # après la dernière position : le résultat final
        for t in range(b - 1, a - 1, -1):
            if t < b - 1:
                g *= z[t] * z[t + 1]    # changement de point de vue (±1)
            g = (1 - lam) * q[t] + lam * g
            out[t] = g
    return np.clip(out, -1, 1).astype(np.float32)
