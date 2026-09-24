"""Auto-jeu : génère des parties et des exemples d'entraînement.

Des dizaines de parties avancent en même temps dans un même processus ; les
requêtes de toutes leurs recherches sont regroupées en un lot unique pour le
réseau (GPU ou CPU).

Techniques utilisées :
  * recherche Gumbel IS-MCTS avec bruit de Gumbel (exploration) ;
  * *playout cap randomization* (KataGo) : une fraction des coups reçoit une
    recherche complète et produit une cible de politique ; les autres coups,
    joués avec une recherche rapide, ne servent qu'à la cible de valeur, ce
    qui multiplie les parties jouées pour un même budget ;
  * décisions forcées (une seule action légale) jouées sans recherche ni
    exemple enregistré.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np

from ..engine import Game
from ..notation import export_record
from .direct import Diffuseur
from .encodage import ACT_F, NONE_CELL, STATE_KEYS, encode_actions, encode_state
from .recherche import ParamsRecherche, RechercheGumbel

A_MAX = 64   # actions stockées par exemple (le maximum observé est ~45)


@dataclass
class ParamsAutoJeu:
    parties: int = 64
    simultanees: int = 64
    simulations: int = 64
    simulations_rapides: int = 16
    p_complete: float = 0.25
    max_manches: int = 80
    m: int = 16
    parallele: int = 1
    c_visit: float = 50.0
    c_scale: float = 0.1
    duree_max: float = 0.0     # > 0 : arrêt après ce nombre de secondes (mesures de débit)
    releves: int = 0           # relevés .nch des premières parties terminées (visualisation)
    direct: str = ""           # fichier de diffusion d'une partie en cours (ia/direct.py), "" = aucune


def jouer_parties(evaluateur, P: ParamsAutoJeu, seed: int | None = None) -> tuple[dict, dict]:
    rng = random.Random(seed)
    rech = RechercheGumbel(ParamsRecherche(simulations=P.simulations, m=P.m, bruit=True,
                                           parallele=P.parallele, c_visit=P.c_visit,
                                           c_scale=P.c_scale), seed=rng.randrange(2**31))
    exemples: list[dict] = []
    releves: list[str] = []
    stats = {"parties": 0, "victoires_blanc": 0, "victoires_noir": 0, "nulles": 0,
             "manches": 0, "decisions": 0, "recherches": 0, "evaluations": 0}
    slots: list[dict] = []
    lances = 0
    t0 = time.time()
    diffuseur = Diffuseur(P.direct, P.parties) if P.direct else None

    def nouvelle():
        g = Game("2J", seed=rng.randrange(2**31))
        g.max_rounds = P.max_manches
        return {"g": g, "ex": [], "gen": None, "req": None, "complet": False}

    def terminer(slot):
        g = slot["g"]
        stats["parties"] += 1
        stats["manches"] += g.round
        if g.winner is None:
            stats["nulles"] += 1
        elif g.winner == 0:
            stats["victoires_blanc"] += 1
        else:
            stats["victoires_noir"] += 1
        for ex in slot["ex"]:
            ex["z"] = 0.0 if g.winner is None else (1.0 if g.winner == ex["equipe"] else -1.0)
            exemples.append(ex)
        if len(releves) < P.releves:
            # relevé complet (pièces cachées incluses) : il se rejoue à l'identique
            releves.append(export_record(g, headers={"Type": "autojeu"}))

    while slots or lances < P.parties:
        if P.duree_max and time.time() - t0 > P.duree_max:
            break   # mesure de débit : on abandonne les parties en cours
        while len(slots) < P.simultanees and lances < P.parties:
            slots.append(nouvelle())
            lances += 1
        # faire avancer chaque partie jusqu'à sa prochaine vraie décision
        for slot in list(slots):
            if slot["gen"] is not None:
                continue
            g = slot["g"]
            while True:
                if g.done:
                    terminer(slot)
                    slots.remove(slot)
                    break
                legal = g.legal_actions()
                if len(legal) == 1:
                    g.apply(legal[0])
                    stats["decisions"] += 1
                    continue
                slot["complet"] = rng.random() < P.p_complete
                sims = P.simulations if slot["complet"] else P.simulations_rapides
                slot["gen"] = rech.generateur(g, sims)
                slot["req"] = next(slot["gen"])
                break
        if diffuseur is not None:
            diffuseur.observer([s["g"] for s in slots], stats["parties"])
        if not slots:
            continue
        flat = [r for s in slots for r in s["req"]]
        reps = evaluateur.evaluer(flat)
        stats["evaluations"] += len(flat)
        pos = 0
        for slot in slots:
            n = len(slot["req"])
            try:
                slot["req"] = slot["gen"].send(reps[pos:pos + n])
            except StopIteration as fin:
                res = fin.value
                g = slot["g"]
                if slot["complet"]:
                    ex = encode_state(g)
                    ex["acts"] = encode_actions(g, res.legal)
                    ex["pi"] = res.politique
                    ex["q"] = res.valeur
                    ex["equipe"] = g.team(g.to_move)
                    slot["ex"].append(ex)
                g.apply(res.action)
                stats["decisions"] += 1
                stats["recherches"] += 1
                slot["gen"] = slot["req"] = None
            pos += n
    stats["secondes"] = time.time() - t0
    stats["releves"] = releves
    return empaqueter(exemples), stats


def jouer(evaluateur, P: ParamsAutoJeu, seed: int | None = None, moteur: str = "auto") -> tuple[dict, dict]:
    """Auto-jeu avec le moteur Rust (champ_rs) si possible, sinon le moteur Python.

    `moteur` : "auto" (Rust si le module est compilé et que l'évaluateur traite des lots déjà
    encodés, c'est-à-dire un réseau), "rust" ou "python". L'amorçage heuristique reste en Python.
    """
    from . import rs
    rust = moteur != "python" and rs.disponible() and hasattr(evaluateur, "evaluer_lot")
    if moteur == "rust" and not rust:
        raise RuntimeError("moteur Rust demandé mais indisponible (module champ_rs absent ?)")
    return rs.jouer_parties_rs(evaluateur, P, seed) if rust else jouer_parties(evaluateur, P, seed)


def empaqueter(exemples: list[dict]) -> dict[str, np.ndarray]:
    """Liste d'exemples -> tableaux numpy compacts de taille fixe (A_MAX actions)."""
    n = len(exemples)
    out = {}
    for k in STATE_KEYS:
        if n:
            arr = np.stack([e[k] for e in exemples])
            out[k] = arr.astype(np.int8 if arr.dtype.kind == "i" else np.float16)
        else:
            out[k] = np.zeros((0,), np.int8)
    acts = np.zeros((n, A_MAX, ACT_F), np.int8)
    acts[:, :, 4:] = NONE_CELL
    pi = np.zeros((n, A_MAX), np.float16)
    n_act = np.zeros(n, np.int16)
    for i, e in enumerate(exemples):
        a, p = e["acts"], e["pi"]
        if len(a) > A_MAX:                   # très rare : on garde les plus probables
            keep = np.argsort(-p)[:A_MAX]
            a, p = a[keep], p[keep] / p[keep].sum()
        acts[i, :len(a)] = a
        pi[i, :len(a)] = p
        n_act[i] = len(a)
    out["acts"], out["pi"], out["n_act"] = acts, pi, n_act
    out["z"] = np.array([e["z"] for e in exemples], np.float32)
    out["q"] = np.array([e["q"] for e in exemples], np.float32)
    return out


def concatener(lots: list[dict]) -> dict[str, np.ndarray]:
    lots = [l for l in lots if len(l["z"])]
    if not lots:
        return {}
    return {k: np.concatenate([l[k] for l in lots]) for k in lots[0]}
