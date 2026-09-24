"""Banc d'essai matériel : calibre l'auto-jeu pour la machine locale.

    champ banc --config configs/rtx-a5000-laptop.json

1. inférence seule : débit du réseau (positions/s) selon la taille des lots ;
2. auto-jeu réel : débit total (évaluations/s) pour plusieurs nombres de processus
   lancés en même temps, avec la configuration donnée ;
3. recommandation : nombre de processus, et alerte si le GPU devient le goulot.
"""
from __future__ import annotations

import os
import random
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

# décisions nécessitant une recherche dans une partie typique (mesuré en démonstration)
RECHERCHES_PAR_PARTIE = 95


def _positions(n: int = 256):
    from ..engine import Game
    rng = random.Random(0)
    out = []
    s = 0
    while len(out) < n:
        g = Game(seed=s)
        s += 1
        while not g.done and len(out) < n:
            legal = g.legal_actions()
            if len(legal) > 1 and rng.random() < 0.3:
                out.append((g.copy(), legal))
            g.apply(rng.choice(legal))
    return out


def banc_inference(cfg_modele, dispositif: str, lots=(64, 128, 256, 512, 1024), repetitions: int = 20):
    """Débit du réseau seul (entrées déjà encodées, comme dans l'auto-jeu du moteur Rust)."""
    import torch

    from .encodage import collate, encode_actions, encode_state
    from .evaluateurs import EvaluateurReseau
    from .modele import ReseauChamp, nb_parametres
    model = ReseauChamp(cfg_modele)
    n_par = nb_parametres(model)
    ev = EvaluateurReseau(model, dispositif)
    pos = _positions(max(lots))
    x_tout, _ = collate([encode_state(g) for g, _ in pos], [encode_actions(g, l) for g, l in pos])
    res = []
    for b in lots:
        x = {k: v[:b] for k, v in x_tout.items()}
        ev.evaluer_lot(x)
        if ev.device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(repetitions):
            ev.evaluer_lot(x)
        if ev.device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / repetitions
        res.append({"lot": b, "ms_par_lot": dt * 1e3, "positions_par_s": b / dt})
    return n_par, res


def _travailleur_banc(args):
    chemin, dispositif, params, seed, depart, moteur = args
    from .entrainement import _init_travailleur
    _init_travailleur()
    import torch

    from .autojeu import jouer
    from .evaluateurs import EvaluateurReseau
    ev = EvaluateurReseau.depuis_fichier(chemin, dispositif)
    ev.evaluer(_positions(8))                 # initialisation CUDA / premiers noyaux
    time.sleep(max(0.0, depart - time.time()))
    _, st = jouer(ev, params, seed, moteur)
    mem = torch.cuda.max_memory_allocated() / 2**20 if ev.device.type == "cuda" else 0.0
    return st["evaluations"], st["secondes"], st["decisions"], mem


def banc_autojeu(cfg, liste_travailleurs, duree: float = 45.0):
    import multiprocessing as mp

    from .autojeu import ParamsAutoJeu
    from .entrainement import choisir_dispositif
    from .modele import ReseauChamp, sauver
    dispositif = choisir_dispositif(cfg.dispositif_autojeu)
    tmp = tempfile.mkdtemp(prefix="champ_banc_")
    chemin = os.path.join(tmp, "modele.pt")
    sauver(chemin, ReseauChamp(cfg.modele))
    lignes = []
    for n in liste_travailleurs:
        p = ParamsAutoJeu(**asdict(cfg.autojeu))
        p.parties = 10_000
        p.duree_max = duree
        p.max_manches = 25     # réseau aléatoire : on évite les parties interminables
        with ProcessPoolExecutor(n, mp_context=mp.get_context("spawn")) as ex:
            depart = time.time() + 20 + 2 * n      # tous les processus démarrent ensemble
            taches = [(chemin, dispositif, p, 1000 + i, depart, cfg.moteur_autojeu) for i in range(n)]
            res = list(ex.map(_travailleur_banc, taches))
        ev_s = sum(e / max(s, 1e-9) for e, s, _, _ in res)
        sims_moy = p.p_complete * p.simulations + (1 - p.p_complete) * p.simulations_rapides
        parties_h = ev_s * 3600 / (sims_moy * RECHERCHES_PAR_PARTIE)
        lignes.append({"travailleurs": n, "evaluations_par_s": ev_s, "par_processus": ev_s / n,
                       "parties_par_heure": parties_h, "vram_mo_par_processus": max(r[3] for r in res)})
        print(f"  {n:>2} processus : {ev_s:8.0f} évaluations/s ({ev_s / n:6.0f} par processus) "
              f"≈ {parties_h:7.0f} parties/h", flush=True)
    return lignes


def executer_banc(cfg, liste_travailleurs=None, duree: float = 45.0) -> dict:
    import torch

    from .entrainement import choisir_dispositif, coeurs_physiques, travailleurs_auto
    dispositif = choisir_dispositif(cfg.dispositif_autojeu)
    print("Machine :")
    print(f"  CPU : {coeurs_physiques()} cœurs physiques, {os.cpu_count()} fils d'exécution")
    print(f"  PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        print(f"  GPU : {pr.name}, {pr.total_memory / 2**30:.1f} Go, capacité {pr.major}.{pr.minor}, "
              f"bf16 {'oui' if torch.cuda.is_bf16_supported() else 'non'}")
    else:
        print("  GPU : aucun (CUDA indisponible)")
    from . import rs
    rust = cfg.moteur_autojeu != "python" and rs.disponible()
    print(f"  Moteur d'auto-jeu : {'Rust (champ_rs)' if rust else 'Python'}")
    print(f"\n1. Réseau seul, entrées déjà encodées ({dispositif}) :")
    n_par, inf = banc_inference(cfg.modele, dispositif)
    print(f"  réseau : {n_par / 1e6:.2f} M paramètres")
    for r in inf:
        print(f"  lot {r['lot']:>4} : {r['ms_par_lot']:6.2f} ms  → {r['positions_par_s']:9.0f} positions/s")
    if liste_travailleurs is None:
        base = travailleurs_auto()
        if rust:   # le GPU devient le goulot : quelques processus aux gros lots suffisent
            liste_travailleurs = [1, 2, 3, 4, 6]
        else:
            liste_travailleurs = sorted({max(1, base // 2), base, base + 1, min(os.cpu_count() or 1, base * 2)})
    print(f"\n2. Auto-jeu réel ({duree:.0f} s par essai, {cfg.autojeu.simultanees} parties simultanées "
          f"par processus) :")
    lignes = banc_autojeu(cfg, liste_travailleurs, duree)
    meilleur = max(lignes, key=lambda l: l["evaluations_par_s"])
    # plus petit nombre de processus à 5 % du meilleur débit (moins de chaleur, plus de marge)
    choix = min((l for l in lignes if l["evaluations_par_s"] >= 0.95 * meilleur["evaluations_par_s"]),
                key=lambda l: l["travailleurs"])
    lot = min(cfg.autojeu.simultanees, max(r["lot"] for r in inf))
    capacite = max(r["positions_par_s"] for r in inf if r["lot"] <= max(lot, 16))
    charge = choix["evaluations_par_s"] / capacite
    print("\n3. Recommandation :")
    print(f"  travailleurs = {choix['travailleurs']}  (≈ {choix['parties_par_heure']:.0f} parties/h)")
    print(f"  charge estimée du {dispositif} d'inférence : {100 * charge:.0f} % de sa capacité")
    if charge > 0.7:
        print("  → le GPU devient un goulot : augmentez autojeu.simultanees (lots plus gros) "
              "ou réduisez la taille du réseau.")
    elif charge < 0.15 and dispositif == "cuda":
        print("  → le GPU a de la marge : un réseau plus gros ne ralentirait presque pas l'auto-jeu.")
    return {"inference": inf, "autojeu": lignes, "recommandation": choix, "charge_gpu": charge}
