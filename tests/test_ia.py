"""Tests de la partie IA. Ceux qui exigent PyTorch sont ignorés s'il est absent."""
import random

import numpy as np
import pytest

from champ_dhonneur.engine import Game
from champ_dhonneur.ia.autojeu import ParamsAutoJeu, jouer_parties
from champ_dhonneur.ia.encodage import (NONE_CELL, PERM, SPEC, encode_actions, encode_state)
from champ_dhonneur.ia.evaluateurs import EvaluateurHeuristique, EvaluateurUniforme
from champ_dhonneur.ia.evaluation import ClassementElo
from champ_dhonneur.ia.recherche import ParamsRecherche, RechercheGumbel, executer


def partie_avancee(seed=3, n=40):
    g = Game(seed=seed)
    rng = random.Random(seed)
    for _ in range(n):
        if g.done:
            break
        g.apply(rng.choice(g.legal_actions()))
    return g


def test_rotation_involution_et_symetrie():
    rot = PERM[1]
    assert all(rot[rot[i]] == i for i in range(SPEC.n_cells))
    for i in range(SPEC.n_cells):
        assert sorted(rot[j] for j in SPEC.neighbors[i]) == sorted(SPEC.neighbors[rot[i]])
    blanc = {SPEC.index[c] for c in SPEC.starts[0]}
    noir = {SPEC.index[c] for c in SPEC.starts[1]}
    assert {rot[c] for c in blanc} == noir
    assert sorted(rot[c] for c in SPEC.locations) == SPEC.locations


def test_position_initiale_symetrique():
    """Au départ, Blanc et Noir voient exactement le même plateau (à l'armée près)."""
    g = Game(seed=1, units=[list("ACLE"), list("ACLE")])
    e0 = encode_state(g)
    g.current = 1 - g.current
    e1 = encode_state(g)
    assert (e0["cell_i"] == e1["cell_i"]).all()


def test_pas_de_fuite_d_information():
    """La main adverse ne doit pas influencer l'observation."""
    g = partie_avancee()
    p = g.to_move
    e = encode_state(g)
    for s in range(5):
        d = g.determinize(p, random.Random(s))
        e2 = encode_state(d)
        for k in e:
            assert np.array_equal(e[k], e2[k]), k


def test_encodage_actions():
    g = partie_avancee(seed=8, n=25)
    legal = g.legal_actions()
    a = encode_actions(g, legal)
    assert a.shape == (len(legal), 7)
    assert len({tuple(r) for r in a}) == len(legal), "deux actions légales ont le même encodage"
    assert (a[:, 4:] <= NONE_CELL).all()


@pytest.mark.parametrize("par", [1, 4])
def test_recherche_heuristique(par):
    g = partie_avancee(seed=5, n=10)
    r = RechercheGumbel(ParamsRecherche(simulations=40, parallele=par, bruit=True), seed=0)
    res = executer([r.generateur(g)], EvaluateurHeuristique())[0]
    assert res.action in g.legal_actions()
    assert abs(res.politique.sum() - 1) < 1e-4
    assert res.simulations == 40
    assert -1 <= res.valeur <= 1


def test_recherche_trouve_la_victoire():
    """Un seul marqueur manque : la recherche doit contrôler le Lieu gagnant."""
    for seed in range(40):
        g = Game(seed=seed)
        rng = random.Random(seed)
        while not g.done:
            legal = g.legal_actions()
            wins = [a for a in legal if a.kind == "control"
                    and g.markers_left[g.team(g.to_move)] == 1]
            if wins and not g.pending:
                r = RechercheGumbel(ParamsRecherche(simulations=64, m=64, bruit=False), seed=0)
                res = executer([r.generateur(g)], EvaluateurUniforme())[0]
                assert res.action.kind == "control"
                return
            g.apply(rng.choice(legal))
    pytest.skip("aucune position de victoire immédiate rencontrée")


def test_autojeu_heuristique():
    data, st = jouer_parties(EvaluateurHeuristique(), ParamsAutoJeu(
        parties=2, simultanees=2, simulations=8, simulations_rapides=4, p_complete=1.0, max_manches=12),
        seed=1)
    assert st["parties"] == 2
    n = len(data["z"])
    assert n > 10
    assert data["acts"].shape == (n, 64, 7)
    assert np.allclose(data["pi"].astype(np.float32).sum(1), 1, atol=1e-2)
    assert set(np.unique(data["z"])) <= {-1.0, 0.0, 1.0}


def test_releves_autojeu_et_evaluation_rejouables():
    from champ_dhonneur.ia.evaluation import agent_depuis_spec, match
    from champ_dhonneur.notation import import_record, parse_headers
    _, st = jouer_parties(EvaluateurHeuristique(), ParamsAutoJeu(
        parties=3, simultanees=3, simulations=8, simulations_rapides=4, max_manches=15, releves=2),
        seed=4)
    r = match(agent_depuis_spec("glouton"), agent_depuis_spec("heur:8"), paires=1, seed=2,
              max_manches=20, releves=2)
    textes = st["releves"] + r["releves"]
    assert len(textes) == 4
    for t in textes:
        g = import_record(t)
        assert g.done and g.result_label() == parse_headers(t)["Resultat"]
    assert parse_headers(r["releves"][0])["Type"] == "evaluation"


def test_classement_elo():
    c = ClassementElo("/tmp/_elo_test.json")
    c.resultats = []
    c.ajouter("a", "glouton", 30, 40)   # 75 %
    elo = c.ajuster()
    assert elo["glouton"] == 0
    assert 120 < elo["a"] < 220


torch = pytest.importorskip("torch")


def test_reseau_et_apprentissage(tmp_path):
    from champ_dhonneur.ia.entrainement import ConfigEntrainement, Entraineur
    from champ_dhonneur.ia.evaluateurs import EvaluateurReseau
    from champ_dhonneur.ia.modele import ConfigModele, ReseauChamp, charger, sauver

    m = ReseauChamp(ConfigModele(d=32, couches=1, tetes=2))
    ev = EvaluateurReseau(m)
    g = partie_avancee()
    [(logits, v)] = ev.evaluer([(g, g.legal_actions())])
    assert len(logits) == len(g.legal_actions()) and -1 <= v <= 1
    sauver(tmp_path / "m.pt", m)
    m2, _ = charger(tmp_path / "m.pt")
    [(l2, v2)] = EvaluateurReseau(m2).evaluer([(g, g.legal_actions())])
    assert np.allclose(logits, l2, atol=1e-5)

    cfg = ConfigEntrainement()
    cfg.appliquer([f"dossier={tmp_path / 'run'}", "iterations=2", "travailleurs=0",
                   "parties_par_iteration=2", "amorce_iterations=1", "amorce_simulations=6",
                   "autojeu.simulations=8", "autojeu.simulations_rapides=4", "autojeu.max_manches=10",
                   "fenetre_min=10", "lot=16", "eval_tous=2", "eval_paires=1", "eval_simulations=4",
                   '"modele"={"d": 32, "couches": 1, "tetes": 2}'.replace('"modele"', "modele")])
    Entraineur(cfg).executer()
    assert (tmp_path / "run" / "modeles" / "meilleur.pt").exists()
    assert (tmp_path / "run" / "journal.jsonl").read_text().count("\n") == 2


def test_bot_ia(tmp_path):
    from champ_dhonneur.bots import make_bot
    from champ_dhonneur.ia.modele import ConfigModele, ReseauChamp, sauver
    sauver(tmp_path / "m.pt", ReseauChamp(ConfigModele(d=32, couches=1, tetes=2)))
    bot = make_bot(f"ia:modele={tmp_path / 'm.pt'},sims=16")
    g = partie_avancee(seed=4, n=6)
    assert bot.choose(g) in g.legal_actions()
    assert bot.analyse()


def test_configs_valides():
    import json
    from pathlib import Path

    from champ_dhonneur.ia.entrainement import ConfigEntrainement
    for f in Path("configs").glob("*.json"):
        cfg = ConfigEntrainement.depuis_dict(json.loads(f.read_text(encoding="utf-8")))
        assert cfg.modele.d % cfg.modele.tetes == 0, f
        if cfg.dispositif_autojeu == "cuda":
            # au moins 3 vagues de parties simultanées par processus (7 par défaut : portable 8 cœurs)
            n = cfg.travailleurs if cfg.travailleurs > 0 else 7
            assert cfg.parties_par_iteration / n >= 3 * cfg.autojeu.simultanees, f


def test_travailleurs_auto():
    from champ_dhonneur.ia.entrainement import coeurs_physiques, travailleurs_auto
    assert coeurs_physiques() >= 1
    assert 1 <= travailleurs_auto() <= max(1, coeurs_physiques())


def test_match_parallele_identique_au_match_sequentiel():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    from champ_dhonneur.ia.evaluation import agent_depuis_spec, match, match_parallele
    seq = match(agent_depuis_spec("glouton"), agent_depuis_spec("aleatoire"), paires=3, seed=5,
                max_manches=30)
    with ProcessPoolExecutor(2, mp_context=mp.get_context("spawn")) as ex:
        par = match_parallele("glouton", "aleatoire", 3, 5, ex, 2, max_manches=30)
    assert par["parties"] == seq["parties"] == 6
    assert seq["victoires"] >= 4 and par["victoires"] >= 4


def test_determinisation_rapide():
    g = partie_avancee(seed=9, n=30)
    rng = random.Random(1)
    d = g.determinize(g.to_move, rng, rapide=True)
    assert d.rng is rng and d.log == []
    assert encode_state(d)["cell_i"].tolist() == encode_state(g)["cell_i"].tolist()
    while not d.done and d.round < g.round + 3:
        d.apply(d.legal_actions()[0])
