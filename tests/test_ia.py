"""Tests de la partie IA. Ceux qui exigent PyTorch sont ignorés s'il est absent."""
import random

import numpy as np
import pytest

from champ_dhonneur.engine import Game
from champ_dhonneur.ia.autojeu import ParamsAutoJeu, jouer_parties
from champ_dhonneur.ia.encodage import (NONE_CELL, PERM, SPEC, encode_actions, encode_state)
from champ_dhonneur.ia.evaluateurs import EvaluateurHeuristique, EvaluateurUniforme
from champ_dhonneur.ia.evaluation import ClassementElo
from champ_dhonneur.ia.recherche import (Limite, ParamsRecherche, RechercheGumbel, choisir, classement,
                                          coups_gagnants, executer)


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


def test_encodage_manche_independant_de_la_limite():
    """La même position s'encode pareil en auto-jeu (100 manches) et en jeu (150)."""
    g1, g2 = Game(seed=2, max_rounds=100), Game(seed=2, max_rounds=150)
    rng = random.Random(0)
    for _ in range(30):
        a = rng.choice(g1.legal_actions())
        g1.apply(a)
        g2.apply(a)
    e1, e2 = encode_state(g1), encode_state(g2)
    for k in e1:
        assert np.array_equal(e1[k], e2[k]), k


def test_arbre_sans_piece_face_cachee_adverse():
    """Une action face cachée d'un autre joueur mène au même nœud quelle que soit la pièce (que
    l'observateur ne voit pas) ; celles de l'observateur gardent leur pièce."""
    from champ_dhonneur.engine import FACE_DOWN
    from champ_dhonneur.ia.recherche import CACHEE
    adverses = 0
    for seed in range(4):
        g = partie_avancee(seed=seed, n=20)
        if g.done:
            continue
        r = RechercheGumbel(ParamsRecherche(simulations=300, bruit=False), seed=0)
        executer([r.generateur(g)], EvaluateurUniforme())
        moi = g.team(g.to_move)
        pile = [r.racine]
        while pile:
            nd = pile.pop()
            for a, ch in nd.enfants.items():
                if a.kind in FACE_DOWN:
                    adverse = ch.equipe != moi
                    assert (a.coin == CACHEE) == adverse, a
                    adverses += adverse
                pile.append(ch)
    assert adverses > 0
    # ancienne recherche (comparaisons seulement) : la pièce de l'adversaire fait partie de l'arête
    g = partie_avancee(seed=0, n=20)
    r = RechercheGumbel(ParamsRecherche(simulations=300, bruit=False, cle_publique=False), seed=0)
    executer([r.generateur(g)], EvaluateurUniforme())
    pile, cachees = [r.racine], 0
    while pile:
        nd = pile.pop()
        cachees += sum(a.coin == CACHEE for a in nd.enfants)
        pile.extend(nd.enfants.values())
    assert cachees == 0


def test_ligne_avec_action_face_cachee_adverse():
    """L'analyse rejoue une arête face cachée avec une pièce que le joueur peut détenir."""
    from champ_dhonneur.engine import PASS, Action
    from champ_dhonneur.ia.analyse import _representant
    from champ_dhonneur.ia.recherche import CACHEE
    g = partie_avancee(seed=4, n=15)
    while g.pending or not g.players[g.to_move].hand:
        g.apply(g.legal_actions()[0])
    b = _representant(g, Action(PASS, CACHEE))
    assert b is not None and b.kind == PASS and b in g.legal_actions()


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


def position_gagnante():
    """Position où un Contrôle gagne immédiatement (dernier marqueur de l'équipe au trait)."""
    for seed in range(60):
        g = Game(seed=seed)
        rng = random.Random(seed)
        while not g.done:
            legal = g.legal_actions()
            if not g.pending and coups_gagnants(g, legal):
                return g
            g.apply(rng.choice(legal))
    pytest.skip("aucune position de victoire immédiate rencontrée")


def test_coup_gagnant_toujours_joue():
    g = position_gagnante()
    legal = g.legal_actions()
    gagnants = coups_gagnants(g, legal)
    for i in gagnants:
        h = g.copy()
        h.apply(legal[i])
        assert h.done and h.winner == g.team(g.to_move)
    # même avec une recherche minuscule, bruitée et limitée à deux candidats
    for graine in range(5):
        r = RechercheGumbel(ParamsRecherche(simulations=4, m=2, bruit=True, bruit_coup=False, coup_gagnant=True),
                            seed=graine)
        res = executer([r.generateur(g)], EvaluateurUniforme())[0]
        assert legal.index(res.action) in gagnants


def test_choix_et_classement_coherents():
    """Le coup joué est le meilleur score parmi les plus explorés ; l'analyse le classe premier."""
    visites = np.array([10, 100, 90, 3, 0.0])
    score = np.array([9.0, 1.0, 2.0, 5.0, 8.0])
    assert choisir(visites, score) == 2
    assert classement(visites, score) == [2, 1, 0, 3, 4]
    assert choisir(np.zeros(3), np.array([0.0, 2.0, 1.0])) == 1      # sans visite : l'a priori


def test_recherche_progressive_profondeur():
    g = partie_avancee(seed=5, n=10)
    rapports = []
    r = RechercheGumbel(ParamsRecherche(parallele=4, bruit=False), seed=0)
    res = executer([r.approfondir(g, Limite(profondeur=4), suivi=rapports.append)], EvaluateurHeuristique())[0]
    assert res.infos["profondeur"] == 4 and not res.infos["en_cours"]
    assert res.simulations == 32 * (2 ** 4 - 1)          # chaque profondeur double le budget
    assert [x.infos["profondeur"] for x in rapports if x.infos["en_cours"]] == [1, 2, 3]
    assert rapports[-1] is res
    assert res.infos["ordre"][0] == res.legal.index(res.action)
    assert 1 <= res.infos["horizon"] <= res.infos["horizon_max"]
    assert res.action in g.legal_actions() and abs(res.politique.sum() - 1) < 1e-4


def test_recherche_progressive_limites_et_arret():
    import threading
    g = partie_avancee(seed=5, n=10)
    ev = EvaluateurUniforme()
    nouvelle = lambda **kw: RechercheGumbel(ParamsRecherche(parallele=4, bruit=False, **kw), seed=0)  # noqa: E731
    res = executer([nouvelle().approfondir(g, Limite(simulations=100))], ev)[0]
    assert 100 <= res.simulations < 104
    res = executer([nouvelle().approfondir(g, Limite(secondes=0.3))], ev)[0]
    assert 0.3 <= res.infos["secondes"] < 3
    # analyse infinie : arrêtée de l'extérieur (ici dès la profondeur 2 atteinte)
    arret = threading.Event()
    suivi = lambda r: arret.set() if r.infos["profondeur"] >= 2 else None  # noqa: E731
    res = executer([nouvelle().approfondir(g, Limite(arret=arret), suivi=suivi)], ev)[0]
    assert res.infos["profondeur"] == 2
    # longue recherche : l'arbre est élagué et la recherche continue
    r = nouvelle(max_noeuds=300)
    res = executer([r.approfondir(g, Limite(profondeur=5))], ev)[0]
    assert res.infos["elagages"] >= 1 and res.simulations == 32 * (2 ** 5 - 1)
    assert res.action in g.legal_actions()


def test_bot_au_temps_et_victoire_forcee():
    from champ_dhonneur.bots.neural import NeuralBot
    g = partie_avancee(seed=5, n=10)
    bot = NeuralBot(heuristique=True, temps=0.3, seed=0)
    assert bot.choose(g) in g.legal_actions()
    assert bot.derniere.infos["profondeur"] >= 1 and bot.derniere.infos["secondes"] >= 0.3
    g = position_gagnante()
    bot = NeuralBot(heuristique=True, simulations=8, seed=0)
    a = bot.choose(g)
    h = g.copy()
    h.apply(a)
    assert h.done and h.winner == g.team(g.to_move) and bot.mat is not None


def test_autojeu_meilleur_coup():
    data, st = jouer_parties(EvaluateurHeuristique(), ParamsAutoJeu(
        parties=2, simultanees=2, simulations=8, simulations_rapides=4, p_complete=0.5, max_manches=12,
        meilleur_coup=True), seed=1)
    assert st["parties"] == 2 and len(data["z"]) > 0


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


def test_statistiques_par_paire_et_sprt():
    from champ_dhonneur.ia.evaluation import (bornes_sprt, llr_sprt, pentanomial, resume, spec_reseau,
                                              stats_paires)
    # (graine, camp de A, score de A, manches) : 3 paires gagnées 2-0, 1 paire partagée, 1 paire perdue
    parties = [(1, 0, 1, 10), (1, 1, 1, 10), (2, 0, 1, 9), (2, 1, 1, 9), (3, 0, 1, 8), (3, 1, 1, 8),
               (4, 0, 1, 12), (4, 1, -1, 12), (5, 0, -1, 7), (5, 1, -1, 7)]
    assert pentanomial(parties) == [1, 0, 1, 0, 3]
    r = resume(parties)
    assert (r["victoires"], r["defaites"], r["parties"], r["paires"]) == (7, 3, 10, 5)
    assert r["elo_bas"] < r["elo"] < r["elo_haut"] and r["elo"] > 0
    egal = stats_paires([10, 0, 20, 0, 10])
    assert egal["elo"] == 0 and egal["elo_bas"] < 0 < egal["elo_haut"]
    bas, haut = bornes_sprt(0.05, 0.05)
    assert bas < 0 < haut
    assert llr_sprt([2, 10, 30, 30, 28], 0, 20) > 0 > llr_sprt([28, 30, 30, 10, 2], 0, 20)
    assert spec_reseau("glouton") is None
    assert spec_reseau("m/iter_0171.pt") == ("m/iter_0171.pt", {})
    assert spec_reseau("iter.pt:simulations=512,cle_publique=false") == (
        "iter.pt", {"simulations": 512, "cle_publique": False})


def test_retours_lambda():
    """TD(λ) le long de chaque partie, points de vue alternés, parties nulles laissées à q."""
    from champ_dhonneur.ia.cibles import retours_lambda
    from champ_dhonneur.ia.encodage import GLOB_F
    # partie 1 : équipes 0, 1, 0, l'équipe 0 gagne ; partie 2 : nulle
    z = np.array([1, -1, 1, 0, 0], np.float32)
    q = np.array([0.2, -0.4, 0.6, 0.1, -0.3], np.float32)
    data = {"z": z, "q": q, "debut": np.array([1, 0, 0, 1, 0], np.int8)}
    assert np.allclose(retours_lambda(data, 0.0), q)
    assert np.allclose(retours_lambda(data, 1.0), [1, -1, 1, 0.1, -0.3])
    g = retours_lambda(data, 0.5)
    assert np.allclose(g, [0.4, -0.6, 0.8, 0.1, -0.3])
    # anciens fichiers sans colonne « debut » : frontières déduites de la manche
    glob_f = np.zeros((5, GLOB_F), np.float16)
    glob_f[:, 18] = np.array([1, 3, 5, 1, 2]) / 50
    assert np.allclose(retours_lambda({"z": z, "q": q, "glob_f": glob_f}, 0.5), g)


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


def test_reinitialisation(tmp_path):
    """Réseau neuf entraîné depuis zéro sur la fenêtre, qui remplace l'ancien (optimiseur neuf) ;
    auto-jeu joué par le meilleur réseau."""
    from champ_dhonneur.ia.entrainement import ConfigEntrainement, Entraineur
    cfg = ConfigEntrainement()
    cfg.appliquer([f"dossier={tmp_path / 'run'}", "iterations=3", "travailleurs=0",
                   "parties_par_iteration=2", "amorce_iterations=1", "amorce_simulations=6",
                   "autojeu.simulations=8", "autojeu.simulations_rapides=4", "autojeu.max_manches=10",
                   "fenetre_min=10", "lot=16", "eval_tous=1", "eval_paires=1", "eval_simulations=4",
                   "reinit_tous=3", "reinit_reutilisation=1", "autojeu_reseau=meilleur",
                   "ema=0.9", '"modele"={"d": 32, "couches": 1, "tetes": 2}'.replace('"modele"', "modele")])
    Entraineur(cfg).executer()
    import json
    lignes = [json.loads(l) for l in (tmp_path / "run" / "journal.jsonl").read_text().splitlines()]
    assert len(lignes) == 3
    assert lignes[2]["apprentissage"].get("reinitialisation") and lignes[2]["apprentissage"]["pas"] >= 1
    assert json.loads((tmp_path / "run" / "etat.json").read_text())["derniere_reinit"] == 3


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
        if cfg.dispositif_autojeu == "cuda" and not cfg.autojeu.continu:
            # au moins 3 vagues de parties simultanées par processus (7 par défaut : portable 8 cœurs) ;
            # inutile en auto-jeu continu, où les parties en cours passent à l'itération suivante
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


# ------------------------------------------------------------ mise en place avancée
def partie_draft(seed=4, choix=3):
    g = Game("2J", "draft", seed=seed)
    rng = random.Random(seed)
    for _ in range(choix):
        g.apply(rng.choice(g.legal_actions()))
    return g


def test_encodage_draft():
    from champ_dhonneur.ia.encodage import GLOB_F, KIND_ID, PENDING_ID
    g = partie_draft()
    e = encode_state(g)
    assert e["unit_i"].shape == (8, 2) and e["glob_f"].shape == (GLOB_F,)
    camps = sorted(e["unit_i"][:, 1].tolist())
    assert camps.count(2) == 5 and camps.count(0) + camps.count(1) == 3
    assert e["glob_i"][0] == PENDING_ID["draft"] and e["glob_f"][22] == 1.0
    acts = encode_actions(g, g.legal_actions())
    assert len(acts) == 5 and (acts[:, 0] == KIND_ID["draft"]).all()


def test_encodage_apres_draft_identique_aux_armees_imposees():
    g = Game("2J", "draft", seed=8)
    rng = random.Random(0)
    while g.in_draft:
        g.apply(rng.choice(g.legal_actions()))
    h = Game("2J", [p.units for p in g.players], seed=8, first=g.first_player)
    a, b = encode_state(g), encode_state(h)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_autojeu_draft_et_cibles_auxiliaires():
    P = ParamsAutoJeu(parties=3, simultanees=3, simulations=8, simulations_rapides=4, max_manches=15,
                      p_draft=1.0, simulations_draft=8, releves=2)
    data, st = jouer_parties(EvaluateurUniforme(), P, seed=5)
    assert st["parties_draft"] == 3
    n = len(data["z"])
    assert data["lieux"].shape == (n, 37) and data["marge"].shape == (n,) and data["main_adv"].shape == (n, 18)
    assert ((data["glob_i"][:, 0] == 7).sum()) >= 3 * 7   # 7 vrais choix par draft
    from champ_dhonneur.notation import import_record
    for t in st["releves"]:
        assert "[Draft " in t and import_record(t).done


def test_reseau_draft_et_tetes_auxiliaires():
    torch = pytest.importorskip("torch")
    from champ_dhonneur.ia.encodage import collate
    from champ_dhonneur.ia.modele import ConfigModele, ReseauChamp
    g = partie_draft()
    h = partie_avancee()
    x, _ = collate([encode_state(g), encode_state(h)], [encode_actions(g, g.legal_actions()),
                                                        encode_actions(h, h.legal_actions())])
    m = ReseauChamp(ConfigModele(d=32, couches=1, tetes=2))
    logits, v, aux = m({k: torch.from_numpy(v) for k, v in x.items()}, aux=True)
    assert logits.shape[0] == 2 and v.shape == (2, 3)
    assert aux["lieux"].shape == (2, 37, 3) and aux["marge"].shape == (2, 9) and aux["main"].shape == (2, 18, 4)


def test_modele_v1_refuse(tmp_path):
    torch = pytest.importorskip("torch")
    from champ_dhonneur.ia.modele import ConfigModele, ModeleIncompatible, ReseauChamp, charger, sauver
    m = ReseauChamp(ConfigModele(d=32, couches=1, tetes=2))
    sauver(tmp_path / "v2.pt", m)
    charger(tmp_path / "v2.pt")
    ck = torch.load(tmp_path / "v2.pt", weights_only=False)
    ck["version_encodage"] = 1
    torch.save(ck, tmp_path / "v1.pt")
    with pytest.raises(ModeleIncompatible):
        charger(tmp_path / "v1.pt")


def test_valeurs_unites_modele_synthetique():
    """Le modèle d'armée retrouve des effets propres connus et le minimax suit la force brute."""
    from itertools import permutations

    from champ_dhonneur.ia import valeurs as V
    pools = V.tirages(40, graine=3)
    vrai = np.linspace(-0.4, 0.4, 16)
    L = np.array([[sum(vrai[V.IDX[p[i]]] for i in s) - sum(vrai[V.IDX[u]] for i, u in enumerate(p) if i not in s)
                   for s in V.REPARTITIONS] for p in pools])
    fit = V.ajuster(pools, L, ridge=1.0)   # la pénalité sur s et c lève la confusion avec a
    assert np.allclose(fit["a"], vrai - vrai.mean(), atol=1e-3) and fit["r2"] > 0.999
    rng = np.random.default_rng(0)
    lt = rng.normal(size=70)
    d, _ = V.minimax_draft(lt)
    feuille = {sum(1 << i for i in s): lt[r] for r, s in enumerate(V.REPARTITIONS)}
    ordre = (0, 1, 1, 0, 0, 1, 1, 0)

    def force_brute(ma, mb, e):
        if e == 8:
            return feuille[ma]
        vs = [force_brute(ma | 1 << i, mb, e + 1) if ordre[e] == 0 else force_brute(ma, mb | 1 << i, e + 1)
              for i in range(8) if not (ma | mb) >> i & 1]
        return max(vs) if ordre[e] == 0 else min(vs)
    assert abs(force_brute(0, 0, 0) - d) < 1e-12
    assert V.calibrer_k([]) is None
