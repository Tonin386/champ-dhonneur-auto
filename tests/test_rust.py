"""Tests différentiels du moteur Rust (module champ_rs) contre le moteur Python.

Mêmes graines → mêmes parties : à chaque décision, l'état complet, la liste ordonnée des
actions légales et l'encodage pour le réseau doivent être identiques. Ignorés si le module
n'est pas compilé (voir rust/ et le Dockerfile).
"""
import random

import numpy as np
import pytest

champ_rs = pytest.importorskip("champ_rs")

from champ_dhonneur.engine import Action, Game  # noqa: E402
from champ_dhonneur.ia import rs  # noqa: E402
from champ_dhonneur.ia.encodage import COIN_ID, PENDING_ID, encode_actions, encode_state  # noqa: E402
from champ_dhonneur.units import ROYAL  # noqa: E402

CODE = dict(COIN_ID)


def etat_python(g: Game) -> dict:
    code = lambda c: 0 if c is None else CODE[c]  # noqa: E731
    return {
        "manche": g.round, "courant": g.current, "au_trait": g.to_move, "fini": g.done,
        "gagnant": -1 if g.winner is None else g.winner, "marqueurs": list(g.markers_left),
        "premier": g.first_player, "initiative": g.initiative, "init_bougee": g.initiative_moved,
        "premier_manche": g.round_first, "en_draft": g.in_draft,
        "dispo": "".join(g.draft.available) if g.in_draft else "",
        "unites": ["".join(p.units) for p in g.players],
        "controle": [(l, -1 if g.control[l] is None else g.control[l]) for l in g.spec.locations],
        "plateau": [(pos, u.owner, u.utype, u.coins) for pos, u in g.board.items()],
        "joueurs": [("".join(p.bag), "".join(p.hand), "".join(p.disc_up), "".join(p.disc_down),
                     [(u, p.reserve.get(u, 0)) for u in p.units], sorted(p.lost.items()))
                    for p in g.players],
        "attentes": [(PENDING_ID[pd.kind], pd.player, pd.pos, code(pd.coin), pd.drawn, list(pd.positions))
                     for pd in g.pending],
    }


def etat_rust(j) -> dict:
    e = j.etat()
    e["joueurs"] = [(sac, main, du, dd, list(res), sorted(perdues))
                    for sac, main, du, dd, res, perdues in e["joueurs"]]
    e["plateau"] = [(pos, prop, g, n) for pos, prop, g, n in e["plateau"]]
    e["attentes"] = [tuple(a[:5]) + (list(a[5]),) for a in e["attentes"]]
    return e


def comparer_encodage(g: Game, j) -> None:
    ci, cf, ui, uf, gi, gf, acts = j.encoder()
    s = encode_state(g)
    assert np.array_equal(np.array(ci).reshape(37, 4), s["cell_i"])
    assert np.allclose(np.array(cf).reshape(37, 3), s["cell_f"], atol=1e-6)
    assert np.array_equal(np.array(ui).reshape(8, 2), s["unit_i"])
    assert np.allclose(np.array(uf).reshape(8, 12), s["unit_f"], atol=1e-6)
    assert np.array_equal(np.array(gi), s["glob_i"])
    assert np.allclose(np.array(gf), s["glob_f"], atol=1e-6)
    assert np.array_equal(np.array(acts).reshape(-1, 7), encode_actions(g, g.legal_actions()))


def rejouer(seed: int, unites=None, premier=None, max_manches: int = 150, encoder_tous: int = 7,
            strategie: str = "aleatoire", pool=None, choisit=None) -> Game:
    if unites == "draft":
        g = Game("2J", "draft", seed=seed, max_rounds=max_manches, pool=pool, draft_first=choisit)
        j = champ_rs.Jeu(seed, None, None, max_manches, draft=True,
                         cartes=None if pool is None else "".join(pool), choisit=choisit)
    else:
        g = Game("2J", unites, seed=seed, first=premier, max_rounds=max_manches)
        j = champ_rs.Jeu(seed, None if unites is None else ["".join(u) for u in unites], premier, max_manches)
    rng = random.Random(seed)
    k = 0
    while True:
        assert etat_rust(j) == etat_python(g), f"graine {seed}, décision {k}"
        legal = g.legal_actions()
        assert [rs.action_python(a) for a in j.legales()] == legal, f"graine {seed}, décision {k}"
        if (k % encoder_tous == 0 or g.in_draft) and not g.done:
            comparer_encodage(g, j)
        if g.done:
            assert j.fini and j.resultat == g.result_label()
            return g
        if strategie == "agressive":   # privilégie attaques et tactiques : plus d'effets en chaîne
            fortes = [a for a in legal if a.kind in ("attack", "tactic", "control")]
            # et recrute peu, pour garder des réserves (défense de la Garde royale)
            autres = [a for a in legal if a.kind != "recruit"] or legal
            a = rng.choice(fortes) if fortes and rng.random() < 0.8 else rng.choice(autres)
        else:
            a = rng.choice(legal)
        g.apply(a)
        j.jouer(*rs.action_rust(a))
        k += 1


def test_generateur_compatible_python():
    # la mise en place (tirage des armées, premier joueur) et les pioches suivent random.Random
    for seed in (0, 1, 2**31 - 1, 123456789, 2**40 + 7):
        g = Game("2J", seed=seed)
        e = etat_rust(champ_rs.Jeu(seed))
        assert e == etat_python(g)


@pytest.mark.parametrize("strategie", ["aleatoire", "agressive"])
def test_parties_identiques(strategie):
    for seed in range(60):
        rejouer(seed * 7919 + 3, strategie=strategie)


def test_mises_en_place_imposees():
    rejouer(5, [list("SPXH"), list("ACLE")], premier=0)
    rejouer(6, [list("NHPK"), list("CFMG")], premier=1)
    rejouer(7, [list("ADNG"), list("CXLE")])
    rejouer(8, [list("BFRM"), list("DGKS")], max_manches=40)


def test_toutes_les_unites_rencontrees():
    vues = set()
    for seed in range(200, 360):
        g = rejouer(seed, strategie="agressive", encoder_tous=25)
        vues |= {a.unit for _, _, a in g.log if a.unit}
        vues |= {a.kind for _, _, a in g.log}
    assert {"B", "F", "G", "K", "D", "L", "C", "A", "X", "H", "R", "S", "M", "N", "P", "E"} <= vues
    assert {"rg_reserve", "rg_unit", "skip"} <= vues


def test_determinisation_conserve_l_information_publique():
    g = Game("2J", seed=11)
    j = champ_rs.Jeu(11)
    rng = random.Random(0)
    for _ in range(40):
        a = rng.choice(g.legal_actions())
        g.apply(a)
        j.jouer(*rs.action_rust(a))
    obs = j.etat()
    moi = obs["au_trait"]
    d = j.determiniser(moi, 99).etat()
    for i in range(2):
        sac, main, du, dd, res, perdues = obs["joueurs"][i]
        sac2, main2, du2, dd2, res2, perdues2 = d["joueurs"][i]
        assert (du, res, perdues) == (du2, res2, perdues2)
        assert len(main) == len(main2) and len(dd) == len(dd2) and len(sac) == len(sac2)
        assert sorted(sac + main + dd) == sorted(sac2 + main2 + dd2)
        if i == moi:
            assert (sac, main, dd) == (sac2, main2, dd2)
    assert d["plateau"] == obs["plateau"]
    assert ROYAL in "".join("".join(z[:4]) for z in d["joueurs"])


def test_autojeu_rust_produit_des_exemples_valides():
    from champ_dhonneur.ia.autojeu import ParamsAutoJeu
    from champ_dhonneur.ia.evaluateurs import EvaluateurUniforme
    from champ_dhonneur.notation import import_record, parse_headers
    P = ParamsAutoJeu(parties=6, simultanees=3, simulations=16, simulations_rapides=4,
                      max_manches=20, releves=3)
    data, st = rs.jouer_parties_rs(rs.EvaluateurLotUniforme(), P, seed=3)
    assert st["parties"] == 6 and st["evaluations"] > 0
    n = len(data["z"])
    assert n > 0 and data["acts"].shape == (n, 64, 7) and data["cell_i"].dtype == np.int8
    assert np.allclose(data["pi"].astype(np.float32).sum(1), 1, atol=1e-2)
    assert set(np.unique(data["z"])) <= {-1.0, 0.0, 1.0}
    assert len(st["releves"]) == 3
    for t in st["releves"]:            # les relevés du moteur Rust se rejouent dans le moteur Python
        g = import_record(t)
        assert g.done and g.result_label() == parse_headers(t)["Resultat"]
    assert EvaluateurUniforme  # l'évaluateur Python reste disponible pour le chemin Python


def test_autojeu_rust_meilleur_coup():
    """Coup joué sans bruit, recherches rapides sans bruit, coup gagnant toujours joué."""
    from champ_dhonneur.ia.autojeu import ParamsAutoJeu
    P = ParamsAutoJeu(parties=6, simultanees=3, simulations=16, simulations_rapides=4,
                      max_manches=20, meilleur_coup=True, p_draft=0.5)
    data, st = rs.jouer_parties_rs(rs.EvaluateurLotUniforme(), P, seed=3)
    assert st["parties"] == 6 and len(data["z"]) > 0
    assert np.allclose(data["pi"].astype(np.float32).sum(1), 1, atol=1e-2)


def test_direct_moteur_rust(tmp_path):
    """Le fichier de direct écrit par l'auto-jeu Rust se rejoue dans le moteur Python."""
    import json

    from champ_dhonneur.ia.autojeu import ParamsAutoJeu
    chemin = tmp_path / "t00.json"
    P = ParamsAutoJeu(parties=2, simultanees=2, simulations=8, simulations_rapides=4,
                      max_manches=12, direct=str(chemin))
    rs.jouer_parties_rs(rs.EvaluateurLotUniforme(), P, seed=5)
    d = json.loads(chemin.read_text())
    assert d["total"] == 2 and d["actions"]
    g = Game("2J", d["unites"], seed=d["graine"], first=d["initiative"], max_rounds=d["max_manches"])
    for kind, coin, unit, cells, extra in d["actions"]:
        g.apply(Action(kind, coin, unit, tuple(cells), extra))
    assert g.done == d["fini"]


# ------------------------------------------------------------ mise en place avancée
def test_types_d_action_identiques():
    from champ_dhonneur.ia.encodage import KINDS
    assert rs.KINDS == KINDS


def test_draft_generateur_compatible_python():
    for seed in (0, 1, 2**31 - 1, 123456789):
        g = Game("2J", "draft", seed=seed)
        assert etat_rust(champ_rs.Jeu(seed, draft=True)) == etat_python(g)


@pytest.mark.parametrize("strategie", ["aleatoire", "agressive"])
def test_draft_parties_identiques(strategie):
    for seed in range(40):
        g = rejouer(seed * 104729 + 11, "draft", strategie=strategie)
        assert sum(1 for rnd, _, _ in g.log if rnd == 0) == 8


def test_draft_impose():
    rejouer(3, "draft", pool=list("ACGHKLNX"), choisit=1)
    rejouer(4, "draft", pool=list("BDEFMPRS"), choisit=0, max_manches=30)


def test_draft_encodage_invariant():
    """Après le draft, l'observation est celle de la même partie à armées imposées."""
    for seed in range(10):
        j = champ_rs.Jeu(seed, draft=True)
        rng = random.Random(seed)
        while j.etat()["en_draft"]:
            j.jouer(*rng.choice(j.legales()))
        e = j.etat()
        k = champ_rs.Jeu(seed, e["unites"], e["premier"])
        assert j.encoder() == k.encoder()


@pytest.mark.parametrize("draft", [False, True])
def test_match_rust_apparie(draft):
    """Chaque graine est jouée deux fois (agents échangés), reproductible, relevés rejouables par
    le moteur Python ; les agents peuvent avoir des recherches différentes ; un même évaluateur
    pour deux agents (lot fusionné) ne change rien ; arrêt anticipé possible."""
    from champ_dhonneur.notation import import_record, parse_headers
    ev = rs.EvaluateurLotUniforme()
    agents = [{"simulations": 8, "m": 8}, {"simulations": 4, "m": 4, "cle_publique": False}]
    parties = [p for g in (11, 12, 13) for p in ((g, 0, 1), (g, 1, 0))]
    r1 = rs.jouer_match_rs([ev, ev], agents, parties, draft=draft, max_manches=25, releves=2,
                           simultanees=3)
    r2 = rs.jouer_match_rs([ev, rs.EvaluateurLotUniforme()], agents, parties, draft=draft,
                           max_manches=25, simultanees=6)
    assert len(r1["parties"]) == 6
    assert sorted(r1["parties"]) == sorted(r2["parties"])
    assert sorted((g, a, b) for g, a, b, *_ in r1["parties"]) == sorted(parties)
    assert all(s in (-1, 0, 1) for *_, s, _ in r1["parties"])
    assert all(e > 0 for e in r1["evaluations"])
    assert len(r1["releves"]) == 2
    for t in r1["releves"]:
        g = import_record(t)
        assert g.done and g.result_label() == parse_headers(t)["Resultat"]
    r3 = rs.jouer_match_rs([ev, ev], agents, parties, draft=draft, max_manches=25, simultanees=2,
                           arret=lambda finies: len(finies) >= 2)
    assert 2 <= len(r3["parties"]) < 6


def test_autojeu_rust_continu():
    """Auto-jeu continu : chaque appel renvoie au moins `parties` parties terminées ; les parties
    en cours continuent à l'appel suivant (même auto-jeu, lots toujours pleins)."""
    from champ_dhonneur.ia.autojeu import ParamsAutoJeu
    P = ParamsAutoJeu(parties=4, simultanees=6, simulations=8, simulations_rapides=4, max_manches=20,
                      meilleur_coup=True, continu=True)
    rs._CONTINU.clear()
    d1, s1 = rs.jouer_parties_rs(rs.EvaluateurLotUniforme(), P, seed=5)
    aj = next(iter(rs._CONTINU.values()))[0]
    d2, s2 = rs.jouer_parties_rs(rs.EvaluateurLotUniforme(), P, seed=6)
    assert next(iter(rs._CONTINU.values()))[0] is aj
    assert s1["parties"] >= 4 and s2["parties"] >= 4
    assert len(d1["z"]) > 0 and len(d2["z"]) > 0
    assert np.allclose(d2["pi"].astype(np.float32).sum(1), 1, atol=1e-2)
    rs._CONTINU.clear()


def test_tournoi_et_evaluations_rust(tmp_path):
    """Tournoi (réseaux en un seul match Rust, bots en Python), matchs contre plusieurs réseaux
    et test séquentiel, avec de petits réseaux sur CPU."""
    torch = pytest.importorskip("torch")
    from champ_dhonneur.ia.evaluation import matchs_contre, sprt_reseaux, tournoi
    from champ_dhonneur.ia.modele import ConfigModele, ReseauChamp, sauver
    for k in range(2):
        torch.manual_seed(k)
        sauver(tmp_path / f"m{k}.pt", ReseauChamp(ConfigModele(d=16, couches=1, tetes=2)))
    a, b = str(tmp_path / "m0.pt"), str(tmp_path / "m1.pt")
    r = tournoi([a, b, f"{a}:simulations=4", "aleatoire"], paires=2, simulations=8,
                dispositif="cpu", compiler=False)
    assert r["noms"] == ["m0", "m1", "m0@simulations=4", "aleatoire"]
    assert len(r["matchs"]) == 6 and all(m["parties"] == 4 for m in r["matchs"].values())
    assert set(r["elo"]) == set(r["noms"])
    res = matchs_contre(a, [(b, {}, "m1"), (a, {"simulations": 4}, "m0bis")], paires=2, simulations=8,
                        dispositif="cpu", compiler=False, releves=1, nom="m0")
    assert set(res) == {"m1", "m0bis"} and all(x["parties"] == 4 for x in res.values())
    assert all(len(x["releves"]) <= 1 for x in res.values())
    s = sprt_reseaux(a, b, max_paires=3, simulations=4, dispositif="cpu", compiler=False)
    assert s["decision"] in ("H0", "H1", "indécis") and s["paires"] <= 3


@pytest.mark.parametrize("draft", [False, True])
def test_etat_python_vers_rust(draft):
    """Une partie du moteur Python reconstruite dans le moteur Rust (Jeu.depuis_etat) : même état,
    même encodage, mêmes actions légales dans le même ordre, à tout moment de la partie."""
    rng = random.Random(3)
    vues = 0
    for graine in range(25):
        g = Game("2J", "draft" if draft else None, seed=graine)
        for _ in range(rng.randrange(0, 150)):
            if g.done:
                break
            g.apply(rng.choice(g.legal_actions()))
        j = champ_rs.Jeu.depuis_etat(rs.etat_jeu(g), 150, graine)
        assert etat_rust(j) == etat_python(g)
        if not g.done:
            comparer_encodage(g, j)
            assert [rs.action_python(a) for a in j.legales()] == g.legal_actions()
            vues += 1
    assert vues > 10


def test_recherche_rust_sur_une_position():
    """La recherche Rust sur une position du moteur Python renvoie un coup légal et un résultat
    au format de la recherche Python ; un coup gagnant est trouvé."""
    from champ_dhonneur.ia.recherche import Resultat
    ev = rs.EvaluateurLotUniforme()
    rng = random.Random(1)
    for graine in range(6):
        g = Game("2J", "draft" if graine % 2 else None, seed=graine)
        for _ in range(rng.randrange(0, 60)):
            if g.done:
                break
            g.apply(rng.choice(g.legal_actions()))
        if g.done or len(g.legal_actions()) < 2:
            continue
        r = rs.rechercher(g, ev, simulations=32, agent={"parallele": 4}, graine=graine)
        assert isinstance(r, Resultat) and r.action in g.legal_actions()
        assert r.legal == g.legal_actions() and abs(float(r.politique.sum()) - 1) < 1e-3
        assert r.simulations == 32


def test_draft_exact_autojeu_et_match():
    """Choix de carte exact (draft_exact) : exemples de draft valides en auto-jeu, et un agent qui
    l'utilise peut affronter un agent à recherche Gumbel."""
    from champ_dhonneur.ia.autojeu import ParamsAutoJeu
    P = ParamsAutoJeu(parties=4, simultanees=4, simulations=8, simulations_rapides=4, max_manches=20,
                      p_draft=1.0, draft_exact=2, meilleur_coup=True)
    data, st = rs.jouer_parties_rs(rs.EvaluateurLotUniforme(), P, seed=2)
    assert st["parties_draft"] == 4
    d = data["glob_i"][:, 0] == PENDING_ID["draft"]
    assert d.sum() >= 4 * 6
    assert np.allclose(data["pi"][d].astype(np.float32).sum(1), 1, atol=1e-2)
    ev = rs.EvaluateurLotUniforme()
    r = rs.jouer_match_rs([ev, ev], [{"simulations": 8, "draft_exact": 2}, {"simulations": 8}],
                          [(5, 0, 1), (5, 1, 0)], draft=True, max_manches=20)
    assert len(r["parties"]) == 2


def test_recherche_rust_progressive():
    """Recherche progressive Rust : passes successives sur le même arbre (simulations cumulées),
    lignes principales exportées ; au temps, au moins une passe."""
    ev = rs.EvaluateurLotUniforme()
    g = Game(seed=4)
    rng = random.Random(4)
    for _ in range(40):
        g.apply(rng.choice(g.legal_actions()))
    while g.pending or len(g.legal_actions()) < 3:
        g.apply(g.legal_actions()[0])
    r = rs.recherche_jeu(g, 64, {"parallele": 4}, 1)
    rs._conduire(r, ev, r.etape())
    assert r.resultat()[-1] == 64
    rs._conduire(r, ev, r.prolonger(128, 8))
    assert r.resultat()[-1] == 64 + 128
    lignes = r.lignes(6)
    assert len(lignes) == len(g.legal_actions()) and any(lignes)
    res = rs.rechercher_temps(g, ev, 0.5, {"parallele": 4})
    assert res.action in g.legal_actions() and res.infos["passes"] >= 1


def test_analyse_avec_la_recherche_rust(tmp_path):
    """Analyse progressive avec la recherche Rust : profondeur atteinte, lignes de jeu jouables,
    suivi des passes intermédiaires, arrêt sur demande."""
    pytest.importorskip("torch")
    import threading

    from champ_dhonneur.ia.analyse import analyser, bot_analyse
    from champ_dhonneur.ia.modele import ConfigModele, ReseauChamp, sauver
    from champ_dhonneur.ia.recherche import Limite
    sauver(tmp_path / "m.pt", ReseauChamp(ConfigModele(d=16, couches=1, tetes=2)))
    bot = bot_analyse(64, str(tmp_path / "m.pt"), "cpu")
    assert bot.rust
    g = Game(seed=6)
    rng = random.Random(6)
    for _ in range(30):
        g.apply(rng.choice(g.legal_actions()))
    while g.pending or len(g.legal_actions()) < 3:
        g.apply(g.legal_actions()[0])
    vues = []
    a = analyser(g, bot, limite=Limite(profondeur=3), suivi=vues.append)
    assert a["profondeur"] == 3 and len(vues) == 2 and all(v["en_cours"] for v in vues)
    assert a["coups"] and all(isinstance(c["ligne"], list) and c["ligne"] for c in a["coups"])
    assert a["coups"][0]["action"] in g.legal_actions()
    arret = threading.Event()
    arret.set()
    b = analyser(g, bot, limite=Limite(arret=arret))
    assert b["coups"] and not b["en_cours"]
