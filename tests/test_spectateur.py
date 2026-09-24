"""Spectateur web : images des parties, diffusion en direct, API."""
import json
import random

import pytest
from fastapi.testclient import TestClient

from champ_dhonneur.bots import make_bot
from champ_dhonneur.engine import MOVE, Game
from champ_dhonneur.ia.direct import Diffuseur, ecrire_phase, preparer_tables
from champ_dhonneur.notation import export_record
from champ_dhonneur.server import spectateur
from champ_dhonneur.server.app import app


def partie(graine=3, max_manches=40):
    g = Game("2J", seed=graine)
    g.max_rounds = max_manches
    bots = [make_bot("glouton", seed=1), make_bot("glouton", seed=2)]
    while not g.done:
        g.apply(bots[g.team(g.to_move)].choose(g))
    return g


def test_film_une_image_par_decision_et_fin():
    g = partie()
    f = spectateur.Film.depuis_releve(export_record(g))
    assert len(f.images) == len(g.log) + 1
    assert f.images[0]["a"] is None and f.images[-1]["f"]["r"] == g.result_label()
    fin = f.images[-1]
    assert sorted((c, o, t, n) for _, c, o, t, n in fin["u"]) == \
        sorted((c, u.owner, u.utype, u.coins) for c, u in g.board.items())


def test_identifiants_stables_lors_des_deplacements():
    f = spectateur.Film.depuis_releve(export_record(partie(graine=11)))
    deplacements = 0
    for avant, apres in zip(f.images, f.images[1:]):
        a = apres["a"]
        if a["k"] != MOVE:
            continue
        ids = {c: i for i, c, *_ in avant["u"]}
        ids_apres = {c: i for i, c, *_ in apres["u"]}
        assert ids_apres[a["c"][1]] == ids[a["c"][0]]
        deplacements += 1
    assert deplacements > 0


def test_diffusion_en_direct_rejouee_a_l_identique(tmp_path):
    ecrire_phase(tmp_path, "autojeu", 7, joueur="iter_0006")
    chemin, = preparer_tables(tmp_path, 1)
    d = Diffuseur(chemin, total=1, periode=0)
    g = Game("2J", seed=5)
    g.max_rounds = 40
    bot = make_bot("glouton", seed=0)
    table = spectateur.Table(tmp_path / "direct" / "t00.json")
    rng = random.Random(0)
    while not g.done:
        d.observer([g], 0)
        if rng.random() < 0.3:   # le serveur ne lit pas après chaque coup
            table.mtime = 0
            table.actualiser(None)
        g.apply(bot.choose(g))
    d.observer([], 1)
    table.mtime = 0
    table.actualiser(json.loads((tmp_path / "direct" / "phase.json").read_text()))
    assert table.meta["fini"] and table.meta["parties"] == 1
    assert len(table.film.images) == len(g.log) + 1
    assert table.film.images[-1]["f"]["r"] == g.result_label()


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(spectateur, "RUNS", tmp_path)
    d = tmp_path / "essai"
    (d / "parties").mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"iterations": 0, "modele": {"d": 64, "couches": 2}}))
    (d / "etat.json").write_text(json.dumps({"iteration": 2, "meilleur": "iter_0002"}))
    ligne = {"iteration": 2, "exemples": 10, "fenetre": 20,
             "autojeu": {"parties": 4, "nulles": 1, "manches": 40, "victoires_blanc": 2, "parties_par_heure": 900},
             "apprentissage": {"pas": 3, "perte_politique": 2.1, "perte_valeur": 0.9},
             "validation": {}, "temps": {"autojeu": 5, "apprentissage": 2},
             "evaluation": {"elo": 12.5, "meilleur": "iter_0002", "matchs": {"glouton": {"score": 0.6}}}}
    (d / "journal.jsonl").write_text(json.dumps(ligne) + "\n{tronqué")
    (d / "parties" / "00002-autojeu-1.nch").write_text(
        '[Iteration "2"]\n[Blanc "iter_0001"]\n[Noir "iter_0001"]\n' + export_record(partie(), headers={"Type": "autojeu"}))
    return d


def test_api_tableau_et_partie(run):
    c = TestClient(app)
    assert c.get("/api/entrainements").json()["entrainements"][0]["nom"] == "essai"
    t = c.get("/api/entrainements/essai").json()
    assert t["series"]["iteration"] == [2] and t["courbe"][0]["elo"] == 12.5
    assert t["totaux"] == {"parties": 4, "secondes": 7}
    assert t["parties"][0]["fichier"] == "00002-autojeu-1"
    f = c.get("/api/entrainements/essai/parties/00002-autojeu-1").json()
    assert f["decor"]["entetes"]["Iteration"] == "2" and len(f["images"]) > 10
    assert c.get("/api/entrainements/essai/parties/..%2Fconfig").status_code == 404
    assert c.get("/api/entrainements/inconnu").status_code == 404
