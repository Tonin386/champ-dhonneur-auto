"""Historique des parties (page « Jouer ») : enregistrement, relecture à l'identique, variantes."""
import random

import pytest
from fastapi.testclient import TestClient

from champ_dhonneur.bots.neural import NeuralBot
from champ_dhonneur.server import jeu
from champ_dhonneur.server.app import app


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Sans réseau : l'IA joue (sur demande) le premier coup d'une courte recherche heuristique."""
    monkeypatch.setattr(jeu, "PARTIES", tmp_path / "parties")
    monkeypatch.setattr(jeu, "torch_present", lambda: True)
    monkeypatch.setattr(jeu, "modeles", lambda: [])
    bot = NeuralBot(heuristique=True, seed=0)
    bot.k = None
    monkeypatch.setattr(jeu, "_analyste", lambda modele: bot)
    monkeypatch.setattr(jeu, "SIMS_SANS_REFLEXION", 16)
    return TestClient(app)


def jouer(c, e, n, rng):
    """n décisions : l'IA quand elle a le trait (on lui demande de jouer), sinon un coup au hasard
    (pioches : première pièce)."""
    for _ in range(n):
        if e["fini"]:
            break
        if e["tirage"]:
            e = c.post(f"/api/jeu/{e['id']}/tirage", json={"piece": sorted(e["tirage"]["sac"])[0]}).json()
        elif e["ia"]:
            e = c.post(f"/api/jeu/{e['id']}/ia").json()
        else:
            e = c.post(f"/api/jeu/{e['id']}/jouer", json={"index": rng.choice(e["legal"])["i"]}).json()
    return e


def relire(c, gid):
    """Oublie la partie en mémoire (serveur redémarré) et la relit depuis son fichier."""
    avant = c.get(f"/api/jeu/{gid}").json()
    jeu.SESSIONS.pop(gid)
    apres = c.get(f"/api/jeu/{gid}").json()
    for k in ("images", "legal", "trait", "fini", "resultat", "tirage", "possibles", "mise", "joueurs"):
        assert apres[k] == avant[k], k
    return apres


H2 = [{"type": "humain"}, {"type": "humain"}]


@pytest.mark.parametrize("config", [
    {"mode": "libre", "joueurs": H2},
    {"mode": "draft", "joueurs": [{"type": "humain"}, {"type": "ia"}]},
    {"mode": "draft", "cartes": list("ABCDEFGH"), "premier": 1, "joueurs": [{"type": "ia"}, {"type": "ia"}]},
    {"mode": "libre", "armees": [list("RSXH"), list("ACLE")], "hybride": True,
     "joueurs": [{"type": "humain"}, {"type": "ia"}]},
])
def test_partie_relue_a_l_identique(client, config):
    rng = random.Random(3)
    e = client.post("/api/jeu", json=config).json()
    gid = e["id"]
    assert client.get("/api/jeu/parties").json()["parties"] == []     # aucune décision : rien d'enregistré
    e = jouer(client, e, 40, rng)
    relire(client, gid)
    e = jouer(client, client.get(f"/api/jeu/{gid}").json(), 25, rng)   # la partie relue se poursuit
    relire(client, gid)
    liste = client.get("/api/jeu/parties").json()
    assert liste["ecriture"] and [p["id"] for p in liste["parties"]] == [gid]
    p = liste["parties"][0]
    assert p["decisions"] == len(jeu.SESSIONS[gid].actions) and p["fini"] == e["fini"]


def test_pioche_en_attente_conservee(client):
    """Partie hybride relue pendant une pioche de l'IA : le coup en attente et les pièces déjà
    saisies sont conservés."""
    rng = random.Random(1)
    e = client.post("/api/jeu", json={"mode": "libre", "armees": [list("SPXH"), list("ACLE")], "premier": 0,
                                      "hybride": True, "joueurs": [{"type": "humain"}, {"type": "ia"}]}).json()
    gid = e["id"]
    for _ in range(200):
        e = jouer(client, e, 1, rng)
        if e["tirage"] and e["tirage"]["contexte"] != "initial" and len(e["tirage"]["sac"]) > 1:
            break
    e = client.post(f"/api/jeu/{gid}/tirage", json={"piece": sorted(e["tirage"]["sac"])[0]}).json()
    assert e["tirage"] and e["tirage"]["choisies"]
    e2 = relire(client, gid)
    assert e2["tirage"]["choisies"] == e["tirage"]["choisies"]


def test_position_editee_et_variante(client):
    p = {"unites": [["S", "P", "X", "H"], ["A", "C", "L", "E"]], "plateau": [],
         "controle": {"b1": 0, "e1": 0, "c6": 1, "f5": 1}, "trait": 0, "initiative": 0, "manche": 5,
         "joueurs": [{"main": ["S", "P", "X"], "sac": ["*"]}, {"main": ["A", "C"], "sac": ["*"]}]}
    e = client.post("/api/jeu", json={"joueurs": H2, "position": p}).json()
    gid = e["id"]
    e = jouer(client, e, 6, random.Random(2))
    relire(client, gid)
    n = len(jeu.SESSIONS[gid].actions)
    v = client.post(f"/api/jeu/{gid}/revenir", json={"pos": 2, "copie": True}).json()
    assert v["id"] != gid and v["n"] == 3 and v["mise"]["libelle"].endswith("· variante")
    assert len(jeu.SESSIONS[gid].actions) == n                   # l'originale est intacte
    ids = {q["id"] for q in client.get("/api/jeu/parties").json()["parties"]}
    assert ids == {gid, v["id"]}
    assert client.delete(f"/api/jeu/parties/{v['id']}").json() == {"ok": True}
    assert client.get(f"/api/jeu/{v['id']}").status_code == 404
    assert client.delete("/api/jeu/parties/../../etc").status_code == 404


def test_ia_disparue_partie_consultable(client, monkeypatch):
    e = client.post("/api/jeu", json={"mode": "libre", "joueurs": [{"type": "humain"}, {"type": "ia"}]}).json()
    gid = e["id"]
    e = jouer(client, e, 6, random.Random(4))
    jeu.SESSIONS.pop(gid)
    monkeypatch.setattr(jeu, "torch_present", lambda: False)
    e = client.get(f"/api/jeu/{gid}").json()
    assert e["n"] == 7
    if e["ia"]:
        assert client.post(f"/api/jeu/{gid}/ia").status_code == 400


def test_analyse_indisponible_toujours_avec_coups(client):
    """Régression : une analyse sans réseau au trait de l'IA (sa réflexion) répond avec « coups »,
    sinon la page plante (« Cannot read properties of undefined (reading 'length') »)."""
    e = client.post("/api/jeu", json={"mode": "libre", "premier": 1,
                                      "joueurs": [{"type": "humain"}, {"type": "ia"}]}).json()
    assert e["ia"]
    a = client.post(f"/api/jeu/{e['id']}/analyse").json()
    assert a["indisponible"] and a["coups"] == [] and a["observateur"] == 1
