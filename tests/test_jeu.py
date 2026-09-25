"""Page « Jouer » : positions (éditeur), score, victoires forcées, API de partie."""
import pytest
from fastapi.testclient import TestClient

from champ_dhonneur.bots import make_bot
from champ_dhonneur.engine import Game
from champ_dhonneur.notation import action_str
from champ_dhonneur.position import depuis_position, position
from champ_dhonneur.score import (Solveur, appreciation, bastions, depuis_valeur, texte_mat,
                                  texte_score, vers_valeur)
from champ_dhonneur.server.app import app


def pos(**kw):
    p = {"unites": [["S", "P", "X", "H"], ["A", "C", "L", "E"]], "plateau": [],
         "controle": {"b1": 0, "e1": 0, "c6": 1, "f5": 1}, "trait": 0, "initiative": 0, "manche": 5,
         "joueurs": [{"main": ["S"], "sac": ["*"]}, {"main": ["A"], "sac": ["*"]}]}
    p.update(kw)
    return p


# Or contrôle 5 Lieux et a un Soldat sur le Lieu neutre a2, Soldat en main : victoire en 1
AU_BORD = dict(controle={"b1": 0, "e1": 0, "b4": 0, "c3": 0, "e4": 0, "c6": 1, "f5": 1},
               plateau=[{"case": "a2", "joueur": 0, "unite": "S", "pieces": 1}])


@pytest.mark.parametrize("graine", [1, 7, 23])
def test_position_aller_retour(graine):
    g = Game("2J", seed=graine)
    bots = [make_bot("glouton", seed=1), make_bot("glouton", seed=2)]
    for _ in range(40):
        if g.done:
            break
        g.apply(bots[g.to_move].choose(g))
    while g.pending:
        g.apply(g.legal_actions()[0])
    h = depuis_position(position(g))
    assert h.legal_actions() == g.legal_actions()
    assert h.markers_left == g.markers_left
    assert [p.lost for p in h.players] == [p.lost for p in g.players]
    assert position(h) == position(g)


@pytest.mark.parametrize("modif, message", [
    ({"plateau": [{"case": "a2", "joueur": 0, "unite": "A", "pieces": 1}]}, "n'appartient pas"),
    ({"controle": {"a1": 0}}, "n'est pas un Lieu"),
    ({"joueurs": [{"main": ["S"], "sac": []}, {"main": ["A"], "sac": ["*"]}]}, "Sceau royal"),
    ({"joueurs": [{"main": ["S"] * 3, "sac": ["*", "S", "S", "S"]}, {"main": ["A"], "sac": ["*"]}]}, "de trop"),
    ({"trait": 1, "joueurs": [{"main": ["S"], "sac": ["*"]}, {"main": [], "sac": ["*"]}]}, "au trait"),
    ({"unites": [["S", "P", "X", "H"], ["S", "C", "L", "E"]]}, "partager"),
])
def test_position_invalide(modif, message):
    with pytest.raises(ValueError, match=message):
        depuis_position(pos(**modif))


def test_echelle_du_score():
    assert texte_score(12.44) == "+12,4" and texte_score(-3) == "−3,0" and texte_score(0.01) == "0,0"
    assert texte_mat(0, 3) == "#3" and texte_mat(1, 2) == "#-2"
    assert appreciation(1)[0] == "=" and appreciation(-20)[0] == "−+"
    assert abs(depuis_valeur(vers_valeur(10.0)) - 10.0) < 1e-6   # 10 = un bastion d'avance
    assert depuis_valeur(0.5) > 0 > depuis_valeur(-0.5)


def test_victoire_en_un():
    g = depuis_position(pos(**AU_BORD))
    assert bastions(g) == [5, 2]
    r = Solveur().chercher(g)
    assert (r["equipe"], r["coups"], action_str(g, r["action"])) == (0, 1, "Sa2^")


def test_victoire_forcee_malgre_le_trait_adverse():
    # Argent joue d'abord mais n'a rien pour déloger le Soldat : Or gagne quoi qu'il arrive
    g = depuis_position(pos(**AU_BORD, trait=1))
    assert Solveur().chercher(g)["equipe"] == 0
    # vu d'Or, la main d'Argent est inconnue (sac : une Cavalerie, sans unité sur le plateau)
    g = depuis_position(pos(**AU_BORD, trait=1, joueurs=[{"main": ["S"], "sac": ["*"]},
                                                        {"main": ["A"], "sac": ["*", "C"]}]))
    assert Solveur(observateur=0).chercher(g)["coups"] == 1


def test_defense_possible_pas_de_victoire_forcee():
    # une Cavalerie d'Argent en b3 peut attaquer le Soldat avant qu'il ne contrôle a2
    p = pos(**AU_BORD, trait=1, joueurs=[{"main": ["S"], "sac": ["*"]}, {"main": ["C"], "sac": ["*"]}])
    p["plateau"] = p["plateau"] + [{"case": "b3", "joueur": 1, "unite": "C", "pieces": 1}]
    assert Solveur().chercher(depuis_position(p)) is None
    # vu d'Or, Argent pourrait avoir la Cavalerie en main : pas de victoire garantie non plus
    p["joueurs"] = [{"main": ["S"], "sac": ["*"]}, {"main": ["A"], "sac": ["*", "C"]}]
    assert Solveur(observateur=0).chercher(depuis_position(p)) is None


def test_api_partie_humains_et_editeur():
    c = TestClient(app)
    e = c.post("/api/jeu", json={"mode": "libre", "joueurs": [{"type": "humain"}, {"type": "humain"}]}).json()
    gid = e["id"]
    assert e["humain"] and e["legal"] and e["decor"] and e["masques"] == []
    e = c.post(f"/api/jeu/{gid}/jouer?depuis={e['n']}&version={e['version']}", json={"index": 0}).json()
    assert e["debut"] == 1 and len(e["images"]) == 1 and "decor" not in e
    p = c.get(f"/api/jeu/{gid}/position").json()
    e2 = c.post("/api/jeu", json={"joueurs": [{"type": "humain"}] * 2, "position": p}).json()
    assert e2["edite"] and e2["n"] == 1
    assert c.get(f"/api/jeu/{e2['id']}/releve").status_code == 400
    e = c.post(f"/api/jeu/{gid}/annuler").json()
    assert e["n"] == 1 and e["debut"] == 0
    a = c.post(f"/api/jeu/{gid}/analyse").json()
    assert a["bastions"] == [2, 2] and "texte" in a
    edite = c.post("/api/jeu", json={"joueurs": [{"type": "humain"}] * 2, "position": pos(**AU_BORD)}).json()
    a = c.post(f"/api/jeu/{edite['id']}/analyse").json()
    assert a["texte"] == "#1" and a["mat"]["coup"] == "Sa2^"
    assert c.post("/api/jeu", json={"position": pos(controle={"a1": 0})}).status_code == 400


def test_api_mises_en_place():
    c = TestClient(app)
    h = {"joueurs": [{"type": "humain"}] * 2}
    # draft par défaut, cartes tirées au hasard
    e = c.post("/api/jeu", json=h).json()
    assert e["mise"]["mode"] == "draft" and all(l["kind"] == "draft" for l in e["legal"])
    # draft aux cartes choisies, Argent choisit en premier
    cartes = list("ABCDEFGH")
    e = c.post("/api/jeu", json=dict(h, cartes=cartes, premier=1)).json()
    assert e["trait"] == 1 and sorted(l["n"] for l in e["legal"]) == cartes
    assert e["mise"]["libelle"] == "Draft · cartes choisies"
    assert c.post("/api/jeu", json=dict(h, cartes=list("ABCDEFGG"))).status_code == 400
    # libre : armées choisies, modèle reconnu
    e = c.post("/api/jeu", json=dict(h, mode="libre", armees=[list("SPXH"), list("ACLE")], premier=1)).json()
    assert e["decor"]["unites"] == [sorted("SPXH"), sorted("ACLE")] and e["trait"] == 1
    assert e["mise"]["libelle"] == "Libre · Première partie"
    assert c.post("/api/jeu", json=dict(h, mode="libre", armees=[list("SPXH"), list("SCLE")])).status_code == 400
    assert c.post("/api/jeu", json=dict(h, mode="libre", armees=[list("SPX"), list("ACLE")])).status_code == 400
    assert c.post("/api/jeu", json=dict(h, mode="autre")).status_code == 400
