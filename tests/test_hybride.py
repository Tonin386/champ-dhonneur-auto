"""Mode hybride (docs/HYBRIDE.md) : partie sur un vrai plateau, pioches de l'IA saisies,
coups libres du joueur plateau, IA limitée à son information."""
import random

import pytest
from fastapi.testclient import TestClient

from champ_dhonneur.bots import make_bot
from champ_dhonneur.bots.neural import NeuralBot
from champ_dhonneur.engine import FACE_DOWN, Game
from champ_dhonneur.hybride import (CACHEE, TirageRequis, concretiser, coups_libres, essayer,
                                    pieces_possibles)
from champ_dhonneur.notation import action_str
from champ_dhonneur.server import jeu
from champ_dhonneur.server.app import app


def libre(a):
    """Forme « libre » d'un coup réel : la pièce d'un coup face cachée n'est pas connue."""
    return a._replace(coin=CACHEE) if a.coin is not None and a.kind in FACE_DOWN else a


def public(g):
    """Tout ce que la table montre, plus le total des pièces cachées de chaque joueur."""
    return (g.round, g.to_move, g.initiative, g.initiative_moved, tuple(g.markers_left), g.done, g.winner,
            sorted((c, u.owner, u.utype, u.coins) for c, u in g.board.items()),
            sorted((c, t) for c, t in g.control.items() if t is not None),
            [(pd.kind, pd.player, pd.pos) for pd in g.pending],
            [(sorted(p.units), sorted(p.disc_up), len(p.disc_down), len(p.hand), len(p.bag),
              sorted(p.bag + p.hand + p.disc_down + [pd.coin for pd in g.pending
                                                      if pd.kind == "priest" and pd.player == i and pd.coin]),
              sorted(p.reserve.items()), sorted(p.lost.items()))
             for i, p in enumerate(g.players)])


def prive(g, p):
    pl = g.players[p]
    return sorted(pl.hand), sorted(pl.bag), sorted(pl.disc_down)


def partie(seed, n, unites="aleatoire"):
    g = Game("2J", unites, seed=seed)
    bots = [make_bot("glouton", seed=1), make_bot("glouton", seed=2)]
    for _ in range(n):
        if g.done:
            break
        g.apply(bots[g.to_move].choose(g))
    return g


# ------------------------------------------------------------------ coups libres
@pytest.mark.parametrize("seed", [3, 11, 29])
def test_coups_libres_couvrent_toute_main_possible(seed):
    """Quelle que soit la vraie main du joueur caché, son coup est proposé ; et chaque coup libre
    se concrétise en un coup légal sans rien changer à l'information publique."""
    g = Game("2J", seed=seed)
    bots = [make_bot("glouton", seed=1), make_bot("glouton", seed=2)]
    rng = random.Random(seed)
    for _ in range(120):
        if g.done:
            break
        p = g.to_move
        libres = coups_libres(g, p)
        for s in range(3):                        # mains compatibles avec ce que sait l'adversaire
            d = g.determinize(1 - p, random.Random(rng.random()))
            assert {libre(a) for a in d.legal_actions()} <= set(libres)
        assert {libre(a) for a in g.legal_actions()} <= set(libres)
        for f in rng.sample(libres, min(8, len(libres))):
            h = g.copy()
            a = concretiser(h, p, f)
            assert a in h.legal_actions() and public(h) == public(g)
        g.apply(bots[p].choose(g))


def test_piece_impossible_refusee():
    g = partie(5, 40, [list("SPXH"), list("ACLE")])
    while g.pending or g.in_draft:
        g.apply(g.legal_actions()[0])
    p = g.to_move
    absentes = sorted(set(g.players[p].units) - set(pieces_possibles(g, p)))
    for c in absentes:
        assert not any(a.coin == c for a in coups_libres(g, p))


def test_pioche_imposee_de_l_ia():
    g = Game("2J", [list("SPXH"), list("ACLE")], seed=4, first=0)
    # tout le monde passe : la dernière pièce de la manche fait piocher les deux joueurs
    while sum(len(p.hand) for p in g.players) > 1:
        g.apply(next(a for a in g.legal_actions() if a.kind == "pass"))
    dernier = next(a for a in g.legal_actions() if a.kind == "pass")
    with pytest.raises(TirageRequis) as e:
        essayer(g, dernier, 1, [])
    assert e.value.sac == sorted(g.players[1].bag) and (e.value.rang, e.value.total) == (1, 3)
    sac = g.players[1].bag
    choix = [sac[0], sac[1], sac[2]]
    h = essayer(g, dernier, 1, choix)
    assert h.players[1].hand == choix


# ------------------------------------------------------------------ information de l'IA
def test_decision_de_l_ia_independante_de_la_repartition_cachee():
    """À graine égale, l'IA joue pareil quelle que soit la répartition des pièces cachées adverses."""
    g = partie(17, 24)
    while g.pending or g.in_draft:
        g.apply(g.legal_actions()[0])
    assert not g.done and len(g.legal_actions()) > 1
    ia = g.to_move
    decisions = set()
    for k in range(4):
        h = g.copy()
        pl = h.players[1 - ia]
        pool = pl.bag + pl.hand + pl.disc_down
        random.Random(k).shuffle(pool)
        nh, nd = len(pl.hand), len(pl.disc_down)
        pl.hand, pl.disc_down, pl.bag = pool[:nh], pool[nh:nh + nd], pool[nh + nd:]
        bot = NeuralBot(heuristique=True, simulations=48, seed=0)
        a = bot.choose(h)
        decisions.add((a, tuple(bot.derniere.politique.round(6))))
    assert len(decisions) == 1


# ------------------------------------------------------------------ partie hybride complète
class BotFactice:
    """Remplace le réseau (absent des tests) : même interface que NeuralBot."""

    def __init__(self, modele=None, simulations=200, seed=None, **kw):
        self.b = make_bot("glouton", seed=seed)

    def choose(self, g):
        return self.b.choose(g)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(jeu, "torch_present", lambda: True)
    monkeypatch.setattr(jeu, "modeles", lambda: [])   # analyse sans réseau, quels que soient les runs présents
    import champ_dhonneur.bots.neural as neural
    monkeypatch.setattr(neural, "NeuralBot", BotFactice)
    return TestClient(app)


def suite(e):
    """Paramètres pour ne recevoir que les nouvelles images."""
    return f"depuis={e['n']}&version={e['version']}"


def saisir_pioches(c, gid, e, file, vus):
    """Saisit, pièce par pièce, les pioches réelles de l'IA ; vérifie les saisies automatiques."""
    while e["tirage"]:
        t = e["tirage"]
        vus.add((t["contexte"], t["melange"]))
        k = len(t["choisies"])
        assert t["choisies"] == file[:k], (t, file)
        assert file[k] in t["sac"] and len(t["sac"]) > 1
        e = c.post(f"/api/jeu/{gid}/tirage?{suite(e)}", json={"piece": file[k]}).json()
    return e


def jouer_sur_table(c, config, verite, ia, seed, max_decisions=400, plateau="glouton"):
    """La table (`verite`, avec ses vraies pioches) et la session hybride avancent ensemble."""
    tirees = []
    reel = Game._draw

    def enregistrer(p):
        x = reel(verite, p)
        if p == ia and x is not None:
            tirees.append(x)
        return x

    joueurs = [{"type": "ia"}, {"type": "humain"}] if ia == 0 else [{"type": "humain"}, {"type": "ia"}]
    e = c.post("/api/jeu", json=dict(config, joueurs=joueurs, hybride=True, graine=seed + 1000)).json()
    assert e["hybride"] == {"ia": ia, "plateau": 1 - ia}, e
    gid = e["id"]
    s = jeu.SESSIONS[gid]
    vus: set = set()
    e = saisir_pioches(c, gid, e, list(verite.players[ia].hand), vus)   # première main (mode libre)
    verite._draw = enregistrer
    plateau = make_bot(plateau, seed=seed)
    choix_ia = make_bot("glouton", seed=ia)          # l'opérateur, qui choisit les coups de l'IA
    for _ in range(max_decisions):
        assert public(s.game) == public(verite) and prive(s.game, ia) == prive(verite, ia)
        assert all(x == "?" for i in e["images"] for x in i["j"][1 - ia]["h"])   # main fictive masquée
        if verite.done:
            break
        tirees.clear()
        if verite.to_move == ia:
            # l'IA ne joue pas seule : son coup est choisi parmi ses coups réels (sa main est connue)
            assert e["humain"] and not e["ia"] and e["possibles"] is None
            coups = s.game.legal_actions()
            assert [l["kind"] for l in e["legal"]] == [x.kind for x in coups]
            a = choix_ia.choose(s.game)
            e = c.post(f"/api/jeu/{gid}/jouer?{suite(e)}", json={"index": coups.index(a)}).json()
            assert (s.attente["action"] if s.attente else s.actions[-1]) == a
            assert a in verite.legal_actions()
            verite.apply(a)
        else:
            a = plateau.choose(verite)
            if s.game.pending and s.game.pending[-1].kind == "priest":
                vus.add(("moine", "plateau"))
            if a.kind in FACE_DOWN:
                vus.add(("cachee", a.kind))
            coups = coups_libres(s.game, s.game.to_move)
            assert libre(a) in coups                       # le vrai coup est toujours saisissable
            e = c.post(f"/api/jeu/{gid}/jouer?{suite(e)}", json={"index": coups.index(libre(a))}).json()
            verite.apply(a)
        e = saisir_pioches(c, gid, e, list(tirees), vus)
    return gid, s, vus


@pytest.mark.parametrize("seed,ia,armees,plateau,attendus", [
    (1, 1, [list("RSXH"), list("ACLE")], "glouton", [("moine", "plateau")]),     # Moine du joueur plateau
    (2, 0, [list("RSXH"), list("ACLE")], "glouton", [("moine", False)]),          # Moine soldat de l'IA
    (3, 1, [list("ABRM"), list("GKDN")], "aleatoire",                            # joueur plateau au hasard
     [("cachee", "pass"), ("cachee", "initiative"), ("cachee", "recruit")]),
    (4, 0, [list("PFHE"), list("RCGX")], "aleatoire", [("cachee", "initiative"), ("cachee", "pass")]),
])
def test_partie_libre_sur_table(client, seed, ia, armees, plateau, attendus):
    verite = Game("2J", armees, seed=seed, first=seed % 2)
    gid, s, vus = jouer_sur_table(client, {"mode": "libre", "armees": armees, "premier": seed % 2},
                                  verite, ia, seed, plateau=plateau)
    assert ("manche", False) in vus and set(attendus) <= vus, vus
    # l'historique rejoué (pioches et échanges déterministes) redonne exactement la partie
    images = [dict(i) for i in s.film.images]
    client.post(f"/api/jeu/{gid}/revenir", json={"pos": len(s.actions)})
    assert s.film.images == images and public(s.game) == public(verite)
    # retour en arrière puis reprise : même position qu'à ce moment-là
    n = len(s.actions) // 2
    h = s.jeu_a(n)
    client.post(f"/api/jeu/{gid}/revenir", json={"pos": n})
    assert public(s.game) == public(h) and s.film.images == images[:n + 1]


def test_partie_draft_sur_table(client):
    cartes = list("ABCGKMRS")
    verite = Game("2J", "draft", seed=8, pool=cartes, draft_first=0)
    gid, s, vus = jouer_sur_table(client, {"mode": "draft", "cartes": cartes, "premier": 0}, verite, 1, 8)
    assert s.initial is None and len(s.actions) > 8


def test_remise_en_sac_pendant_la_pioche(client):
    """Sur plusieurs parties, la pioche de l'IA traverse une remise de la défausse dans le sac."""
    vus_tous = set()
    for seed in (21, 22, 23):
        armees = [list("RSXH"), list("ACLE")]
        verite = Game("2J", armees, seed=seed, first=0)
        _, _, vus = jouer_sur_table(client, {"mode": "libre", "armees": armees, "premier": 0},
                                    verite, 0, seed, max_decisions=250)
        vus_tous |= vus
    assert ("manche", True) in vus_tous


def test_saisies_refusees_et_annulation(client):
    armees = [list("SPXH"), list("ACLE")]
    e = client.post("/api/jeu", json={"mode": "libre", "armees": armees, "premier": 0, "hybride": True,
                                      "joueurs": [{"type": "humain"}, {"type": "ia"}]}).json()
    gid = e["id"]
    t = e["tirage"]
    assert t["contexte"] == "initial" and t["total"] == 3 and sum(t["sac"].values()) == 9
    assert not e["humain"] and not e["ia"] and e["legal"] == []
    assert client.post(f"/api/jeu/{gid}/tirage", json={"piece": "S"}).status_code == 400   # pas à Argent
    assert client.post(f"/api/jeu/{gid}/jouer", json={"index": 0}).status_code == 400
    for piece in ("A", "A"):
        e = client.post(f"/api/jeu/{gid}/tirage", json={"piece": piece}).json()
    assert e["tirage"]["choisies"] == ["A", "A"]
    assert "A" not in e["tirage"]["sac"]
    e = client.post(f"/api/jeu/{gid}/tirage/annuler").json()
    assert e["tirage"]["choisies"] == []
    for piece in ("A", "C", "*"):
        e = client.post(f"/api/jeu/{gid}/tirage", json={"piece": piece}).json()
    assert e["tirage"] is None and e["images"][0]["j"][1]["h"] == ["*", "A", "C"]
    assert e["images"][0]["j"][0]["h"] == ["?", "?", "?"] and e["humain"]
    # coups libres d'Or : pièce « cachée » pour les coups face cachée
    assert any(l["coin"] == "?" and l["kind"] == "pass" for l in e["legal"])
    assert set(e["possibles"]) == {"S", "P", "X", "H", "*"}
    i = next(l["i"] for l in e["legal"] if l["kind"] == "deploy")
    e = client.post(f"/api/jeu/{gid}/jouer", json={"index": i}).json()
    assert e["n"] == 2 and e["images"][-1]["j"][0]["h"] == ["?", "?"]
    e = client.post(f"/api/jeu/{gid}/annuler").json()
    assert e["n"] == 1 and e["humain"]
    assert client.get(f"/api/jeu/{gid}/releve").status_code == 400
    # l'IA ne change pas de camp en cours de partie
    r = client.post(f"/api/jeu/{gid}/reglages", json={"joueurs": [{"type": "ia"}, {"type": "humain"}]})
    assert r.status_code == 400
    # une partie hybride demande exactement une IA
    assert client.post("/api/jeu", json={"hybride": True, "joueurs": [{"type": "humain"}] * 2}).status_code == 400


def coup_de_l_ia(e):
    """Coup choisi pour l'IA : un coup face visible si possible (déploiement, renfort…)."""
    assert e["trait"] == e["hybride"]["ia"] and e["humain"] and not e["ia"]
    return next((l["i"] for l in e["legal"] if l["kind"] not in FACE_DOWN), e["legal"][0]["i"])


def test_l_ia_ne_joue_pas_seule(client):
    """Hybride : au tour de l'IA, on choisit son coup parmi ses coups réels."""
    armees = [list("SPXH"), list("ACLE")]
    e = client.post("/api/jeu", json={"mode": "libre", "armees": armees, "premier": 1, "hybride": True,
                                      "joueurs": [{"type": "humain"}, {"type": "ia"}]}).json()
    gid = e["id"]
    for piece in ("A", "C", "E"):
        e = client.post(f"/api/jeu/{gid}/tirage", json={"piece": piece}).json()
    assert e["trait"] == 1 and e["humain"] and not e["ia"] and e["possibles"] is None
    assert e["legal"] and all(l["coin"] in ("A", "C", "E") for l in e["legal"])   # sa main, pas « ? »
    assert client.get(f"/api/jeu/{gid}").json()["n"] == 1    # rien ne se joue de soi-même
    assert e["joueurs"][1]["nom"] == "IA conseillère"
    i = coup_de_l_ia(e)
    e = client.post(f"/api/jeu/{gid}/jouer", json={"index": i}).json()
    assert e["n"] == 2 and e["trait"] == 0 and e["possibles"]
    # annuler : revient au choix du coup de l'IA
    e = client.post(f"/api/jeu/{gid}/annuler").json()
    assert e["n"] == 1 and e["trait"] == 1 and e["humain"]


def test_meilleures_lignes_au_tour_de_l_ia(client, monkeypatch):
    """Hybride : au tour de l'IA, sa réflexion (vue par l'IA, avec le modèle choisi pour elle) propose
    ses meilleures lignes, chaque coup avec le camp qui le joue ; « Jouer son meilleur coup » (/ia)
    joue la première."""
    monkeypatch.setattr(jeu, "modeles", lambda: [{"chemin": "heur", "nom": "heuristique"}])
    bot = NeuralBot(heuristique=True, seed=0)
    bot.k = None
    vus = []
    monkeypatch.setattr(jeu, "_analyste", lambda modele: vus.append(modele) or bot)
    armees = [list("SPXH"), list("ACLE")]
    e = client.post("/api/jeu", json={"mode": "libre", "armees": armees, "premier": 1, "hybride": True,
                                      "joueurs": [{"type": "humain"}, {"type": "ia", "modele": "heur"}]}).json()
    gid = e["id"]
    for piece in ("A", "C", "E"):
        e = client.post(f"/api/jeu/{gid}/tirage", json={"piece": piece}).json()
    assert e["joueurs"][1]["nom"] == "IA conseillère — heuristique"
    a = client.post(f"/api/jeu/{gid}/analyse?simulations=64").json()
    assert vus == ["heur"] and a["observateur"] == 1 and a["coups"]
    s = jeu.SESSIONS[gid]
    for c in a["coups"]:
        assert len(c["equipes"]) == len(c["ligne"]) == len(c["ligne_cases"]) == len(c["ligne_desc"])
        assert c["equipes"][0] == s.game.team(1) and set(c["equipes"]) <= {0, 1}
        assert c["ligne_cases"][0] == c["cases"] and c["ligne_desc"][0] == c["description"]
        assert 0 <= c["gain_or"] <= 1
        # pièce d'un coup face cachée du joueur plateau : supposée par la recherche, jamais affichée
        for d, e in zip(c["ligne_desc"], c["equipes"]):
            assert e == s.game.team(1) or "(pièce" not in d or not d.startswith(("Passer", "Prendre", "Recruter")), d
    g = s.game.copy()
    meilleur = g.legal_actions()[a["coups"][0]["i"]]
    assert action_str(g, meilleur) == a["coups"][0]["coup"]
    e = client.post(f"/api/jeu/{gid}/ia").json()
    assert (s.attente["action"] if s.attente else s.actions[-1]) == meilleur


def test_coup_cache_du_joueur_plateau_jamais_devoile(client):
    """Le joueur plateau passe jusqu'à la fin de la manche, qui fait piocher l'IA. Ni le panneau de
    pioche ni le déroulé ne montrent la pièce (fictive) qu'il a défaussée."""
    armees = [list("SPXH"), list("ACLE")]
    e = client.post("/api/jeu", json={"mode": "libre", "armees": armees, "premier": 1, "hybride": True,
                                      "joueurs": [{"type": "humain"}, {"type": "ia"}]}).json()
    gid = e["id"]
    for piece in ("A", "C", "E"):
        e = client.post(f"/api/jeu/{gid}/tirage", json={"piece": piece}).json()
    while not e["tirage"]:
        if e["trait"] == 1:
            i = coup_de_l_ia(e)
        else:
            i = next(l["i"] for l in e["legal"] if l["kind"] == "pass")
        e = client.post(f"/api/jeu/{gid}/jouer", json={"index": i}).json()
    coup = e["tirage"]["coup"]
    assert e["tirage"]["contexte"] == "manche" and coup["j"] == 0    # Argent a commencé : Or finit la manche
    assert coup["d"] == "Passer (pièce cachée)"
    for img in e["images"]:
        a = img["a"]
        if a and a["j"] == 0:
            assert a["pc"] is None and "pièce" not in a["d"], a


def test_apercu_du_coup_de_l_ia_avant_sa_pioche(client):
    """Le coup de l'IA qui déclenche sa pioche est montré (aperçu) avant la saisie des pièces."""
    armees = [list("SPXH"), list("ACLE")]
    e = client.post("/api/jeu", json={"mode": "libre", "armees": armees, "premier": 0, "hybride": True,
                                      "joueurs": [{"type": "humain"}, {"type": "ia"}]}).json()
    gid = e["id"]
    for piece in ("A", "C", "E"):
        e = client.post(f"/api/jeu/{gid}/tirage", json={"piece": piece}).json()
    assert e["apercu"] is None
    # Or ouvre et passe : l'IA (Argent) joue la dernière pièce de la manche, puis doit piocher
    while not e["tirage"]:
        if e["trait"] == 1:
            i = coup_de_l_ia(e)
        else:
            i = next(l["i"] for l in e["legal"] if l["kind"] == "pass")
        e = client.post(f"/api/jeu/{gid}/jouer", json={"index": i}).json()
    t, ap = e["tirage"], e["apercu"]
    assert t["coup"]["j"] == 1 and ap is not None
    assert ap["a"]["j"] == 1 and ap["a"]["n"] == t["coup"]["n"]     # le coup de l'IA est joué dans l'aperçu
    assert len(ap["j"][1]["h"]) == 0                                # sa nouvelle main n'est pas encore piochée
    assert all(x == "?" for x in ap["j"][0]["h"])                  # main du joueur plateau masquée
    n = e["n"]
    for _ in range(3):
        if not e["tirage"]:
            break
        e = client.post(f"/api/jeu/{gid}/tirage", json={"piece": sorted(e["tirage"]["sac"])[0]}).json()
    assert e["apercu"] is None and e["n"] == n + 1


def test_analyse_au_trait_du_joueur_plateau(client):
    """Hybride : l'évaluation reste disponible au tour du joueur plateau, vue par l'IA."""
    e = client.post("/api/jeu", json={"mode": "libre", "premier": 0, "hybride": True,
                                      "joueurs": [{"type": "humain"}, {"type": "ia"}]}).json()
    gid = e["id"]
    while e["tirage"]:
        e = client.post(f"/api/jeu/{gid}/tirage", json={"piece": sorted(e["tirage"]["sac"])[0]}).json()
    assert e["humain"]
    a = client.post(f"/api/jeu/{gid}/analyse").json()
    assert "indisponible" not in a or "Réseau" in a["indisponible"]
    assert a["observateur"] == 1 and "score" in a and a["coups"] == []
