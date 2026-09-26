"""Évaluation : matchs entre agents et classement Elo (Bradley-Terry).

* Les matchs sont **appariés** : chaque graine est jouée deux fois en
  inversant les camps, ce qui neutralise l'avantage du tirage des unités et
  de l'initiative et réduit fortement la variance.
* Les agents à recherche neuronale d'un même match jouent leurs parties en
  parallèle et partagent les lots d'évaluation.
* Le classement Elo est ajusté par maximum de vraisemblance (algorithme MM
  de Hunter pour Bradley-Terry) sur l'ensemble des résultats enregistrés,
  avec le bot glouton comme ancre (0 Elo).
* Statistiques **par paire** (pentanomiales, comme Fishtest) : les deux parties d'une
  graine ne sont pas indépendantes, l'incertitude se calcule sur les scores de paires
  (0, ¼, ½, ¾, 1). Intervalle de confiance de l'écart Elo, et test séquentiel (SPRT,
  approximation GSPRT) pour décider qu'un agent est plus fort qu'un autre.
* Les matchs entre réseaux se jouent avec le moteur Rust (`match_reseaux`) : toutes les
  parties en même temps, un lot GPU par réseau.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from ..bots.base import Bot
from ..engine import Game
from ..notation import export_record
from .recherche import ParamsRecherche, RechercheGumbel


@dataclass
class AgentRecherche:
    nom: str
    evaluateur: object
    params: ParamsRecherche

    def nouvelle_recherche(self, seed):
        return RechercheGumbel(self.params, seed=seed)


@dataclass
class AgentBot:
    nom: str
    bot: Bot


def graines_appariees(paires: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(2**31) for _ in range(paires)]


def match(a, b, paires: int = 20, seed: int = 0, max_manches: int = 150,
          simultanees: int = 32, graines: list[int] | None = None, releves: int = 0,
          protocole: str = "aleatoire") -> dict:
    """Joue 2 × `paires` parties entre a et b. Renvoie les statistiques du point de vue de a.

    `releves` : nombre de relevés .nch (premières parties terminées) renvoyés dans res["releves"].
    `protocole` : "aleatoire" (armées tirées au hasard) ou "draft" (mise en place avancée ; les
    deux parties d'une paire ont les mêmes 8 cartes et le même premier siège à choisir, a et b
    échangent leurs camps : l'avantage du premier choix s'annule). Les bots sans recherche
    (glouton, mcts…) choisissent leurs cartes au hasard.
    """
    draft = protocole == "draft"
    if graines is None:
        graines = graines_appariees(paires, seed)
    specs = []
    for s in graines:
        specs += [(s, 0), (s, 1)]      # (graine, camp de a)
    res = {"victoires": 0, "defaites": 0, "nulles": 0, "parties": 0, "manches": 0}
    parties: list[tuple] = []
    textes: list[str] = []
    todo = list(specs)
    slots: list[dict] = []
    while todo or slots:
        while todo and len(slots) < simultanees:
            s, side = todo.pop()
            g = Game("2J", "draft" if draft else None, seed=s, max_rounds=max_manches)
            agents = [a, b] if side == 0 else [b, a]
            slots.append({"g": g, "agents": agents, "side": side, "gen": None, "req": None, "seed": s,
                          "rech": [x.nouvelle_recherche(s + i) if isinstance(x, AgentRecherche) else None
                                   for i, x in enumerate(agents)]})
        # avancer jusqu'à une décision nécessitant une recherche
        for slot in list(slots):
            if slot["gen"] is not None:
                continue
            g = slot["g"]
            while not g.done:
                legal = g.legal_actions()
                p = g.to_move
                ag = slot["agents"][p]
                if len(legal) == 1:
                    g.apply(legal[0])
                elif isinstance(ag, AgentBot):
                    g.apply(ag.bot.rng.choice(legal) if g.in_draft else ag.bot.choose(g))
                else:
                    slot["gen"] = slot["rech"][p].generateur(g)
                    slot["req"] = next(slot["gen"])
                    break
            if g.done:
                _compter(res, g, slot["side"])
                parties.append((slot["seed"], slot["side"], _score(g, slot["side"]), g.round))
                if len(textes) < releves:
                    ag = slot["agents"]
                    textes.append(export_record(g, headers={   # relevé complet
                        "Type": "evaluation", "Blanc": ag[0].nom, "Noir": ag[1].nom}))
                slots.remove(slot)
        if not slots:
            continue
        # regrouper les requêtes par évaluateur
        groupes: dict[int, list] = {}
        for slot in slots:
            ev = slot["agents"][slot["g"].to_move].evaluateur
            groupes.setdefault(id(ev), [ev, []])[1].append(slot)
        for ev, groupe in groupes.values():
            flat = [r for s in groupe for r in s["req"]]
            reps = ev.evaluer(flat)
            pos = 0
            for slot in groupe:
                n = len(slot["req"])
                try:
                    slot["req"] = slot["gen"].send(reps[pos:pos + n])
                except StopIteration as fin:
                    slot["g"].apply(fin.value.action)
                    slot["gen"] = slot["req"] = None
                pos += n
    res["releves"] = textes
    res["detail"] = parties
    return resume(parties, res)


def _score(g: Game, side: int) -> int:
    return 0 if g.winner is None else (1 if g.winner == side else -1)


def resume(parties: list[tuple], res: dict | None = None) -> dict:
    """Bilan d'un match du point de vue de A à partir des parties (graine, camp de A, score de A
    (1, 0, -1), manches) : victoires, nulles, défaites, score, écart Elo, statistiques par paire."""
    res = dict(res or {})
    res.update(victoires=sum(1 for p in parties if p[2] > 0), nulles=sum(1 for p in parties if p[2] == 0),
               defaites=sum(1 for p in parties if p[2] < 0), parties=len(parties),
               manches=sum(p[3] for p in parties))
    res["score"] = (res["victoires"] + 0.5 * res["nulles"]) / max(res["parties"], 1)
    res["elo_diff"] = elo_depuis_score(res["score"], res["parties"])
    penta = pentanomial(parties)
    res["pentanomial"] = penta
    res.update(stats_paires(penta))
    return res


def pentanomial(parties: list[tuple]) -> list[int]:
    """Nombre de paires (même graine, camps inversés) où A marque 0, ½, 1, 1½, 2 points."""
    par_graine: dict[int, list[float]] = {}
    for graine, _, score, *_ in parties:
        par_graine.setdefault(graine, []).append((score + 1) / 2)
    comptes = [0] * 5
    for pts in par_graine.values():
        if len(pts) == 2:
            comptes[round(2 * sum(pts))] += 1
    return comptes


def _elo(s: float) -> float:
    s = min(max(s, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / s - 1.0)


def _score_elo(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def _moments(penta: list[int]) -> tuple[int, float, float]:
    n = sum(penta)
    if n == 0:
        return 0, 0.5, 0.0
    m = sum(c * k / 4 for k, c in enumerate(penta)) / n
    var = sum(c * (k / 4 - m) ** 2 for k, c in enumerate(penta)) / n
    return n, m, var


def stats_paires(penta: list[int]) -> dict:
    """Écart Elo et son intervalle de confiance à 95 %, estimés sur les scores de paires."""
    n, m, var = _moments(penta)
    if n == 0:
        return {"paires": 0}
    se = math.sqrt(var / n) if n > 1 else 0.5
    return {"paires": n, "elo": round(_elo(m), 1), "elo_bas": round(_elo(m - 1.96 * se), 1),
            "elo_haut": round(_elo(m + 1.96 * se), 1)}


def llr_sprt(penta: list[int], elo0: float, elo1: float) -> float:
    """Log-rapport de vraisemblance de H1 (écart elo1) contre H0 (écart elo0), approximation
    GSPRT sur les scores de paires : N (s1 − s0)(2 m − s0 − s1) / (2 σ²)."""
    n, m, var = _moments(penta)
    if n < 2:
        return 0.0
    var = max(var, 1e-4)
    s0, s1 = _score_elo(elo0), _score_elo(elo1)
    return n * (s1 - s0) * (2 * m - s0 - s1) / (2 * var)


def bornes_sprt(alpha: float = 0.05, beta: float = 0.05) -> tuple[float, float]:
    """(borne basse : H0 acceptée, borne haute : H1 acceptée)."""
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)


def _compter(res: dict, g: Game, side: int) -> None:
    res["parties"] += 1
    res["manches"] += g.round
    if g.winner is None:
        res["nulles"] += 1
    elif g.winner == side:
        res["victoires"] += 1
    else:
        res["defaites"] += 1


def elo_depuis_score(s: float, n: int = 0) -> float:
    eps = 0.5 / max(n, 1)
    s = min(max(s, eps), 1 - eps)
    return -400.0 * math.log10(1.0 / s - 1.0)


class ClassementElo:
    """Résultats cumulés et ajustement Bradley-Terry, persistés en JSON."""

    def __init__(self, chemin: Path | str | None, ancre: str = "glouton"):
        self.chemin = Path(chemin) if chemin is not None else None   # None : en mémoire seulement
        self.ancre = ancre
        self.resultats: list[dict] = []
        if self.chemin is not None and self.chemin.exists():
            self.resultats = json.loads(self.chemin.read_text())["resultats"]

    def ajouter(self, a: str, b: str, points_a: float, parties: int) -> None:
        self.resultats.append({"a": a, "b": b, "points_a": points_a, "parties": parties})

    def ajuster(self, iterations: int = 500) -> dict[str, float]:
        noms = sorted({r["a"] for r in self.resultats} | {r["b"] for r in self.resultats})
        if not noms:
            return {}
        # points (avec un léger a priori : 1 nulle virtuelle contre l'ancre)
        wins = {n: 0.0 for n in noms}
        games: dict[tuple[str, str], float] = {}
        for r in self.resultats:
            a, b, pa, n = r["a"], r["b"], r["points_a"], r["parties"]
            wins[a] += pa
            wins[b] += n - pa
            games[(a, b)] = games.get((a, b), 0) + n
            games[(b, a)] = games.get((b, a), 0) + n
        anc = self.ancre if self.ancre in noms else noms[0]
        for n in noms:
            if n != anc:
                games[(n, anc)] = games.get((n, anc), 0) + 1
                games[(anc, n)] = games.get((anc, n), 0) + 1
                wins[anc] += 0.5
                wins[n] += 0.5
        voisins: dict[str, list[tuple[str, float]]] = {n: [] for n in noms}
        for (i, j), g in games.items():
            voisins[i].append((j, g))
        gamma = {n: 1.0 for n in noms}
        for _ in range(iterations):
            new = {}
            for i in noms:
                den = sum(g / (gamma[i] + gamma[j]) for j, g in voisins[i])
                new[i] = wins[i] / den if den > 0 else gamma[i]
            ref = new[anc]
            gamma = {n: v / ref for n, v in new.items()}
        return {n: 400.0 * math.log10(max(v, 1e-12)) for n, v in gamma.items()}

    def sauver(self) -> None:
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        self.chemin.write_text(json.dumps({"resultats": self.resultats, "elo": self.ajuster()},
                                          indent=1, ensure_ascii=False))


# ------------------------------------------------------------ matchs parallèles
def agent_depuis_spec(spec: str, simulations: int = 200, dispositif: str = "cpu",
                      nom: str | None = None):
    """« chemin.pt » (réseau), « heur[:N] » (recherche heuristique) ou un bot (glouton, mcts:800…)."""
    from ..bots import make_bot
    from .evaluateurs import EvaluateurHeuristique
    from .recherche import ParamsRecherche
    nom = nom or spec
    reseau = spec_reseau(spec)
    if reseau is not None:
        from .evaluateurs import EvaluateurReseau
        chemin, reglages = reseau
        params = {"simulations": simulations, "m": 32, "bruit": False, "parallele": 4, **reglages}
        return AgentRecherche(nom, EvaluateurReseau.depuis_fichier(chemin, dispositif),
                              ParamsRecherche(**params))
    if spec == "heur" or spec.startswith("heur:"):
        n = int(spec.partition(":")[2] or simulations)
        return AgentRecherche(nom, EvaluateurHeuristique(), ParamsRecherche(simulations=n, bruit=False))
    return AgentBot(nom, make_bot(spec, seed=len(nom)))


def _match_morceau(args):
    (spec_a, nom_a, spec_b, nom_b, sims_a, sims_b, dispositif, graines, max_manches, releves,
     protocole) = args
    a = agent_depuis_spec(spec_a, sims_a, dispositif, nom_a)
    b = agent_depuis_spec(spec_b, sims_b, dispositif, nom_b)
    return match(a, b, graines=graines, max_manches=max_manches, simultanees=max(1, len(graines) * 2),
                 releves=releves, protocole=protocole)


def match_parallele(spec_a: str, spec_b: str, paires: int, seed: int, executeur,
                    n_morceaux: int, simulations: int = 200, simulations_b: int | None = None,
                    dispositif: str = "cpu", max_manches: int = 150,
                    nom_a: str | None = None, nom_b: str | None = None, releves: int = 0,
                    protocole: str = "aleatoire") -> dict:
    """Même résultat que `match`, réparti sur les processus d'un exécuteur (ProcessPoolExecutor)."""
    graines = graines_appariees(paires, seed)
    n = max(1, min(n_morceaux, len(graines)))
    morceaux = [graines[i::n] for i in range(n)]
    morceaux = [m for m in morceaux if m]
    # relevés répartis sur les premiers morceaux (un ou plusieurs chacun)
    parts = [releves // len(morceaux) + (1 if i < releves % len(morceaux) else 0)
             for i in range(len(morceaux))]
    taches = [(spec_a, nom_a or spec_a, spec_b, nom_b or spec_b, simulations,
               simulations_b or simulations, dispositif, m, max_manches, r, protocole)
              for m, r in zip(morceaux, parts)]
    parties: list[tuple] = []
    textes: list[str] = []
    for r in executeur.map(_match_morceau, taches):
        parties += r["detail"]
        textes += r["releves"]
    return resume(parties, {"releves": textes, "detail": parties})


# ------------------------------------------------------------ matchs entre réseaux (moteur Rust)
_EVALUATEURS: dict = {}   # (dispositif, compiler, emplacement) -> [chemin, date, évaluateur]


def evaluateur_reseau(chemin: str, dispositif: str = "cuda", compiler: bool = True, emplacement: int = 0):
    """Évaluateur par lots d'un modèle, réutilisé d'un match à l'autre : pour un autre modèle de
    même architecture, les poids sont recopiés en place (le réseau compilé est conservé)."""
    import os

    from .evaluateurs import EvaluateurReseau
    cle = (dispositif, compiler, emplacement)
    date = os.path.getmtime(chemin)
    e = _EVALUATEURS.get(cle)
    if e is not None and (e[0], e[1]) == (str(chemin), date):
        return e[2]
    if e is not None and e[2].recharger(chemin):
        e[0], e[1] = str(chemin), date
        return e[2]
    ev = EvaluateurReseau.depuis_fichier(chemin, dispositif, compiler=compiler)
    _EVALUATEURS[cle] = [str(chemin), date, ev]
    return ev


def _vue_de_a(parties: list[tuple], a: int = 0) -> list[tuple]:
    """Parties (graine, agent 0, agent 1, score du camp 0, manches) -> (graine, camp de a,
    score de a, manches), pour les parties où l'agent a joue."""
    out = []
    for g, a0, a1, s0, manches in parties:
        if a0 == a:
            out.append((g, 0, s0, manches))
        elif a1 == a:
            out.append((g, 1, -s0, manches))
    return out


def _agents_reseaux(chemins: list[str], reglages: list[dict], simulations: int, dispositif: str,
                    compiler: bool) -> tuple[list, list[dict]]:
    """Évaluateurs (un par fichier distinct, partagé entre agents) et réglages de recherche."""
    places: dict[str, int] = {}
    evs = []
    for c in chemins:
        k = places.setdefault(str(c), len(places))
        evs.append(evaluateur_reseau(c, dispositif, compiler, k))
    return evs, [{"simulations": simulations, **r} for r in reglages]


def match_reseaux(chemin_a: str, chemin_b: str, paires: int = 200, seed: int = 0,
                  simulations: int = 128, simulations_b: int | None = None,
                  agent_a: dict | None = None, agent_b: dict | None = None,
                  dispositif: str = "cuda", protocole: str = "draft", max_manches: int = 150,
                  simultanees: int = 1024, releves: int = 0, compiler: bool = True,
                  noms: tuple[str, str] | None = None, graines: list[int] | None = None,
                  arret=None) -> dict:
    """Match apparié entre deux réseaux (fichiers .pt) avec le moteur Rust ; même bilan que
    `match`. `agent_a`, `agent_b` : réglages de recherche de chaque agent (m, c_scale,
    coup_gagnant, cle_publique…), par défaut ceux du bot : meilleur coup, sans bruit, m = 32,
    coup gagnant toujours joué. Le même fichier des deux côtés partage un seul évaluateur (un
    lot GPU). `arret(parties vues de A)` : arrêt anticipé (voir `sprt_reseaux`)."""
    from . import rs
    if graines is None:
        graines = graines_appariees(paires, seed)
    evs, agents = _agents_reseaux([chemin_a, chemin_b], [agent_a or {}, agent_b or {}], simulations,
                                  dispositif, compiler)
    if simulations_b and "simulations" not in (agent_b or {}):
        agents[1]["simulations"] = simulations_b
    noms = noms or (Path(chemin_a).stem, Path(chemin_b).stem)
    parties = [p for g in graines for p in ((g, 0, 1), (g, 1, 0))]     # paires consécutives
    r = rs.jouer_match_rs(evs, agents, parties, protocole == "draft", max_manches, simultanees,
                          releves, list(noms), (lambda f: arret(_vue_de_a(f))) if arret else None)
    detail = _vue_de_a(r["parties"])
    return resume(detail, {"releves": r["releves"], "detail": detail, "secondes": round(r["secondes"], 1),
                           "evaluations": r["evaluations"]})


def matchs_contre(chemin: str, adversaires: list[tuple[str, dict, str]], paires: int = 100,
                  seed: int = 0, simulations: int = 128, dispositif: str = "cuda",
                  protocole: str = "draft", releves: int = 0, compiler: bool = True,
                  nom: str = "candidat", simultanees: int = 1024) -> dict[str, dict]:
    """Le réseau `chemin` contre plusieurs réseaux [(chemin, réglages de recherche, nom)], en
    matchs appariés sur les mêmes graines, joués ensemble dans un seul match Rust (lots pleins,
    une seule fin de match). Renvoie le bilan de chaque match (du point de vue de `chemin`)."""
    from . import rs
    if not adversaires:
        return {}
    evs, agents = _agents_reseaux([chemin] + [c for c, _, _ in adversaires],
                                  [{}] + [r for _, r, _ in adversaires], simulations, dispositif, compiler)
    graines = graines_appariees(paires, seed)
    parties = [p for g in graines for k in range(1, len(agents)) for p in ((g, 0, k), (g, k, 0))]
    noms = [nom] + [n for _, _, n in adversaires]
    r = rs.jouer_match_rs(evs, agents, parties, protocole == "draft", 150, simultanees,
                          releves * len(adversaires), noms)
    out = {}
    for k, n in enumerate(noms[1:], 1):
        detail = _vue_de_a([p for p in r["parties"] if k in (p[1], p[2])], 0)
        textes = [t for t in r["releves"] if f'"{n}"]' in t]
        out[n] = resume(detail, {"releves": textes[:releves], "detail": detail,
                                 "secondes": round(r["secondes"], 1)})
    return out


def sprt_reseaux(chemin_a: str, chemin_b: str, elo0: float = 0.0, elo1: float = 20.0,
                 alpha: float = 0.05, beta: float = 0.05, tranche: int = 100, max_paires: int = 2000,
                 seed: int = 0, suivi=None, **kw) -> dict:
    """Test séquentiel : A est-il plus fort que B d'au moins elo1 (H1) ou pas plus que elo0
    (H0) ? Les paires sont jouées au fil de l'eau (lots pleins) ; le test est fait à chaque paire
    terminée, jusqu'à la décision ou `max_paires`. `suivi(bilan)` est appelé toutes les
    `tranche` paires terminées."""
    import time as _time
    bas, haut = bornes_sprt(alpha, beta)
    t0 = _time.time()
    etat = {"vues": 0, "llr": 0.0}

    def arret(detail):
        penta = pentanomial(detail)
        n = sum(penta)
        if n == etat["vues"]:
            return False
        etat["vues"] = n
        etat["llr"] = llr = llr_sprt(penta, elo0, elo1)
        fini = llr <= bas or llr >= haut
        if suivi is not None and (n % tranche == 0 or fini):
            suivi(resume(detail, {"llr": round(llr, 3), "bornes": (round(bas, 3), round(haut, 3)),
                                  "secondes": round(_time.time() - t0, 1)}))
        return fini

    r = match_reseaux(chemin_a, chemin_b, graines=graines_appariees(max_paires, seed), arret=arret, **kw)
    llr = llr_sprt(r["pentanomial"], elo0, elo1)
    r.update(llr=round(llr, 3), bornes=(round(bas, 3), round(haut, 3)),
             decision="H1" if llr >= haut else ("H0" if llr <= bas else "indécis"))
    return r


# ------------------------------------------------------------ tournois
def spec_reseau(spec: str) -> tuple[str, dict] | None:
    """« chemin.pt » ou « chemin.pt:clé=valeur,… » (réglages de recherche : simulations, m,
    c_scale, cle_publique, coup_gagnant…) -> (chemin, réglages) ; None pour un bot."""
    import json
    i = spec.find(".pt")
    if i < 0 or (spec[i + 3:] and not spec[i + 3:].startswith(":")):
        return None
    reglages = {}
    for kv in filter(None, spec[i + 4:].split(",")):
        k, _, v = kv.partition("=")
        try:
            reglages[k.strip()] = json.loads(v)
        except ValueError:
            reglages[k.strip()] = v
    return spec[:i + 3], reglages


def nom_spec(spec: str) -> str:
    r = spec_reseau(spec)
    if r is None:
        return spec
    chemin, reglages = r
    return Path(chemin).stem + "".join(f"@{k}={v}" for k, v in reglages.items())


def tournoi(specs: list[str], paires: int = 100, seed: int = 0, simulations: int = 128,
            protocole: str = "draft", dispositif: str = "cuda", compiler: bool = True,
            executeur=None, n_morceaux: int = 1, suivi=None, simultanees: int = 2048) -> dict:
    """Toutes les paires d'agents s'affrontent sur les mêmes graines (matchs appariés). Les
    réseaux jouent tous leurs matchs ensemble dans un seul match Rust (lots pleins) ; les matchs
    avec un bot se jouent avec le moteur Python (processus d'`executeur` s'il y en a). Renvoie la
    matrice des résultats (score, pentanomial, écart Elo et son intervalle à 95 %) et le
    classement Bradley-Terry (premier agent = 0)."""
    from . import rs
    noms = []
    for sp in specs:
        n = nom_spec(sp)
        while n in noms:
            n += "'"
        noms.append(n)
    graines = graines_appariees(paires, seed)
    classement = ClassementElo(None, ancre=noms[0])
    matchs = {}

    def noter(i, j, r):
        classement.ajouter(noms[i], noms[j], r["victoires"] + 0.5 * r["nulles"], r["parties"])
        m = {k: r.get(k) for k in ("score", "pentanomial", "elo", "elo_bas", "elo_haut", "parties")}
        matchs[f"{noms[i]} | {noms[j]}"] = m
        if suivi is not None:
            suivi(noms[i], noms[j], m)

    reseaux = [k for k, sp in enumerate(specs) if spec_reseau(sp) is not None and rs.disponible()]
    couples = [(i, j) for i in range(len(specs)) for j in range(i + 1, len(specs))]
    if len(reseaux) >= 2:
        chemins = [spec_reseau(specs[k])[0] for k in reseaux]
        evs, agents = _agents_reseaux(chemins, [spec_reseau(specs[k])[1] for k in reseaux], simulations,
                                      dispositif, compiler)
        pos = {k: n for n, k in enumerate(reseaux)}
        parties = [p for g in graines for i, j in couples if i in pos and j in pos
                   for p in ((g, pos[i], pos[j]), (g, pos[j], pos[i]))]
        r = rs.jouer_match_rs(evs, agents, parties, protocole == "draft", 150, simultanees,
                              noms=[noms[k] for k in reseaux])
        for i, j in couples:
            if i in pos and j in pos:
                vus = [p for p in r["parties"] if {p[1], p[2]} == {pos[i], pos[j]}]
                noter(i, j, resume(_vue_de_a(vus, pos[i])))
    for i, j in couples:
        if f"{noms[i]} | {noms[j]}" in matchs:
            continue
        if executeur is not None:
            r = match_parallele(specs[i], specs[j], paires, seed, executeur, n_morceaux, simulations,
                                dispositif=dispositif, nom_a=noms[i], nom_b=noms[j], protocole=protocole)
        else:
            a = agent_depuis_spec(specs[i], simulations, dispositif, noms[i])
            b = agent_depuis_spec(specs[j], simulations, dispositif, noms[j])
            r = match(a, b, graines=graines, protocole=protocole)
        noter(i, j, r)
    return {"noms": noms, "paires": paires, "simulations": simulations, "protocole": protocole,
            "matchs": matchs, "elo": {k: round(v, 1) for k, v in classement.ajuster().items()}}
