"""Évaluation : matchs entre agents et classement Elo (Bradley-Terry).

* Les matchs sont **appariés** : chaque graine est jouée deux fois en
  inversant les camps, ce qui neutralise l'avantage du tirage des unités et
  de l'initiative et réduit fortement la variance.
* Les agents à recherche neuronale d'un même match jouent leurs parties en
  parallèle et partagent les lots d'évaluation.
* Le classement Elo est ajusté par maximum de vraisemblance (algorithme MM
  de Hunter pour Bradley-Terry) sur l'ensemble des résultats enregistrés,
  avec le bot glouton comme ancre (0 Elo).
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
          simultanees: int = 32, graines: list[int] | None = None, releves: int = 0) -> dict:
    """Joue 2 × `paires` parties entre a et b. Renvoie les statistiques du point de vue de a.

    `releves` : nombre de relevés .nch (premières parties terminées) renvoyés dans res["releves"].
    """
    if graines is None:
        graines = graines_appariees(paires, seed)
    specs = []
    for s in graines:
        specs += [(s, 0), (s, 1)]      # (graine, camp de a)
    res = {"victoires": 0, "defaites": 0, "nulles": 0, "parties": 0, "manches": 0}
    textes: list[str] = []
    todo = list(specs)
    slots: list[dict] = []
    while todo or slots:
        while todo and len(slots) < simultanees:
            s, side = todo.pop()
            g = Game("2J", seed=s)
            g.max_rounds = max_manches
            agents = [a, b] if side == 0 else [b, a]
            slots.append({"g": g, "agents": agents, "side": side, "gen": None, "req": None,
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
                    g.apply(ag.bot.choose(g))
                else:
                    slot["gen"] = slot["rech"][p].generateur(g)
                    slot["req"] = next(slot["gen"])
                    break
            if g.done:
                _compter(res, g, slot["side"])
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
    res["score"] = (res["victoires"] + 0.5 * res["nulles"]) / max(res["parties"], 1)
    res["elo_diff"] = elo_depuis_score(res["score"], res["parties"])
    res["releves"] = textes
    return res


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

    def __init__(self, chemin: Path | str, ancre: str = "glouton"):
        self.chemin = Path(chemin)
        self.ancre = ancre
        self.resultats: list[dict] = []
        if self.chemin.exists():
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
    if spec.endswith(".pt"):
        from .evaluateurs import EvaluateurReseau
        return AgentRecherche(nom, EvaluateurReseau.depuis_fichier(spec, dispositif),
                              ParamsRecherche(simulations=simulations, m=32, bruit=False, parallele=4))
    if spec == "heur" or spec.startswith("heur:"):
        n = int(spec.partition(":")[2] or simulations)
        return AgentRecherche(nom, EvaluateurHeuristique(), ParamsRecherche(simulations=n, bruit=False))
    return AgentBot(nom, make_bot(spec, seed=len(nom)))


def _match_morceau(args):
    (spec_a, nom_a, spec_b, nom_b, sims_a, sims_b, dispositif, graines, max_manches, releves) = args
    a = agent_depuis_spec(spec_a, sims_a, dispositif, nom_a)
    b = agent_depuis_spec(spec_b, sims_b, dispositif, nom_b)
    return match(a, b, graines=graines, max_manches=max_manches, simultanees=max(1, len(graines) * 2),
                 releves=releves)


def match_parallele(spec_a: str, spec_b: str, paires: int, seed: int, executeur,
                    n_morceaux: int, simulations: int = 200, simulations_b: int | None = None,
                    dispositif: str = "cpu", max_manches: int = 150,
                    nom_a: str | None = None, nom_b: str | None = None, releves: int = 0) -> dict:
    """Même résultat que `match`, réparti sur les processus d'un exécuteur (ProcessPoolExecutor)."""
    graines = graines_appariees(paires, seed)
    n = max(1, min(n_morceaux, len(graines)))
    morceaux = [graines[i::n] for i in range(n)]
    morceaux = [m for m in morceaux if m]
    # relevés répartis sur les premiers morceaux (un ou plusieurs chacun)
    parts = [releves // len(morceaux) + (1 if i < releves % len(morceaux) else 0)
             for i in range(len(morceaux))]
    taches = [(spec_a, nom_a or spec_a, spec_b, nom_b or spec_b, simulations,
               simulations_b or simulations, dispositif, m, max_manches, r)
              for m, r in zip(morceaux, parts)]
    res = {"victoires": 0, "defaites": 0, "nulles": 0, "parties": 0, "manches": 0}
    textes: list[str] = []
    for r in executeur.map(_match_morceau, taches):
        for k in res:
            res[k] += r[k]
        textes += r["releves"]
    res["score"] = (res["victoires"] + 0.5 * res["nulles"]) / max(res["parties"], 1)
    res["elo_diff"] = elo_depuis_score(res["score"], res["parties"])
    res["releves"] = textes
    return res
