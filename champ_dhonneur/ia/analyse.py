"""Analyse d'une position par l'IA, lisible par un humain (notation NCH)."""
from __future__ import annotations

import json
from pathlib import Path

from ..engine import Game
from ..notation import action_str, describe


def bot_analyse(simulations: int = 400, modele: str | None = None, dispositif: str | None = None):
    """Bot IA dont la recherche garde son arbre (`recherche.racine`) : ligne principale des coups."""
    from ..bots.neural import NeuralBot, modele_par_defaut
    from ..score import K
    from .valeurs import lire_k
    bot = NeuralBot(modele=modele, simulations=simulations, dispositif=dispositif, seed=0)
    bot.k = lire_k(modele or modele_par_defaut(), K)      # échelle recalibrée par l'entraînement
    return bot


def analyser(game: Game, bot, top: int = 5, observateur: int | None = None,
             solveur: bool = True, profondeur: int = 8, limite=None, suivi=None,
             joue=None) -> dict:
    """Recherche depuis le point de vue du joueur au trait et renvoie les meilleurs coups.

    La recherche n'utilise que l'information du joueur au trait (déterminisations). Le score
    est donné du point de vue d'Or (voir champ_dhonneur.score) ; `observateur` restreint la
    recherche de victoire forcée à l'information de ce joueur (None : omnisciente).

    Les coups sont classés comme l'IA choisit le sien (`recherche.classement`) : le premier est
    celui qu'elle jouerait. `limite` (recherche.Limite) : recherche progressive, arrêtée par une
    durée, une profondeur ou une demande extérieure ; `suivi(analyse)` reçoit alors les analyses
    intermédiaires (`en_cours` vrai). Sans `limite` : `bot.simulations` simulations.
    `joue` : coup effectivement joué dans la partie depuis cette position (relecture) ; son score
    et son écart avec le meilleur coup sont ajoutés (`joue`).
    """
    from ..score import Solveur, bastions
    if game.done:
        return {"coups": [], "valeur": None, "fini": game.result_label(), "bastions": bastions(game)}
    mat = Solveur(observateur).chercher(game) if solveur else None
    rech = bot.recherche
    from .recherche import executer
    if limite is None:
        res = executer([rech.generateur(game, bot.simulations)], bot.ev)[0]
        return _formater(game, bot, res, mat, top, observateur, profondeur, joue)
    rappel = None
    if suivi is not None:
        def rappel(r):
            suivi(_formater(game, bot, r, mat, top, observateur, profondeur, joue))
    res = executer([rech.approfondir(game, limite, suivi=rappel)], bot.ev)[0]
    return _formater(game, bot, res, mat, top, observateur, profondeur, joue)


def _formater(game: Game, bot, res, mat: dict | None, top: int, observateur: int | None,
              profondeur: int, joue) -> dict:
    from ..score import appreciation, bastions, depuis_valeur, texte_mat, texte_score
    from .recherche import classement
    k = getattr(bot, "k", None)
    legal = res.legal
    sign = 1.0 if game.team(game.to_move) == 0 else -1.0
    racine = getattr(bot.recherche, "racine", None) if res.simulations else None
    ordre = res.infos.get("ordre") or classement(res.visites, res.politique)
    best = legal.index(res.action)
    ordre = [best] + [i for i in ordre if i != best]
    total = max(float(res.visites.sum()), 1.0)

    def ligne(a) -> dict:
        coups, equipes = _ligne(game, a, racine, observateur, profondeur)
        return {"ligne": coups, "equipes": equipes}

    def coup(i: int) -> dict:
        a = legal[i]
        v_or = sign * float(res.q[i])
        sc = depuis_valeur(v_or, k)
        return {"coup": action_str(game, a), "description": describe(game, a),
                "probabilite": round(float(res.politique[i]), 3), "q": round(float(res.q[i]), 3),
                "visites": int(res.visites[i]), "part": round(float(res.visites[i]) / total, 3),
                "score": round(sc, 1), "texte": texte_score(sc) if res.visites[i] > 0 else "—", "action": a,
                **ligne(a)}

    coups = [coup(i) for i in ordre[:top]]
    if mat and mat["action"] is not None:
        # le coup gagnant en tête, avec sa distance
        a = mat["action"]
        c0 = next((c for c in coups if c["action"] == a), None)
        if c0 is None:
            c0 = coup(legal.index(a)) if a in legal else {
                "coup": action_str(game, a), "description": describe(game, a), "probabilite": 0.0, "q": None,
                "visites": 0, "part": 0.0, "action": a, **ligne(a)}
        coups = [c0] + [c for c in coups if c is not c0][:top - 1]
        c0["score"] = 99.9 if mat["equipe"] == 0 else -99.9
        c0["texte"] = texte_mat(mat["equipe"], mat["coups"])
    # score de la position : celui du coup choisi (comme un moteur d'échecs) ; la moyenne de toutes
    # les simulations de la racine compterait aussi les coups médiocres explorés puis écartés
    v_pos = float(res.q[best]) if res.visites[best] > 0 else float(res.valeur)
    v_or = sign * v_pos
    score = depuis_valeur(v_or, k)
    infos = res.infos
    out = {"coups": coups, "valeur": round(v_pos, 3), "v_reseau": round(float(res.v_reseau), 3),
           "simulations": res.simulations, "force": len(legal) == 1, "v_or": round(v_or, 3),
           "score": round(score, 1), "texte": texte_score(score), "appreciation": appreciation(score),
           "bastions": bastions(game), "mat": None, "k": k,
           "profondeur": infos.get("profondeur"), "horizon": round(float(infos.get("horizon", 0.0)), 1),
           "horizon_max": infos.get("horizon_max"), "secondes": round(float(infos.get("secondes", 0.0)), 2),
           "en_cours": bool(infos.get("en_cours", False))}
    if joue is not None and joue in legal:
        i = legal.index(joue)
        c = next((c for c in coups if c["action"] == joue), None) or coup(i)
        meilleur = coups[0]
        ecart = abs(meilleur["score"] - c["score"]) if c is not meilleur else 0.0
        out["joue"] = {"coup": c["coup"], "score": c["score"], "texte": c["texte"], "visites": c["visites"],
                       "rang": ordre.index(i) + 1, "meilleur": c["action"] == meilleur["action"],
                       "ecart": round(ecart, 1),
                       "mat_manque": bool(mat and mat["action"] is not None and joue != mat["action"])}
    if mat:
        out["mat"] = {"equipe": mat["equipe"], "coups": mat["coups"],
                      "coup": action_str(game, mat["action"]) if mat["action"] is not None else None}
        out["texte"] = texte_mat(mat["equipe"], mat["coups"])
        out["score"] = 99.9 if mat["equipe"] == 0 else -99.9
        out["appreciation"] = ("+−" if mat["equipe"] == 0 else "−+",
                               f"Victoire forcée {('Or', 'Argent')[mat['equipe']]} en {mat['coups']}")
    return out


def _ligne(game: Game, a0, racine, observateur: int | None, profondeur: int) -> tuple[list[str], list[int]]:
    """Suite la plus explorée par la recherche après a0, groupée en coups (notation publique), et
    l'équipe qui joue chacun de ces coups (0 Or, 1 Argent)."""
    if racine is None or a0 not in racine.enfants:
        return [action_str(game, a0, hidden=False)], [game.team(game.to_move)]
    from ..score import _mettre_en_main
    g = game.copy()
    coups: list[str] = []
    equipes: list[int] = []
    node, a = racine, a0
    for _ in range(profondeur):
        p = g.to_move
        principal = not g.pending
        if principal and observateur is not None and p != observateur and a.coin:
            _mettre_en_main(g.players[p], a.coin)   # pièce supposée par la recherche
        if a not in g.legal_actions():
            break
        s = action_str(g, a, hidden=False)
        if principal or not coups:
            coups.append(s)
            equipes.append(g.team(p))
        elif a.kind == "rg_reserve":
            coups[-1] += "(R)"
        elif a.kind not in ("skip", "rg_unit"):
            coups[-1] += ">" + s
        g.apply(a)
        node = node.enfants.get(a)
        if g.done or node is None or not node.enfants:
            break
        a, suivant = max(node.enfants.items(), key=lambda kv: kv[1].n)
        if suivant.n < 2:
            break
    return coups, equipes


def texte(analyse: dict) -> str:
    if not analyse["coups"]:
        return "Partie terminée."
    lignes = []
    if analyse.get("valeur") is not None:
        symbole, libelle = analyse["appreciation"]
        prof = (f"profondeur {analyse['profondeur']}, horizon {analyse['horizon']:.1f} coups "
                f"(max {analyse['horizon_max']}), {analyse['secondes']:.1f} s, "
                if analyse.get("profondeur") else "")
        lignes.append(f"Évaluation : {analyse['texte']}  {symbole} {libelle}  "
                      f"(bastions Or {analyse['bastions'][0]} – Argent {analyse['bastions'][1]}, "
                      f"{prof}{analyse['simulations']} simulations ; 10 = un bastion d'avance)")
    for i, c in enumerate(analyse["coups"], 1):
        lignes.append(f"  {i}. {c['coup']:<16} {c['texte']:>6}  {c['visites']:>7} sim.  "
                      f"{' '.join(c['ligne'][1:])}")
    return "\n".join(lignes)


def suivi(dossier: str | Path, dernieres: int = 30) -> str:
    """Tableau de bord texte d'un entraînement (journal.jsonl + elo.json)."""
    d = Path(dossier)
    lignes = [json.loads(l) for l in (d / "journal.jsonl").read_text(encoding="utf-8").splitlines() if l]
    out = [f"Entraînement {d} — {len(lignes)} itérations"]
    out.append(f"{'it':>5} {'parties':>7} {'nulles':>6} {'manches':>7} {'exemples':>8} {'pol':>6} "
               f"{'val':>6} {'préc':>5} {'préc*':>5} {'part./h':>8} {'Elo':>6}")
    for l in lignes[-dernieres:]:
        st, ap = l["autojeu"], l["apprentissage"]
        n = max(st["parties"], 1)
        elo = f"{l['evaluation']['elo']:+.0f}" if "evaluation" in l else ""
        out.append(f"{l['iteration']:>5} {st['parties']:>7} {st['nulles'] / n:>6.0%} "
                   f"{st['manches'] / n:>7.1f} {l['exemples']:>8} "
                   f"{ap.get('perte_politique', float('nan')):>6.3f} {ap.get('perte_valeur', float('nan')):>6.3f} "
                   f"{ap.get('precision_valeur', float('nan')):>5.2f} "
                   f"{(l.get('validation') or {}).get('precision_valeur', float('nan')):>5.2f} "
                   f"{st.get('parties_par_heure', 0):>8} {elo:>6}")
    out.append("préc = précision de valeur sur les exemples d'apprentissage ; préc* = sur des parties "
               "inédites (écart important = sur-apprentissage)")
    elo_path = d / "elo.json"
    if elo_path.exists():
        elo = json.loads(elo_path.read_text())["elo"]
        out.append("\nClassement Elo (glouton = 0) :")
        for nom, e in sorted(elo.items(), key=lambda kv: -kv[1])[:10]:
            out.append(f"  {nom:<16} {e:+7.0f}")
    etat = d / "etat.json"
    if etat.exists():
        out.append(f"\nMeilleur modèle : {json.loads(etat.read_text()).get('meilleur')} → {d / 'modeles' / 'meilleur.pt'}")
    return "\n".join(out)
