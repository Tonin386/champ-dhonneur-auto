"""Analyse d'une position par l'IA, lisible par un humain (notation NCH)."""
from __future__ import annotations

import json
from pathlib import Path

from ..engine import Game
from ..notation import action_str, describe


def bot_analyse(simulations: int = 400, modele: str | None = None, dispositif: str | None = None):
    """Bot IA dont la recherche garde son arbre : ligne principale de chaque coup candidat."""
    from ..bots.neural import NeuralBot
    from .recherche import RechercheGumbel

    class RechercheAnalyse(RechercheGumbel):
        racine = None

        def _vague(self, game, me, racine, taches):
            self.racine = racine
            yield from super()._vague(game, me, racine, taches)

    bot = NeuralBot(modele=modele, simulations=simulations, dispositif=dispositif, seed=0)
    bot.recherche = RechercheAnalyse(bot.recherche.p, seed=0)
    return bot


def analyser(game: Game, bot, top: int = 5, observateur: int | None = None,
             solveur: bool = True, profondeur: int = 8) -> dict:
    """Recherche depuis le point de vue du joueur au trait et renvoie les meilleurs coups.

    La recherche n'utilise que l'information du joueur au trait (déterminisations). Le score
    est donné du point de vue d'Or (voir champ_dhonneur.score) ; `observateur` restreint la
    recherche de victoire forcée à l'information de ce joueur (None : omnisciente).
    """
    from ..score import Solveur, appreciation, bastions, depuis_valeur, texte_mat, texte_score
    if game.done:
        return {"coups": [], "valeur": None, "fini": game.result_label(), "bastions": bastions(game)}
    legal = game.legal_actions()
    sign = 1.0 if game.team(game.to_move) == 0 else -1.0
    mat = Solveur(observateur).chercher(game) if solveur else None
    rech = getattr(bot, "recherche", None)
    if rech is not None and hasattr(bot, "ev"):
        from .recherche import executer
        res = executer([rech.generateur(game, bot.simulations)], bot.ev)[0]
    else:
        bot.choose(game)
        res = bot.derniere
    racine = getattr(rech, "racine", None) if res.simulations else None
    ordre = sorted(range(len(res.legal)), key=lambda i: -res.politique[i])[:top]
    coups = []
    for i in ordre:
        a = res.legal[i]
        v_or = sign * float(res.q[i])
        c = {"coup": action_str(game, a), "description": describe(game, a),
             "probabilite": round(float(res.politique[i]), 3), "q": round(float(res.q[i]), 3),
             "visites": int(res.visites[i]), "score": round(depuis_valeur(v_or), 1),
             "texte": texte_score(depuis_valeur(v_or)), "action": a,
             "ligne": _ligne(game, a, racine, observateur, profondeur)}
        coups.append(c)
    if mat and mat["action"] is not None:
        # le coup gagnant en tête, avec sa distance
        c0 = next((c for c in coups if c["action"] == mat["action"]), None)
        if c0 is None:
            a = mat["action"]
            c0 = {"coup": action_str(game, a), "description": describe(game, a), "probabilite": 0.0,
                  "q": None, "visites": 0, "action": a, "ligne": _ligne(game, a, racine, observateur, profondeur)}
        coups = [c0] + [c for c in coups if c is not c0][:top - 1]
        c0["score"] = 99.9 if mat["equipe"] == 0 else -99.9
        c0["texte"] = texte_mat(mat["equipe"], mat["coups"])
    v_or = sign * float(res.valeur)
    score = depuis_valeur(v_or)
    out = {"coups": coups, "valeur": round(float(res.valeur), 3), "v_reseau": round(float(res.v_reseau), 3),
           "simulations": res.simulations, "force": len(legal) == 1, "v_or": round(v_or, 3),
           "score": round(score, 1), "texte": texte_score(score), "appreciation": appreciation(score),
           "bastions": bastions(game), "mat": None}
    if mat:
        out["mat"] = {"equipe": mat["equipe"], "coups": mat["coups"],
                      "coup": action_str(game, mat["action"]) if mat["action"] is not None else None}
        out["texte"] = texte_mat(mat["equipe"], mat["coups"])
        out["score"] = 99.9 if mat["equipe"] == 0 else -99.9
        out["appreciation"] = ("+−" if mat["equipe"] == 0 else "−+",
                               f"Victoire forcée {('Or', 'Argent')[mat['equipe']]} en {mat['coups']}")
    return out


def _ligne(game: Game, a0, racine, observateur: int | None, profondeur: int) -> list[str]:
    """Suite la plus explorée par la recherche après a0, groupée en coups (notation publique)."""
    if racine is None or a0 not in racine.enfants:
        return [action_str(game, a0, hidden=False)]
    from ..score import _mettre_en_main
    g = game.copy()
    coups: list[str] = []
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
    return coups


def texte(analyse: dict) -> str:
    if not analyse["coups"]:
        return "Partie terminée."
    lignes = []
    if analyse.get("valeur") is not None:
        symbole, libelle = analyse["appreciation"]
        lignes.append(f"Évaluation : {analyse['texte']}  {symbole} {libelle}  "
                      f"(bastions Or {analyse['bastions'][0]} – Argent {analyse['bastions'][1]}, "
                      f"{analyse['simulations']} simulations ; 10 = un bastion d'avance)")
    for i, c in enumerate(analyse["coups"], 1):
        lignes.append(f"  {i}. {c['coup']:<16} {c['texte']:>6}  {100 * c['probabilite']:5.1f} %  "
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
