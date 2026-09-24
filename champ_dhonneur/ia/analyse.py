"""Analyse d'une position par l'IA, lisible par un humain (notation NCH)."""
from __future__ import annotations

import json
from pathlib import Path

from ..engine import Game
from ..notation import action_str, describe


def analyser(game: Game, bot, top: int = 5) -> dict:
    """Recherche depuis le point de vue du joueur au trait et renvoie les meilleurs coups.

    Le bot n'utilise que l'information de ce joueur (déterminisations).
    """
    if game.done or len(game.legal_actions()) < 1:
        return {"coups": [], "valeur": None}
    legal = game.legal_actions()
    if len(legal) == 1:
        a = legal[0]
        return {"coups": [{"coup": action_str(game, a), "description": describe(game, a),
                           "probabilite": 1.0, "q": None, "visites": 0}],
                "valeur": None, "force": True}
    bot.choose(game)
    res = bot.derniere
    coups = [{"coup": action_str(game, a), "description": describe(game, a),
              "probabilite": round(p, 3), "q": round(q, 3), "visites": int(n)}
             for a, p, q, n in bot.analyse(top)]
    return {"coups": coups, "valeur": round(res.valeur, 3), "v_reseau": round(res.v_reseau, 3),
            "simulations": res.simulations, "force": False}


def texte(analyse: dict) -> str:
    if not analyse["coups"]:
        return "Partie terminée."
    lignes = []
    if analyse.get("valeur") is not None:
        v = analyse["valeur"]
        pct = 50 * (1 + v)
        lignes.append(f"Évaluation : {v:+.2f} (≈ {pct:.0f} % de chances de gain pour le joueur au trait, "
                      f"{analyse['simulations']} simulations)")
    for i, c in enumerate(analyse["coups"], 1):
        q = "" if c["q"] is None else f"  Q {c['q']:+.2f}"
        lignes.append(f"  {i}. {c['coup']:<16} {100 * c['probabilite']:5.1f} %{q}  {c['description']}")
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
