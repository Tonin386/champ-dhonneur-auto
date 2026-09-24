"""Interface console de Champ d'honneur.

    python -m champ_dhonneur.cli jouer --blanc humain --noir mcts
    python -m champ_dhonneur.cli arene mcts glouton --parties 20
    python -m champ_dhonneur.cli relire partie.nch
    python -m champ_dhonneur.cli serveur --port 8000
    python -m champ_dhonneur.cli entrainer --config configs/gpu.json
    python -m champ_dhonneur.cli evaluer runs/principal/modeles/meilleur.pt glouton --paires 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .arena import match
from .bots import BOT_NAMES, make_bot
from .engine import Game, IllegalAction
from .notation import (AmbiguousNotation, action_str, describe, export_record, find_action,
                       import_record, team_name)
from .render import board_text, full_text
from .units import UNITS, unit_name

HELP = """Commandes :
  <coup>        jouer un coup en notation NCH (ex. S@b1, Sb1-b2, Sb2xb3, Sb2^, $P{X}, --{*})
  <numéro>      jouer le coup n° de la liste
  ?             lister les coups légaux
  u             cartes des unités en jeu
  p             réafficher le plateau
  r             afficher le relevé de la partie
  annuler       revenir à votre décision précédente
  sauver <f>    enregistrer le relevé complet dans un fichier
  quitter       abandonner
Notation : voir docs/NOTATION.md — MAJUSCULE = Blanc, minuscule = Noir sur le plateau."""


def parse_units(s: str | None, mode: str):
    if s in (None, "aleatoire", "premiere"):
        return s
    groups = s.upper().split("/")
    return [list(g) for g in groups]


def side_label(game: Game, p: int) -> str:
    t = team_name(game.team(p))
    return f"J{p + 1} {t}" if game.n_players == 4 else t


def rebuild(game: Game, n: int) -> Game:
    g = Game(game.mode, [p.units for p in game.players], seed=game.seed,
             first=game.first_player, max_rounds=game.max_rounds)
    for _, _, a in game.log[:n]:
        g.apply(a)
    return g


def human_turn(game: Game, humans: set[int]) -> Game | None:
    p = game.to_move
    legal = game.legal_actions()
    label = game.pending_label()
    print()
    print(full_text(game, viewer=p if len(humans) > 0 else None))
    if label:
        print(f"\n» {label}")
    while True:
        try:
            raw = input(f"\n{side_label(game, p)} > ").strip()
        except EOFError:
            return None
        if not raw:
            continue
        if raw in ("quitter", "q"):
            return None
        if raw == "?":
            for i, a in enumerate(legal, 1):
                print(f"  {i:>3}. {action_str(game, a):<16} {describe(game, a)}")
            continue
        if raw in ("aide", "h", "help"):
            print(HELP)
            continue
        if raw == "p":
            print(board_text(game))
            continue
        if raw == "u":
            for pl in game.players:
                for u in pl.units:
                    d = UNITS[u]
                    print(f"  [{side_label(game, pl.idx)}] {u} {d.name} (x{d.count}) — "
                          f"{('Tactique : ' + d.tactic) if d.tactic else 'Pas de tactique'} {d.ability}")
            continue
        if raw == "r":
            print(export_record(game, hidden=False))
            continue
        if raw.startswith("sauver"):
            path = raw.split(maxsplit=1)[1] if " " in raw else "partie.nch"
            Path(path).write_text(export_record(game), encoding="utf-8")
            print(f"Relevé enregistré dans {path}")
            continue
        if raw == "annuler":
            idx = [i for i, (_, pp, _) in enumerate(game.log) if pp in humans]
            if not idx:
                print("Rien à annuler.")
                continue
            return rebuild(game, idx[-1])
        try:
            if raw.isdigit():
                a = legal[int(raw) - 1]
            else:
                a = find_action(game, raw)
        except AmbiguousNotation as e:
            print("Précisez la pièce utilisée :", ", ".join(action_str(game, c) for c in e.candidates))
            continue
        except (IllegalAction, IndexError) as e:
            print(f"{e}. Tapez « ? » pour la liste des coups, « aide » pour l'aide.")
            continue
        game.apply(a)
        return game


def cmd_play(args) -> None:
    mode = args.mode
    n = 2 if mode == "2J" else 4
    if n == 2:
        roles = [args.blanc, args.noir]
    else:
        roles = (args.joueurs or "humain,mcts,mcts,mcts").split(",")
    game = Game(mode, parse_units(args.unites, mode), seed=args.graine)
    bots = {i: make_bot(r, seed=i) for i, r in enumerate(roles) if r not in ("humain", "human")}
    humans = {i for i in range(n) if i not in bots}
    print(f"Champ d'honneur — {mode}, graine {game.seed}. Tapez « aide » à tout moment.")
    for pl in game.players:
        print(f"  {side_label(game, pl.idx)} : " + ", ".join(unit_name(u) for u in pl.units)
              + f"  ({roles[pl.idx]})")
    print(f"  Initiative : {side_label(game, game.initiative)}")
    while not game.done:
        p = game.to_move
        if p in bots:
            a = bots[p].choose(game)
            print(f"  {side_label(game, p)} joue {action_str(game, a, hidden=False)}  — {describe(game, a) if a.kind not in ('pass', 'initiative', 'recruit') else ''}")
            game.apply(a)
        else:
            nxt = human_turn(game, humans)
            if nxt is None:
                print("Partie abandonnée.")
                break
            game = nxt
    print()
    print(board_text(game))
    print(f"\nRésultat : {game.result_label()} — vainqueur : {team_name(game.winner) if game.done else '—'}")
    if args.releve:
        Path(args.releve).write_text(export_record(game), encoding="utf-8")
        print(f"Relevé enregistré dans {args.releve}")


def cmd_arena(args) -> None:
    score = match(args.a, args.b, games=args.parties, mode=args.mode, seed=args.graine,
                  save_dir=args.dossier)
    total = sum(score.values())
    for k, v in score.most_common():
        print(f"  {k:<12} {v:>4}  ({100 * v / total:.0f} %)")


def cmd_replay(args) -> None:
    text = Path(args.fichier).read_text(encoding="utf-8")
    g = import_record(text)
    print(full_text(g, None))
    print(f"\nRésultat : {g.result_label()}")


def cmd_serve(args) -> None:
    import uvicorn
    uvicorn.run("champ_dhonneur.server.app:app", host=args.hote, port=args.port)


def cmd_train(args) -> None:
    import json

    from .ia.entrainement import ConfigEntrainement, entrainer
    cfg = ConfigEntrainement()
    if args.config:
        cfg = ConfigEntrainement.depuis_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    if args.dossier:
        cfg.dossier = args.dossier
    precedent = Path(cfg.dossier) / "config.json"
    if not args.config and precedent.exists():
        # reprise : on repart de la configuration enregistrée
        cfg = ConfigEntrainement.depuis_dict(json.loads(precedent.read_text(encoding="utf-8")))
    cfg.appliquer(args.reglage or [])
    entrainer(cfg)


def cmd_evaluate(args) -> None:
    from .ia.evaluation import ClassementElo, agent_depuis_spec, match, match_parallele
    nom_a = Path(args.a).stem if args.a.endswith(".pt") else args.a
    nom_b = Path(args.b).stem if args.b.endswith(".pt") else args.b
    if nom_b == nom_a:
        nom_b += "'"
    sims_b = args.simulations_b or args.simulations
    if args.travailleurs and args.travailleurs > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        from .ia.entrainement import _init_travailleur
        with ProcessPoolExecutor(args.travailleurs, mp_context=mp.get_context("spawn"),
                                 initializer=_init_travailleur) as ex:
            r = match_parallele(args.a, args.b, args.paires, args.graine, ex, args.travailleurs,
                                args.simulations, sims_b, args.dispositif, nom_a=nom_a, nom_b=nom_b)
    else:
        a = agent_depuis_spec(args.a, args.simulations, args.dispositif, nom_a)
        b = agent_depuis_spec(args.b, sims_b, args.dispositif, nom_b)
        r = match(a, b, paires=args.paires, seed=args.graine, simultanees=args.simultanees)
    print(f"{nom_a} contre {nom_b} : +{r['victoires']} ={r['nulles']} -{r['defaites']} "
          f"sur {r['parties']} parties appariées — score {r['score']:.3f}, "
          f"écart Elo ≈ {r['elo_diff']:+.0f}, {r['manches'] / max(r['parties'], 1):.1f} manches/partie")
    if args.elo:
        c = ClassementElo(args.elo)
        c.ajouter(nom_a, nom_b, r["victoires"] + 0.5 * r["nulles"], r["parties"])
        c.sauver()
        for nom, e in sorted(c.ajuster().items(), key=lambda kv: -kv[1]):
            print(f"  {nom:<24} {e:+7.0f}")


def cmd_analyse(args) -> None:
    from .ia.analyse import analyser, bot_analyse, texte
    g = import_record(Path(args.fichier).read_text(encoding="utf-8"))
    n = len(g.log) if args.coup is None else args.coup
    g = rebuild(g, n)
    bot = bot_analyse(args.simulations, args.modele, args.dispositif)
    print(board_text(g))
    print(f"\nAprès {n} décisions — au trait : {side_label(g, g.to_move)}")
    if g.pending_label():
        print(f"» {g.pending_label()}")
    print(texte(analyser(g, bot, args.top)))


def cmd_calibrer(args) -> None:
    from .score import K, calibrer
    fichiers = [f for d in args.dossiers for f in (sorted(Path(d).glob("*.nch")) if Path(d).is_dir() else [Path(d)])]
    r = calibrer(fichiers)
    print(f"{len(fichiers)} parties, {r['positions']} positions : K = {r['K']:.2f} "
          f"(actuel {K:.2f}) — un bastion d'avance ≈ {100 * r['gain_1_bastion']:.0f} % de victoires")
    print("Pour l'adopter : champ_dhonneur/score.py, constante K.")


def cmd_bench(args) -> None:
    import json

    from .ia.banc import executer_banc
    from .ia.entrainement import ConfigEntrainement
    cfg = ConfigEntrainement()
    if args.config:
        cfg = ConfigEntrainement.depuis_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    cfg.appliquer(args.reglage or [])
    liste = [int(x) for x in args.travailleurs.split(",")] if args.travailleurs else None
    executer_banc(cfg, liste, args.duree)


def cmd_follow(args) -> None:
    from .ia.analyse import suivi
    print(suivi(args.dossier, args.dernieres))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="champ", description="Champ d'honneur — moteur, bots, interface")
    sub = ap.add_subparsers(dest="cmd", required=True)
    roles = ["humain"] + BOT_NAMES
    p = sub.add_parser("jouer", help="jouer une partie en console")
    p.add_argument("--mode", choices=["2J", "4J"], default="2J")
    p.add_argument("--blanc", default="humain", help=f"{roles} ou mcts:2000, mcts:t=3")
    p.add_argument("--noir", default="mcts")
    p.add_argument("--joueurs", help="4J : rôles séparés par des virgules (J1..J4)")
    p.add_argument("--unites", default="aleatoire", help="aleatoire | premiere | SPXH/ACLE")
    p.add_argument("--graine", type=int)
    p.add_argument("--releve", help="fichier .nch où enregistrer la partie")
    p.set_defaults(fn=cmd_play)
    a = sub.add_parser("arene", help="faire s'affronter deux bots")
    a.add_argument("a")
    a.add_argument("b")
    a.add_argument("--parties", type=int, default=20)
    a.add_argument("--mode", choices=["2J", "4J"], default="2J")
    a.add_argument("--graine", type=int, default=0)
    a.add_argument("--dossier", help="dossier où enregistrer les relevés")
    a.set_defaults(fn=cmd_arena)
    r = sub.add_parser("relire", help="rejouer un relevé .nch et afficher la position finale")
    r.add_argument("fichier")
    r.set_defaults(fn=cmd_replay)
    s = sub.add_parser("serveur", help="lancer l'interface web")
    s.add_argument("--hote", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)
    t = sub.add_parser("entrainer", help="apprentissage par renforcement (auto-jeu)")
    t.add_argument("--config", help="fichier JSON de configuration (voir configs/)")
    t.add_argument("--dossier", help="dossier de l'entraînement (reprise automatique)")
    t.add_argument("--set", dest="reglage", action="append", metavar="CLÉ=VALEUR",
                   help="réglage ponctuel, ex. --set autojeu.simulations=128 --set travailleurs=8")
    t.set_defaults(fn=cmd_train)
    e = sub.add_parser("evaluer", help="match apparié entre deux agents (bots ou modèles .pt)")
    e.add_argument("a", help="bot (glouton, mcts:800, heur…) ou chemin d'un modèle .pt")
    e.add_argument("b")
    e.add_argument("--paires", type=int, default=20, help="nombre de paires de parties (camps inversés)")
    e.add_argument("--simulations", type=int, default=200)
    e.add_argument("--simulations-b", type=int)
    e.add_argument("--dispositif", default="cpu")
    e.add_argument("--simultanees", type=int, default=32)
    e.add_argument("--graine", type=int, default=0)
    e.add_argument("--elo", help="fichier JSON de classement à mettre à jour")
    e.add_argument("--travailleurs", type=int, default=0,
                   help="répartir les parties sur N processus (0 = un seul processus)")
    e.set_defaults(fn=cmd_evaluate)
    an = sub.add_parser("analyser", help="analyse IA d'une position d'un relevé .nch")
    an.add_argument("fichier")
    an.add_argument("--coup", type=int, help="nombre de décisions à rejouer (défaut : toutes)")
    an.add_argument("--modele")
    an.add_argument("--simulations", type=int, default=400)
    an.add_argument("--dispositif", default="cpu")
    an.add_argument("--top", type=int, default=5)
    an.set_defaults(fn=cmd_analyse)
    ca = sub.add_parser("calibrer", help="recalculer l'échelle du score (10 = un bastion d'avance)")
    ca.add_argument("dossiers", nargs="*", default=["runs/continu/parties"],
                    help="relevés .nch ou dossiers qui en contiennent")
    ca.set_defaults(fn=cmd_calibrer)
    bc = sub.add_parser("banc", help="mesurer le matériel et recommander les réglages d'auto-jeu")
    bc.add_argument("--config", help="configuration d'entraînement à tester")
    bc.add_argument("--travailleurs", help="liste de nombres de processus à essayer, ex. 4,6,7,8")
    bc.add_argument("--duree", type=float, default=45.0, help="secondes par essai")
    bc.add_argument("--set", dest="reglage", action="append", metavar="CLÉ=VALEUR")
    bc.set_defaults(fn=cmd_bench)
    su = sub.add_parser("suivi", help="tableau de bord d'un entraînement")
    su.add_argument("dossier", nargs="?", default="runs/principal")
    su.add_argument("--dernieres", type=int, default=30)
    su.set_defaults(fn=cmd_follow)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
