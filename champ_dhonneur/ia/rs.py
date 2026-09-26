"""Pont vers le moteur Rust (module `champ_rs`, compilé depuis rust/).

Le moteur Rust joue les parties d'auto-jeu entières (règles, déterminisations, recherche
Gumbel IS-MCTS, encodage) ; Python n'évalue que les lots de positions avec le réseau. Les
règles sont identiques à celles du moteur Python, vérifiées décision par décision par
tests/test_rust.py, et les graines donnent les mêmes pioches : chaque relevé produit par Rust
est rejoué et exporté par le moteur Python, ce qui contrôle aussi la cohérence en production.
"""
from __future__ import annotations

import time

import numpy as np

from ..engine import Action, Game
from ..notation import export_record
from .encodage import ALL_LETTERS, CELL_F, GLOB_F, NONE_CELL, UNIT_F

try:
    import champ_rs
except ImportError:   # module non compilé : l'auto-jeu reste en Python
    champ_rs = None

KINDS = list(champ_rs.NOMS_TYPES) if champ_rs else []
# 255 : pièce d'une action face cachée d'un autre joueur, dans les lignes de la recherche Rust
LETTRE = {i + 1: u for i, u in enumerate(ALL_LETTERS)} | {17: "*", 255: "?"}
CODE = {u: i for i, u in LETTRE.items()}


def disponible() -> bool:
    return champ_rs is not None


def action_python(t) -> Action:
    genre, piece, unite, extra, cases = t
    return Action(KINDS[genre], LETTRE.get(piece), LETTRE.get(unite), tuple(cases), LETTRE.get(extra))


def etat_jeu(g: Game) -> dict:
    """État complet d'une partie du moteur Python au format de `champ_rs.Jeu.etat`, plus le draft :
    `champ_rs.Jeu.depuis_etat(etat_jeu(g))` reconstruit la même partie dans le moteur Rust (bot et
    analyse avec la recherche Rust). Testé décision par décision (tests/test_rust.py)."""
    from .encodage import COIN_ID, PENDING_ID
    code = lambda c: 0 if c is None else COIN_ID[c]  # noqa: E731
    d = g.draft
    return {
        "manche": g.round, "courant": g.current, "au_trait": g.to_move, "fini": g.done,
        "gagnant": -1 if g.winner is None else g.winner, "marqueurs": list(g.markers_left),
        "premier": g.first_player, "initiative": g.initiative, "init_bougee": g.initiative_moved,
        "premier_manche": g.round_first, "en_draft": g.in_draft,
        "dispo": "".join(d.available) if g.in_draft else "",
        "unites": ["".join(p.units) for p in g.players],
        "controle": [(l, -1 if g.control[l] is None else g.control[l]) for l in g.spec.locations],
        "plateau": [(pos, u.owner, u.utype, u.coins) for pos, u in g.board.items()],
        "joueurs": [("".join(p.bag), "".join(p.hand), "".join(p.disc_up), "".join(p.disc_down),
                     [(u, p.reserve.get(u, 0)) for u in p.units], sorted(p.lost.items()))
                    for p in g.players],
        "attentes": [(PENDING_ID[pd.kind], pd.player, pd.pos, code(pd.coin), pd.drawn, list(pd.positions))
                     for pd in g.pending],
        "draft": d is not None, "cartes": "".join(sorted(d.pool)) if d is not None else "",
        "etape_draft": d.step if d is not None else 0, "choisit": d.first if d is not None else 0,
    }


def action_rust(a: Action) -> tuple:
    return (KINDS.index(a.kind), CODE.get(a.coin, 0), CODE.get(a.unit, 0), CODE.get(a.extra, 0),
            list(a.cells))


class EvaluateurLotUniforme:
    """Politique uniforme, valeur nulle (tests)."""

    def evaluer_lot(self, x: dict) -> tuple[np.ndarray, np.ndarray]:
        b, a = x["acts"].shape[:2]
        return np.zeros((b, a), np.float32), np.zeros(b, np.float32)


def lot_numpy(lot: dict) -> dict[str, np.ndarray]:
    """Lot renvoyé par champ_rs (bytes) -> tableaux numpy (entrées du réseau)."""
    b, a_len = lot["b"], lot["a_len"]
    f = lambda k, dt, *forme: np.frombuffer(lot[k], dt).reshape(b, *forme)  # noqa: E731
    return {"cell_i": f("cell_i", np.int64, 37, 4), "cell_f": f("cell_f", np.float32, 37, CELL_F),
            "unit_i": f("unit_i", np.int64, 8, 2), "unit_f": f("unit_f", np.float32, 8, UNIT_F),
            "glob_i": f("glob_i", np.int64, 2), "glob_f": f("glob_f", np.float32, GLOB_F),
            "acts": f("acts", np.int64, a_len, 7)}


def _donnees(r: dict) -> dict[str, np.ndarray]:
    n = r["n"]
    if n == 0:
        return {"z": np.zeros(0, np.float32)}
    g = lambda k, dt, *forme: np.frombuffer(r[k], dt).reshape(n, *forme)  # noqa: E731
    return {"cell_i": g("cell_i", np.int8, 37, 4), "cell_f": g("cell_f", np.float32, 37, CELL_F).astype(np.float16),
            "unit_i": g("unit_i", np.int8, 8, 2), "unit_f": g("unit_f", np.float32, 8, UNIT_F).astype(np.float16),
            "glob_i": g("glob_i", np.int8, 2), "glob_f": g("glob_f", np.float32, GLOB_F).astype(np.float16),
            "acts": g("acts", np.int8, 64, 7).copy(), "pi": g("pi", np.float32, 64).astype(np.float16),
            "n_act": g("n_act", np.int16).copy(), "z": g("z", np.float32).copy(),
            "q": g("q", np.float32).copy(), "lieux": g("lieux", np.int8, 37).copy(),
            "marge": g("marge", np.int8).copy(), "main_adv": g("main_adv", np.int8, 18).copy(),
            "debut": g("debut", np.int8).copy()}


def releve(graine: int, actions: list, resultat: str, max_manches: int, entetes: dict,
           draft: bool = False) -> str:
    """Rejoue une partie du moteur Rust dans le moteur Python et exporte son relevé."""
    g = Game("2J", "draft" if draft else None, seed=graine, max_rounds=max_manches)
    for t in actions:
        g.apply(action_python(t))
    if g.result_label() != resultat:
        raise RuntimeError(f"moteurs divergents (graine {graine}) : {g.result_label()} contre {resultat}")
    return export_record(g, headers=entetes)


class DiffuseurRs:
    """Diffusion en direct (même fichier que direct.Diffuseur) d'une partie du moteur Rust."""

    def __init__(self, chemin: str, total: int, periode: float = 0.25):
        from pathlib import Path
        self.chemin, self.total, self.periode = Path(chemin), total, periode
        self.n, self.t, self.graine = -1, 0.0, None

    def observer(self, aj, max_manches: int, forcer: bool = False) -> None:
        if not forcer and time.time() - self.t < self.periode:
            return
        s = aj.suivie()
        if s is None:
            return
        graine, premier, armees, actions, fini, finies, draft = s
        if graine == self.graine and len(actions) == self.n and not fini:
            return
        self.graine, self.n, self.t = graine, len(actions), time.time()
        from .direct import _ecrire
        try:
            _ecrire(self.chemin, {
                "id": f"{graine}", "mode": "2J", "unites": [list(a) for a in armees],
                "setup": Game("2J", "draft", seed=graine, max_rounds=max_manches).setup if draft else None,
                "graine": graine, "initiative": premier, "max_manches": max_manches,
                "actions": [[a.kind, a.coin, a.unit, list(a.cells), a.extra]
                            for a in map(action_python, actions)],
                "fini": fini, "parties": finies, "total": self.total, "maj": self.t})
        except OSError:
            pass   # la diffusion ne doit jamais interrompre l'auto-jeu


_CONTINU: dict = {}   # auto-jeu continu : (réglages) -> [AutoJeu, lot en attente], par processus


def jouer_parties_rs(evaluateur, P, seed: int | None = None) -> tuple[dict, dict]:
    """Même contrat que autojeu.jouer_parties, avec le moteur Rust.

    `evaluateur.evaluer_lot(x)` reçoit les entrées numpy du réseau et renvoie
    (logits (B, A) float32, valeurs (B,) float32 dans [-1, 1]).

    `P.continu` : l'auto-jeu persiste dans le processus d'un appel à l'autre ; chaque appel
    joue jusqu'à ce que `P.parties` parties soient terminées et renvoie leurs exemples, les
    parties en cours continuent à l'appel suivant (avec le réseau d'alors). Les lots restent
    pleins : pas de fin d'itération où les parties se terminent une à une.
    """
    t0 = time.time()
    params = {k: getattr(P, k) for k in ("parties", "simultanees", "simulations", "simulations_rapides",
                                         "p_complete", "max_manches", "m", "parallele", "c_visit",
                                         "c_scale", "releves", "p_draft", "simulations_draft",
                                         "meilleur_coup", "draft_exact")}
    continu = bool(getattr(P, "continu", False))
    graine = int(seed if seed is not None else time.time_ns()) % 2**63
    if continu:
        cle = tuple(sorted((k, v) for k, v in params.items() if k not in ("parties", "releves")))
        etat = _CONTINU.get(cle)
        if etat is None:
            _CONTINU.clear()
            aj = champ_rs.AutoJeu({**params, "parties": 2**62, "releves": params["releves"]}, graine)
            etat = _CONTINU[cle] = [aj, aj.etape()]
        aj, lot = etat
    else:
        aj = champ_rs.AutoJeu(params, graine)
        lot = aj.etape()
    diffuseur = DiffuseurRs(P.direct, P.parties) if getattr(P, "direct", "") else None
    while lot is not None:
        if diffuseur is not None:
            diffuseur.observer(aj, P.max_manches)
        if P.duree_max and time.time() - t0 > P.duree_max:
            break   # mesure de débit : on abandonne les parties en cours
        if continu and aj.terminees() >= P.parties:
            break   # le lot en attente sera évalué à l'appel suivant
        logits, valeurs = evaluateur.evaluer_lot(lot_numpy(lot))
        lot = aj.etape(np.ascontiguousarray(logits, np.float32).tobytes(), lot["a_len"],
                       np.ascontiguousarray(valeurs, np.float32).tobytes())
    if continu:
        _CONTINU[cle][1] = lot
    if diffuseur is not None:
        diffuseur.observer(aj, P.max_manches, forcer=True)   # position finale de la partie suivie
    r = aj.resultats()
    stats = dict(r["stats"])
    stats["secondes"] = time.time() - t0
    stats["releves"] = [releve(gr, acts, res, P.max_manches, {"Type": "autojeu"}, draft)
                        for gr, acts, res, draft in r["releves"]]
    return _donnees(r), stats


def _fusionner(xs: list[dict]) -> dict:
    """Plusieurs lots d'entrées en un seul (actions complétées jusqu'au plus grand nombre)."""
    a_len = max(x["acts"].shape[1] for x in xs)
    out = {}
    for k in xs[0]:
        if k == "acts":
            parts = []
            for x in xs:
                v = x[k]
                p = np.zeros((len(v), a_len, v.shape[2]), v.dtype)
                p[..., 4:] = NONE_CELL
                p[:, :v.shape[1]] = v
                parts.append(p)
            out[k] = np.concatenate(parts)
        else:
            out[k] = np.concatenate([x[k] for x in xs])
    return out


def jouer_match_rs(evaluateurs: list, agents: list[dict], parties: list[tuple[int, int, int]],
                   draft: bool = False, max_manches: int = 150, simultanees: int = 512,
                   releves: int = 0, noms: list[str] | None = None, arret=None,
                   fils: int | None = None) -> dict:
    """Parties entre agents à recherche neuronale avec le moteur Rust.

    `parties` : (graine, agent du camp 0, agent du camp 1) ; `agents[i]` : réglages de recherche
    de l'agent i (simulations, m, c_scale, coup_gagnant, cle_publique…) ; `evaluateurs[i]` : son
    évaluateur par lots (`evaluer_lot`) — un même objet pour plusieurs agents reçoit un seul lot.
    Les parties sont lancées au fil de l'eau, `simultanees` à la fois, et avancées par `fils`
    fils d'exécution (défaut : cœurs physiques − 1 ; le résultat n'en dépend pas). `arret(terminees)` est
    appelé quand de nouvelles parties sont terminées : s'il renvoie vrai, les parties en cours
    sont abandonnées. Renvoie les parties terminées (graine, agent 0, agent 1, score du camp 0
    (1, 0, -1), manches), les évaluations de chaque agent, la durée et les relevés .nch.
    """
    t0 = time.time()
    if fils is None:
        from .entrainement import travailleurs_auto
        fils = travailleurs_auto()
    m = champ_rs.Match([(int(g), int(a), int(b)) for g, a, b in parties], agents, draft, max_manches,
                       simultanees, releves, fils)
    finies: list[tuple] = []
    textes: list[str] = []
    noms = noms or [f"agent{i}" for i in range(len(agents))]

    def recolter() -> bool:
        nouvelles = m.resultats()
        for graine, a0, a1, s0, manches, _, res, acts in nouvelles:
            finies.append((graine, a0, a1, s0, manches))
            if acts is not None:
                textes.append(releve(graine, acts, res, max_manches,
                                     {"Type": "evaluation", "Blanc": noms[a0], "Noir": noms[a1]}, draft))
        return bool(nouvelles) and arret is not None and arret(finies)

    lots = m.etape()
    while lots is not None:
        if recolter():
            break
        groupes: dict[int, list[int]] = {}
        for a, lot in enumerate(lots):
            if lot is not None:
                groupes.setdefault(id(evaluateurs[a]), []).append(a)
        reps: list = [None] * len(lots)
        for ids in groupes.values():
            xs = [lot_numpy(lots[a]) for a in ids]
            logits, valeurs = evaluateurs[ids[0]].evaluer_lot(xs[0] if len(xs) == 1 else _fusionner(xs))
            pos = 0
            for a, x in zip(ids, xs):
                n, al = len(x["acts"]), lots[a]["a_len"]
                reps[a] = (np.ascontiguousarray(logits[pos:pos + n, :al], np.float32).tobytes(), al,
                           np.ascontiguousarray(valeurs[pos:pos + n], np.float32).tobytes())
                pos += n
        lots = m.etape(reps)
    else:
        recolter()
    return {"parties": finies, "evaluations": m.evaluations(), "secondes": time.time() - t0,
            "releves": textes}


def _conduire(r, evaluateur, lot, arret=None, compte: list | None = None) -> bool:
    """Évalue les lots d'une recherche `champ_rs.RechercheJeu` jusqu'à la fin de sa passe.
    `arret()` vrai : la passe est interrompue (résultat calculé sur ce qui a été cherché) ;
    renvoie vrai dans ce cas. `compte[0]` cumule les positions évaluées."""
    while lot is not None:
        logits, valeurs = evaluateur.evaluer_lot(lot_numpy(lot))
        reps = (np.ascontiguousarray(logits, np.float32).tobytes(), lot["a_len"],
                np.ascontiguousarray(valeurs, np.float32).tobytes())
        if compte is not None:
            compte[0] += len(valeurs)
        if arret is not None and arret():
            r.interrompre(*reps)
            return True
        lot = r.etape(*reps)
    return False


def _resultat(r, g: Game):
    from .recherche import Resultat
    action, legal, politique, visites, q, valeur, v_reseau, sims = r.resultat()
    legal = [action_python(a) for a in legal]
    res = Resultat(action=action_python(action), legal=legal, politique=np.asarray(politique, np.float32),
                   visites=np.asarray(visites), q=np.asarray(q), logits=np.zeros(len(legal), np.float32),
                   v_reseau=v_reseau, valeur=valeur, simulations=sims)
    if res.action not in g.legal_actions():     # garde-fou : les deux moteurs doivent concorder
        raise RuntimeError(f"coup du moteur Rust illégal dans le moteur Python : {res.action}")
    return res


def recherche_jeu(g: Game, simulations: int, agent: dict | None = None, graine: int = 0):
    """Recherche du moteur Rust (`champ_rs.RechercheJeu`) sur une position du moteur Python."""
    j = champ_rs.Jeu.depuis_etat(etat_jeu(g), g.max_rounds, graine)
    return champ_rs.RechercheJeu(j, {"simulations": simulations, **(agent or {})}, graine)


def rechercher(g: Game, evaluateur, simulations: int = 400, agent: dict | None = None, graine: int = 0):
    """Recherche du moteur Rust sur une position du moteur Python (bot de jeu) : la partie est
    reconstruite dans le moteur Rust (`etat_jeu`), cherchée, et le résultat est renvoyé comme
    celui de la recherche Python (`recherche.Resultat`). `agent` : réglages de recherche
    (parallele, m, c_scale, coup_gagnant…), par défaut ceux du bot (meilleur coup, sans bruit)."""
    r = recherche_jeu(g, simulations, agent, graine)
    _conduire(r, evaluateur, r.etape())
    return _resultat(r, g)


def rechercher_temps(g: Game, evaluateur, secondes: float, agent: dict | None = None, graine: int = 0,
                     depart: int = 256):
    """Recherche progressive du moteur Rust au temps (comme `RechercheGumbel.approfondir`) :
    passes de halving séquentiel de budget doublé (256, 512…) sur les meilleurs candidats du
    moment, l'arbre étant conservé, tant que la passe suivante (≈ 2 × la précédente) tient dans
    le temps restant."""
    t0 = time.time()
    k = len(g.legal_actions())
    budget = depart
    r = recherche_jeu(g, budget, agent, graine)
    lot = r.etape()
    passes = 0
    while True:
        t = time.time()
        _conduire(r, evaluateur, lot)
        passes += 1
        dt = time.time() - t
        if time.time() - t0 + 2.1 * dt > secondes:
            break
        budget *= 2
        lot = r.prolonger(budget, min(k, max(32, budget // 16)))
        if lot is None:
            break
    res = _resultat(r, g)
    res.infos.update(secondes=round(time.time() - t0, 2), passes=passes)
    return res
