"""Boucle d'apprentissage par renforcement de type AlphaZero (Gumbel).

Chaque itération :
  1. **Auto-jeu** — plusieurs processus jouent des parties avec la dernière
     version du réseau (ou l'heuristique pendant l'amorçage) ;
  2. **Apprentissage** — le réseau est entraîné sur une fenêtre glissante des
     exemples récents (politique : entropie croisée vers π' ; valeur :
     entropie croisée V/N/D vers le résultat final, éventuellement mêlé à la
     valeur de recherche) ;
  3. **Évaluation** périodique — matchs appariés contre le bot glouton et la
     version précédente, classement Elo Bradley-Terry, sauvegarde du meilleur.

Tout est reprenable : relancer la même commande reprend là où l'entraînement
s'était arrêté (modèle, optimiseur, fenêtre d'exemples, classement).

Avec `iterations: 0`, l'entraînement ne s'arrête jamais (mode continu) : le taux
d'apprentissage décroît sur `lr_horizon` itérations puis reste à `lr_min`, et
seuls les `modeles_gardes` derniers modèles évalués sont conservés.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import shutil
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

import numpy as np

from .autojeu import ParamsAutoJeu, concatener, jouer
from .direct import ecrire_phase, preparer_tables
from .encodage import PENDING_ID
from .modele import ConfigModele


@dataclass
class ConfigEntrainement:
    dossier: str = "runs/principal"
    iterations: int = 300            # 0 = sans fin (mode continu)
    travailleurs: int = -1           # -1 = automatique (cœurs physiques - 1), 0 = sans processus
    parties_par_iteration: int = 256
    autojeu: ParamsAutoJeu = field(default_factory=ParamsAutoJeu)
    amorce_iterations: int = 2
    amorce_simulations: int = 32
    amorce_pas: int = 1000           # pas supervisés supplémentaires à la fin de l'amorçage
    amorce_melange_q: float = 0.8    # part de la valeur de recherche dans la cible pendant l'amorçage
    modele: ConfigModele = field(default_factory=ConfigModele)
    lot: int = 512
    lr: float = 1e-3
    lr_min: float = 5e-5
    lr_horizon: int = 0              # itérations de décroissance cosinus (0 = iterations)
    echauffement: int = 300
    poids_decroissance: float = 1e-4
    fenetre: int = 500_000
    fenetre_min: int = 20_000
    reutilisation: float = 4.0
    poids_valeur: float = 1.0
    melange_q: float = 0.25
    # > 0 : la valeur de recherche mêlée à la cible de valeur est remplacée par le retour TD(λ)
    # de la partie (moyenne des valeurs de recherche des positions suivantes, puis du résultat ;
    # voir ia/cibles.py) : cible bien moins bruitée que le résultat final seul
    valeur_lambda: float = 0.0
    # mise en place avancée : poids des exemples de draft dans la perte de politique, part de la
    # valeur de recherche dans leur cible de valeur (le résultat final y est très bruité)
    poids_draft: float = 0.5
    melange_q_draft: float = 0.5
    # têtes auxiliaires (0 = désactivée) : contrôle final des Lieux, marge finale, main adverse
    poids_aux: dict = field(default_factory=lambda: {"lieux": 0.15, "marge": 0.1, "main_adverse": 0.1})
    ema: float = 0.0                 # > 0 : moyenne mobile des poids publiée (dernier.pt, meilleur.pt)
    # > 0 : `reinit_tous` itérations après la précédente (état `derniere_reinit`, 0 au départ),
    # un réseau neuf est entraîné depuis zéro sur la
    # fenêtre d'exemples (`reinit_reutilisation` passages, échauffement puis décroissance cosinus
    # de `lr` à `lr_min`) et remplace l'ancien : un réseau entraîné en continu perd de sa
    # plasticité (mesuré : log-loss de valeur 0,73 contre 0,58 pour un réseau neuf, mêmes données)
    reinit_tous: int = 0
    reinit_reutilisation: float = 4.0
    eval_protocole: str = "aleatoire"  # "draft" : les matchs d'évaluation commencent par le draft
    # valeur dynamique des unités (ia/valeurs.py) et recalibrage de l'échelle du scoreur
    unites: dict = field(default_factory=lambda: {"actif": False, "pools": 256, "pioches": 2,
                                                  "bootstrap": 30, "ridge": 1.0})
    dispositif: str = "auto"
    dispositif_autojeu: str = "auto"
    moteur_autojeu: str = "auto"     # auto (Rust si compilé), rust ou python
    compiler: bool = False
    eval_tous: int = 5
    eval_paires: int = 24
    eval_simulations: int = 64
    eval_ancres: list = field(default_factory=lambda: ["glouton"])   # ex. "heur:64", "mcts:400",
    #   ou un modèle figé « chemin.pt » (hors du dossier modeles/, dont les anciens fichiers sont purgés)
    eval_meilleur: bool = True       # affronter aussi le meilleur modèle (s'il n'est pas le précédent)
    eval_precedent: bool = True      # affronter la version évaluée précédente
    # paires des matchs contre un réseau (moteur Rust, bien plus rapide que les matchs contre les bots)
    eval_paires_reseaux: int = 100
    # promotion de meilleur.pt : "elo" (meilleur Elo Bradley-Terry, sans marge) ou "ic" (le nouveau
    # modèle bat le meilleur, borne basse de l'intervalle de confiance à 95 % de l'écart > 0)
    eval_promotion: str = "elo"
    inference_compilee: bool = False # réseau compilé (torch.compile) pour l'auto-jeu et l'évaluation
    # réseau qui joue l'auto-jeu : "dernier" (le plus récent, comme AlphaZero) ou "meilleur" (le
    # meilleur démontré, comme AlphaGo Zero : avec eval_promotion "ic", un réseau n'est promu que
    # si sa supériorité est établie ; l'apprentissage continue sur le dernier réseau)
    autojeu_reseau: str = "dernier"
    travailleurs_evaluation: int = 0 # processus pour les évaluations (0 = travailleurs)
    purger_donnees: bool = True      # supprimer les fichiers d'exemples sortis de la fenêtre
    purger_modeles: bool = True      # ne garder que les modèles évalués (tous les eval_tous)
    modeles_gardes: int = 0          # > 0 : nombre maximal de modèles évalués conservés
    releves_autojeu: int = 4         # parties d'auto-jeu enregistrées en .nch par itération
    releves_evaluation: int = 2      # parties enregistrées par adversaire d'évaluation
    releves_gardes: int = 400        # relevés conservés dans parties/ (les plus récents)
    tensorboard: bool = True         # journal TensorBoard si le paquet est installé
    direct: bool = True              # diffusion des parties en cours pour le spectateur web (direct/)
    graine: int = 1

    # ------------------------------------------------------------ chargement
    @classmethod
    def depuis_dict(cls, d: dict) -> "ConfigEntrainement":
        cfg = cls()
        for k, v in d.items():
            if not k.startswith("_"):      # clés « _commentaire » : notes libres
                _affecter(cfg, k, v)
        return cfg

    def appliquer(self, reglages: list[str]) -> None:
        """Applique des réglages « clé=valeur » (clés pointées : autojeu.simulations=128)."""
        for r in reglages:
            k, _, v = r.partition("=")
            _affecter(self, k.strip(), json.loads(v) if _json_ok(v) else v)


def _json_ok(v: str) -> bool:
    try:
        json.loads(v)
        return True
    except ValueError:
        return False


def _affecter(obj, cle: str, valeur) -> None:
    tete, _, reste = cle.partition(".")
    if not any(f.name == tete for f in fields(obj)):
        raise KeyError(f"Réglage inconnu : {cle}")
    cur = getattr(obj, tete)
    if reste:
        _affecter(cur, reste, valeur)
    elif isinstance(cur, list) and isinstance(valeur, str):
        setattr(obj, tete, [v.strip() for v in valeur.split(",") if v.strip()])
    elif is_dataclass(cur) and isinstance(valeur, dict):
        for k, v in valeur.items():
            _affecter(cur, k, v)
    else:
        setattr(obj, tete, type(cur)(valeur) if cur is not None and not isinstance(valeur, type(cur)) else valeur)


def coeurs_physiques() -> int:
    try:
        with open("/proc/cpuinfo", encoding="ascii", errors="ignore") as f:
            ids = {(p, c) for p, c in _paires_coeurs(f)}
        if ids:
            return len(ids)
    except OSError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def _paires_coeurs(lignes):
    paquet = "0"
    for l in lignes:
        if l.startswith("physical id"):
            paquet = l.split(":")[1].strip()
        elif l.startswith("core id"):
            yield paquet, l.split(":")[1].strip()


def travailleurs_auto() -> int:
    """Cœurs physiques - 1 : le processus principal et le système gardent un peu de marge."""
    return max(1, coeurs_physiques() - 1)


def ecrire_atomique(chemin: Path, texte: str) -> None:
    """Écrit via un fichier temporaire : un arrêt brutal ne laisse jamais un fichier tronqué."""
    tmp = chemin.with_suffix(chemin.suffix + ".tmp")
    tmp.write_text(texte, encoding="utf-8")
    os.replace(tmp, chemin)


def choisir_dispositif(d: str) -> str:
    import torch
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


# ---------------------------------------------------------------- travailleur
_EV_CACHE: dict = {}


def _init_travailleur() -> None:
    import threading

    import torch
    torch.set_num_threads(1)
    parent = os.getppid()

    def surveiller():   # le processus s'arrête si l'entraîneur principal disparaît (kill -9…)
        while True:
            time.sleep(5)
            if os.getppid() != parent:
                os._exit(1)

    threading.Thread(target=surveiller, daemon=True).start()


def _travailleur(args):
    """Joue un paquet de parties. Le réseau n'est rechargé que s'il a changé sur disque, et en
    place (le réseau compilé est conservé d'une itération à l'autre)."""
    chemin, dispositif, params, seed, moteur, *reste = args
    compiler = bool(reste and reste[0])
    from .evaluateurs import EvaluateurHeuristique, EvaluateurReseau
    if chemin is None:
        ev = EvaluateurHeuristique()
    else:
        cle = (dispositif, compiler)
        date = os.path.getmtime(chemin)
        e = _EV_CACHE.get(cle)
        if e is None or (e[0], e[1]) != (chemin, date):
            if e is not None and e[2].recharger(chemin):
                e[0], e[1] = chemin, date
            else:
                _EV_CACHE.clear()
                _EV_CACHE[cle] = [chemin, date, EvaluateurReseau.depuis_fichier(chemin, dispositif,
                                                                                compiler=compiler)]
        ev = _EV_CACHE[cle][2]
    return jouer(ev, params, seed, moteur)


# -------------------------------------------------------------- entraîneur
class Entraineur:
    def __init__(self, cfg: ConfigEntrainement):
        import torch
        self.torch = torch
        self.cfg = cfg
        self.dir = Path(cfg.dossier)
        (self.dir / "modeles").mkdir(parents=True, exist_ok=True)
        (self.dir / "donnees").mkdir(parents=True, exist_ok=True)
        (self.dir / "parties").mkdir(parents=True, exist_ok=True)
        if cfg.travailleurs < 0:
            cfg.travailleurs = travailleurs_auto()
        self.dev = torch.device(choisir_dispositif(cfg.dispositif))
        self.dev_auto = choisir_dispositif(cfg.dispositif_autojeu)
        if self.dev.type == "cuda":
            from .evaluateurs import activer_tf32
            activer_tf32()
        torch.manual_seed(cfg.graine)
        from .modele import ReseauChamp, nb_parametres
        self.etat = {"iteration": 0, "pas": 0, "dernier_evalue": None}
        etat_path = self.dir / "etat.json"
        if etat_path.exists():
            self.etat = json.loads(etat_path.read_text())
            print(f"Reprise de {self.dir} à l'itération {self.etat['iteration']}")
        ck = None
        if (self.dir / "modeles" / "dernier.pt").exists():
            ck = torch.load(self.dir / "modeles" / "dernier.pt", map_location="cpu", weights_only=False)
            cfg.modele = ConfigModele.depuis(ck["config"])   # l'architecture du modèle fait foi
        ecrire_atomique(self.dir / "config.json", json.dumps(asdict(cfg), indent=1, ensure_ascii=False))
        self.model = ReseauChamp(cfg.modele).to(self.dev)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.poids_decroissance, betas=(0.9, 0.99))
        self.ema = None
        if cfg.ema > 0:
            import copy
            self.ema = copy.deepcopy(self.model).requires_grad_(False)
        if ck is not None:
            if self.ema is not None:
                self.ema.load_state_dict(ck["etat"])       # dernier.pt publie les poids moyennés
            brut = self.dir / "modeles" / "brut.pt"
            if self.ema is not None and brut.exists():
                self.model.load_state_dict(torch.load(brut, map_location="cpu", weights_only=False)["etat"])
            else:
                self.model.load_state_dict(ck["etat"])
            if (self.dir / "optim.pt").exists():
                self.opt.load_state_dict(torch.load(self.dir / "optim.pt", map_location=self.dev))
        self.fwd = torch.compile(self.model, dynamic=True) if cfg.compiler else self.model
        self.pool = None
        self.tb = None
        if cfg.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(str(self.dir / "tensorboard"))
            except Exception:  # noqa: BLE001 — TensorBoard est facultatif
                self.tb = None
        print(f"Réseau : {nb_parametres(self.model) / 1e6:.2f} M paramètres — apprentissage sur "
              f"{self.dev}, auto-jeu sur {self.dev_auto}, {cfg.travailleurs} processus")
        self.fenetre: list[dict] = []
        self._charger_fenetre()

    # ---------------------------------------------------------- données
    def _charger_fenetre(self) -> None:
        shards = sorted((self.dir / "donnees").glob("iter_*.npz"), reverse=True)
        total = 0
        for s in shards:
            if total >= self.cfg.fenetre:
                break
            d = self._preparer(dict(np.load(s)))
            self.fenetre.insert(0, d)
            total += len(d["z"])
        if total:
            print(f"Fenêtre rechargée : {total} exemples ({len(self.fenetre)} fichiers)")

    def _preparer(self, d: dict) -> dict:
        """Exemples d'un fichier prêts pour l'apprentissage : retours TD(λ) si demandés ; la
        colonne `debut` (absente des anciens fichiers) ne sert qu'à les calculer."""
        if self.cfg.valeur_lambda > 0 and len(d.get("z", [])):
            from .cibles import retours_lambda
            d["g"] = retours_lambda(d, self.cfg.valeur_lambda)
        d.pop("debut", None)
        return d

    def _ajouter(self, data: dict) -> None:
        self.fenetre.append(self._preparer(data))
        while sum(len(d["z"]) for d in self.fenetre) - len(self.fenetre[0]["z"]) >= self.cfg.fenetre:
            self.fenetre.pop(0)
        if self.cfg.purger_donnees:
            fichiers = sorted((self.dir / "donnees").glob("iter_*.npz"))
            for f in fichiers[: max(0, len(fichiers) - len(self.fenetre))]:
                f.unlink(missing_ok=True)

    def taille_fenetre(self) -> int:
        return sum(len(d["z"]) for d in self.fenetre)

    # ---------------------------------------------------------- auto-jeu
    def autojeu(self, it: int) -> tuple[dict, dict]:
        cfg = self.cfg
        amorce = it <= cfg.amorce_iterations
        params = ParamsAutoJeu(**asdict(cfg.autojeu))
        chemin = None
        if amorce:
            # l'heuristique est peu coûteuse : tous les coups reçoivent une cible
            params.simulations = cfg.amorce_simulations
            params.p_complete = 1.0
            params.p_draft = 0.0     # l'heuristique ne sait pas évaluer une composition d'armée
        else:
            chemin = str(self.dir / "modeles" / "dernier.pt")
        meilleur = (cfg.autojeu_reseau == "meilleur" and not amorce and self.etat.get("meilleur")
                    and (self.dir / "modeles" / "meilleur.pt").exists())
        if meilleur:
            chemin = str(self.dir / "modeles" / "meilleur.pt")
        n_w = max(1, cfg.travailleurs)
        per = [cfg.parties_par_iteration // n_w + (1 if i < cfg.parties_par_iteration % n_w else 0)
               for i in range(n_w)]
        joueur = "heuristique" if amorce else (self.etat["meilleur"] if meilleur else f"iter_{it - 1:04d}")
        if cfg.direct:   # spectateur web : phase et une partie en cours par processus (direct/)
            ecrire_phase(self.dir, "autojeu", it, joueur=joueur, amorce=bool(amorce))
            tables = preparer_tables(self.dir, n_w)
        taches = []
        for i, n in enumerate(per):
            if n == 0:
                continue
            p = ParamsAutoJeu(**asdict(params))
            p.parties = n
            p.releves = cfg.releves_autojeu // n_w + (1 if i < cfg.releves_autojeu % n_w else 0)
            # au moins ~3 vagues de parties par processus : sinon la fin de l'itération tourne
            # avec des lots presque vides (GPU sous-utilisé, processus qui attendent) ; inutile en
            # auto-jeu continu (les parties en cours passent à l'itération suivante)
            if amorce or not p.continu:
                p.continu = False     # l'amorçage heuristique joue avec le moteur Python
                p.simultanees = max(1, min(p.simultanees, max(8, n // 3), n))
            p.direct = tables[i] if cfg.direct else ""
            taches.append((chemin, self.dev_auto, p, cfg.graine * 1_000_003 + it * 1009 + i,
                           cfg.moteur_autojeu, cfg.inference_compilee))
        if cfg.travailleurs <= 0:
            resultats = [_travailleur(t) for t in taches]
        else:
            if self.pool is None:   # processus persistants : CUDA n'est initialisé qu'une fois
                # l'évaluation (moteur Python, limitée par le CPU) peut utiliser plus de processus
                # que l'auto-jeu (moteur Rust, limité par le GPU)
                n_pool = max(cfg.travailleurs, cfg.travailleurs_evaluation)
                self.pool = ProcessPoolExecutor(n_pool, mp_context=mp.get_context("spawn"),
                                                initializer=_init_travailleur)
            resultats = list(self.pool.map(_travailleur, taches))
        data = concatener([r[0] for r in resultats])
        stats: dict = {}
        releves = []
        for _, s in resultats:
            releves += s.pop("releves", [])
            for k, v in s.items():
                stats[k] = stats.get(k, 0) + v
        for k, texte in enumerate(releves):
            self.ecrire_releve(f"{it:05d}-autojeu-{k + 1}", texte,
                               {"Iteration": it, "Blanc": joueur, "Noir": joueur})
        stats["amorce"] = bool(amorce)
        stats["secondes"] = max(s["secondes"] for _, s in resultats)
        stats["parties_par_heure"] = round(3600 * stats["parties"] / max(stats["secondes"], 1e-6))
        stats["evaluations_par_s"] = round(stats["evaluations"] / max(stats["secondes"], 1e-6))
        return data, stats

    def ecrire_releve(self, nom: str, texte: str, entetes: dict) -> None:
        """Relevé d'une partie pour la visualisation web (parties/), les plus récents seulement."""
        from ..notation import parse_headers
        deja = parse_headers(texte)
        lignes = "".join(f'[{k} "{v}"]\n' for k, v in entetes.items() if k not in deja)
        ecrire_atomique(self.dir / "parties" / f"{nom}.nch", lignes + texte)
        fichiers = sorted((self.dir / "parties").glob("*.nch"))
        for f in fichiers[: max(0, len(fichiers) - self.cfg.releves_gardes)]:
            f.unlink(missing_ok=True)

    # ---------------------------------------------------------- apprentissage
    def _lot(self, rng: np.random.Generator) -> tuple[dict, "object"]:
        torch = self.torch
        tailles = np.array([len(d["z"]) for d in self.fenetre])
        choix = rng.choice(len(tailles), size=self.cfg.lot, p=tailles / tailles.sum())
        parts = []
        for k in np.unique(choix):
            idx = rng.integers(0, tailles[k], size=int((choix == k).sum()))
            parts.append({key: v[idx] for key, v in self.fenetre[k].items()})
        b = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        return self._tenseurs(b)

    def _tenseurs(self, b: dict) -> tuple[dict, dict]:
        torch = self.torch
        a_len = int(b["n_act"].max())
        x = {}
        for k in ("cell_i", "unit_i", "glob_i"):
            x[k] = torch.from_numpy(b[k].astype(np.int64))
        for k in ("cell_f", "unit_f", "glob_f"):
            x[k] = torch.from_numpy(b[k].astype(np.float32))
        x["acts"] = torch.from_numpy(b["acts"][:, :a_len].astype(np.int64))
        x = {k: v.to(self.dev, non_blocking=True) for k, v in x.items()}
        y = {"pi": torch.from_numpy(b["pi"][:, :a_len].astype(np.float32)).to(self.dev),
             "n_act": torch.from_numpy(b["n_act"].astype(np.int64)).to(self.dev),
             "z": torch.from_numpy(b["z"]).to(self.dev),
             "q": torch.from_numpy(b["q"]).to(self.dev)}
        if "g" in b:
            y["g"] = torch.from_numpy(b["g"]).to(self.dev)
        for k in ("lieux", "marge", "main_adv"):
            if k in b:
                y[k] = torch.from_numpy(b[k].astype(np.int64)).to(self.dev)
        y["draft"] = x["glob_i"][:, 0] == PENDING_ID["draft"]
        return x, y

    def _pertes_aux(self, aux: dict, y: dict) -> dict:
        """Pertes des têtes auxiliaires (entropies croisées), absentes si les cibles manquent."""
        F = self.torch.nn.functional
        out = {}
        if "lieux" in y:
            t = y["lieux"]
            out["lieux"] = F.cross_entropy(aux["lieux"].float().reshape(-1, 3), t.reshape(-1).clamp_min(0),
                                           reduction="none").reshape(t.shape)[t >= 0].mean()
        if "marge" in y:
            out["marge"] = F.cross_entropy(aux["marge"].float(), y["marge"] + 4)
        if "main_adv" in y:
            jeu = ~y["draft"]
            if jeu.any():
                m = aux["main"][jeu].float()
                out["main_adverse"] = F.cross_entropy(m.reshape(-1, m.shape[-1]),
                                                      y["main_adv"][jeu].clamp(0, m.shape[-1] - 1).reshape(-1))
        return out

    def valider(self, data: dict, maximum: int = 4096) -> dict:
        """Mesures sur des parties que le réseau n'a jamais vues (avant d'apprendre dessus).

        Mesures du modèle publié (poids moyennés EMA s'il y en a : c'est lui qui joue), et
        mesures de valeur des poids bruts (suffixe `_brut`).

        Précision de valeur : signe correct du résultat final. Log-loss de valeur : entropie
        croisée V/N/D contre le résultat (l'étalonnage, plus pertinent que la précision dans un
        jeu où le hasard domine en début de partie) ; `logloss_q` est celle de la valeur de la
        recherche, `logloss_base` celle d'une prédiction constante (fréquences des résultats).
        Par tranche de manche (`_m1_3`, `_m4_9`, `_m10`). Premier coup : le coup préféré du
        réseau est aussi celui de la cible π'. KL : écart entre politique et cible.
        """
        n = len(data.get("z", []))
        if n == 0:
            return {}
        idx = np.random.default_rng(0).permutation(n)[:maximum]
        res = self._mesurer(self.ema if self.ema is not None else self.model, data, idx)
        if self.ema is not None:
            brut = self._mesurer(self.model, data, idx, valeur_seule=True)
            res.update({f"{k}_brut": v for k, v in brut.items()})
        return res

    def _mesurer(self, model, data: dict, idx: np.ndarray, valeur_seule: bool = False) -> dict:
        torch = self.torch
        out = {"kl": 0.0, "premier_coup": 0.0, "precision_valeur": 0.0, "erreur_valeur": 0.0,
               "logloss_valeur": 0.0, "logloss_q": 0.0}
        dr = {"kl_draft": 0.0, "premier_coup_draft": 0.0}
        tranches = {"m1_3": (0, 3), "m4_9": (4, 9), "m10": (10, 10**9)}
        par_tranche = {t: {"n": 0, "precision_valeur": 0.0, "logloss_valeur": 0.0, "logloss_q": 0.0}
                       for t in tranches}
        tot = n_draft = 0
        classes = torch.zeros(3, device=self.dev)
        etait = model.training
        model.eval()
        with torch.inference_mode():
            for s in range(0, len(idx), 512):
                x, y = self._tenseurs({k: v[idx[s:s + 512]] for k, v in data.items()})
                logits, vlog = model(x)
                logits, vlog = logits.float(), vlog.float()
                z = y["z"]
                cz = torch.where(z > 0, 0, torch.where(z < 0, 2, 1))       # classe V/N/D
                ll = torch.nn.functional.cross_entropy(vlog, cz, reduction="none")
                # q n'a pas de probabilité de nulle : 1 % fixe (nulles quasi absentes en auto-jeu)
                q = y["q"].float()
                pq = torch.stack([0.99 * (1 + q) / 2, torch.full_like(q, 0.01), 0.99 * (1 - q) / 2], -1)
                llq = -torch.log(pq.gather(1, cz[:, None])[:, 0].clamp(1e-3, 1.0))
                classes += torch.bincount(cz, minlength=3).float()
                out["logloss_valeur"] += ll.sum().item()
                out["logloss_q"] += llq.sum().item()
                mask = torch.arange(logits.shape[1], device=self.dev)[None] < y["n_act"][:, None]
                logp = torch.log_softmax(logits.masked_fill(~mask, -1e9), -1)
                pi = y["pi"] / y["pi"].sum(-1, keepdim=True).clamp_min(1e-8)
                kl = (pi * (torch.log(pi.clamp_min(1e-9)) - logp).masked_fill(~mask, 0.0)).sum(-1)
                top = (logits.masked_fill(~mask, -1e9).argmax(-1) == pi.argmax(-1)).float()
                pv = torch.softmax(vlog, -1)
                v = pv[:, 0] - pv[:, 2]
                prec = ((v.sign() == z.sign()) | ((z == 0) & (v.abs() < 0.3))).float()
                m = len(z)
                out["kl"] += kl.sum().item()
                out["premier_coup"] += top.sum().item()
                out["precision_valeur"] += prec.sum().item()
                out["erreur_valeur"] += ((v - z) ** 2).sum().item()
                d = y["draft"]
                dr["kl_draft"] += kl[d].sum().item()
                dr["premier_coup_draft"] += top[d].sum().item()
                n_draft += int(d.sum().item())
                manche = torch.round(x["glob_f"][:, 18] * 50)      # gf[18] = min(manche, 100) / 50
                for t, (lo, hi) in tranches.items():
                    sel = (manche >= lo) & (manche <= hi) & ~d
                    pt = par_tranche[t]
                    pt["n"] += int(sel.sum().item())
                    pt["precision_valeur"] += prec[sel].sum().item()
                    pt["logloss_valeur"] += ll[sel].sum().item()
                    pt["logloss_q"] += llq[sel].sum().item()
                tot += m
        if etait:
            model.train()
        res = {k: round(v / tot, 4) for k, v in out.items()}
        f = classes / classes.sum()
        res["logloss_base"] = round(float(-(f * torch.log(f.clamp_min(1e-9))).sum()), 4)
        for t, pt in par_tranche.items():
            if pt["n"]:
                res.update({f"{k}_{t}": round(v / pt["n"], 4) for k, v in pt.items() if k != "n"})
        if n_draft and not valeur_seule:
            res.update({k: round(v / n_draft, 4) for k, v in dr.items()})
        res["part_draft"] = round(n_draft / tot, 4)
        if valeur_seule:
            res = {k: v for k, v in res.items() if "valeur" in k}
        return res

    def lr_actuel(self, it: int) -> float:
        cfg = self.cfg
        pas = self.etat["pas"]
        if pas < cfg.echauffement:
            return cfg.lr * (pas + 1) / cfg.echauffement
        horizon = cfg.lr_horizon or cfg.iterations
        if horizon <= 0:
            return cfg.lr
        f = min(1.0, it / horizon)
        return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * f))

    def reinitialiser(self, it: int) -> dict:
        """Réseau neuf entraîné depuis zéro sur la fenêtre d'exemples, qui remplace l'ancien
        (poids, moyenne mobile et optimiseur). Taux d'apprentissage : échauffement puis
        décroissance cosinus de `lr` à `lr_min` sur tous les pas."""
        import copy

        from .modele import ReseauChamp
        torch = self.torch
        cfg = self.cfg
        t0 = time.time()
        self.model = ReseauChamp(cfg.modele).to(self.dev)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.poids_decroissance, betas=(0.9, 0.99))
        if self.ema is not None:
            self.ema = copy.deepcopy(self.model).requires_grad_(False)
        self.fwd = torch.compile(self.model, dynamic=True) if cfg.compiler else self.model
        n_pas = max(1, math.ceil(self.taille_fenetre() * cfg.reinit_reutilisation / cfg.lot))
        chauffe = min(cfg.echauffement, n_pas // 10)

        def lr(k: int) -> float:
            if k < chauffe:
                return cfg.lr * (k + 1) / chauffe
            f = (k - chauffe) / max(1, n_pas - chauffe)
            return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * f))
        out = self.apprendre(it, 0, n_pas=n_pas, lr=lr, ema_depuis_zero=True)
        out["secondes"] = round(time.time() - t0, 1)
        return out

    def apprendre(self, it: int, n_nouveaux: int, n_pas: int | None = None, lr=None,
                  ema_depuis_zero: bool = False) -> dict:
        """`lr(k)` : taux d'apprentissage du pas k (par défaut `lr_actuel`) ; `ema_depuis_zero` :
        moyenne mobile avec démarrage progressif compté depuis le premier de ces pas."""
        torch = self.torch
        cfg = self.cfg
        if self.taille_fenetre() < min(cfg.fenetre_min, cfg.lot):
            return {"pas": 0}
        if n_pas is None:
            n_pas = max(1, math.ceil(n_nouveaux * cfg.reutilisation / cfg.lot))
        rng = np.random.default_rng(cfg.graine * 7 + it)
        self.model.train()
        cumul = {"perte_politique": 0.0, "perte_valeur": 0.0, "entropie": 0.0, "precision_valeur": 0.0}
        poids_aux = {k: v for k, v in (cfg.poids_aux or {}).items() if v}
        amp = self.dev.type == "cuda"
        # bf16 sur GPU récents (Ampere+), sinon fp16 avec mise à l'échelle des gradients
        bf16 = amp and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16 else torch.float16
        if not hasattr(self, "scaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=amp and not bf16)
        for i_pas in range(n_pas):
            for grp in self.opt.param_groups:
                grp["lr"] = lr(i_pas) if lr is not None else self.lr_actuel(it)
            x, y = self._lot(rng)
            with torch.autocast(self.dev.type, dtype=dtype, enabled=amp):
                logits, vlog, aux = self.fwd(x, aux=True)
            logits, vlog = logits.float(), vlog.float()
            mask = torch.arange(logits.shape[1], device=self.dev)[None] < y["n_act"][:, None]
            logp = torch.log_softmax(logits.masked_fill(~mask, -1e9), dim=-1)
            pi = y["pi"] / y["pi"].sum(-1, keepdim=True).clamp_min(1e-8)
            w = torch.where(y["draft"], cfg.poids_draft, 1.0)
            l_pol = (w * -(pi * logp.masked_fill(~mask, 0.0)).sum(-1)).sum() / w.sum()
            z = y["z"]
            cible = torch.stack([(z > 0).float(), (z == 0).float(), (z < 0).float()], -1)
            mq = cfg.amorce_melange_q if it <= cfg.amorce_iterations else cfg.melange_q
            mq = torch.where(y["draft"], max(mq, cfg.melange_q_draft), mq).unsqueeze(-1)
            q = y.get("g", y["q"]).clamp(-1, 1)
            cq = torch.stack([(1 + q) / 2, torch.zeros_like(q), (1 - q) / 2], -1)
            cible = (1 - mq) * cible + mq * cq
            l_val = -(cible * torch.log_softmax(vlog, -1)).sum(-1).mean()
            perte = l_pol + cfg.poids_valeur * l_val
            pertes_aux = self._pertes_aux(aux, y) if poids_aux else {}
            for k, l in pertes_aux.items():
                perte = perte + poids_aux.get(k, 0.0) * l
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(perte).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.etat["pas"] += 1
            if self.ema is not None:
                with torch.no_grad():   # moyenne mobile, avec démarrage progressif
                    n = i_pas if ema_depuis_zero else self.etat["pas"]
                    dec = min(cfg.ema, (1 + n) / (10 + n))
                    for pe, pm in zip(self.ema.parameters(), self.model.parameters()):
                        pe.lerp_(pm, 1 - dec)
            with torch.no_grad():
                p = torch.softmax(logits.masked_fill(~mask, -1e9), -1)
                ent = -(p * logp.masked_fill(~mask, 0.0)).sum(-1).mean()
                pv = torch.softmax(vlog, -1)
                v = pv[:, 0] - pv[:, 2]
                prec = ((v.sign() == z.sign()) | ((z == 0) & (v.abs() < 0.3))).float().mean()
            cumul["perte_politique"] += l_pol.item()
            cumul["perte_valeur"] += l_val.item()
            cumul["entropie"] += ent.item()
            cumul["precision_valeur"] += prec.item()
            for k, l in pertes_aux.items():
                cumul[f"perte_{k}"] = cumul.get(f"perte_{k}", 0.0) + l.item()
        self.model.eval()
        out = {k: v / n_pas for k, v in cumul.items()}
        out["pas"] = n_pas
        out["lr"] = lr(n_pas - 1) if lr is not None else self.lr_actuel(it)
        return out

    # ---------------------------------------------------------- évaluation
    def evaluer(self, it: int) -> dict:
        """Matchs appariés du nouveau réseau contre les ancres, la version évaluée précédente et
        le meilleur modèle. Le classement Bradley-Terry est réajusté sur tous les résultats : l'Elo
        de chaque modèle déjà évalué est ainsi réévalué à chaque nouvelle évaluation.

        Contre un réseau (modèle précédent, meilleur, ancre figée .pt), le match se joue avec le
        moteur Rust dans ce processus (`eval_paires_reseaux` paires) ; contre un bot, avec le moteur
        Python, réparti sur les processus d'auto-jeu s'il y en a, en même temps.
        """
        from concurrent.futures import ThreadPoolExecutor

        from . import rs
        from .evaluation import (ClassementElo, agent_depuis_spec, match, match_parallele, matchs_contre,
                                 nom_spec, spec_reseau)
        cfg = self.cfg
        nom = f"iter_{it:04d}"
        chemin = str(self.dir / "modeles" / f"{nom}.pt")
        adversaires = [(spec, nom_spec(spec)) for spec in cfg.eval_ancres]
        prec = self.etat.get("dernier_evalue")
        if cfg.eval_precedent and prec and (self.dir / "modeles" / f"{prec}.pt").exists():
            adversaires.append((str(self.dir / "modeles" / f"{prec}.pt"), prec))
        meilleur = self.etat.get("meilleur")
        if (cfg.eval_meilleur or cfg.eval_promotion == "ic") and meilleur and meilleur != nom                 and all(n != meilleur for _, n in adversaires)                 and (self.dir / "modeles" / f"{meilleur}.pt").exists():
            adversaires.append((str(self.dir / "modeles" / f"{meilleur}.pt"), meilleur))
        rust = rs.disponible() and cfg.moteur_autojeu != "python"
        reseaux = [a for a in adversaires if rust and spec_reseau(a[0]) is not None]
        bots = [a for a in adversaires if a not in reseaux]
        classement = ClassementElo(self.dir / "elo.json")
        rapports = {}

        def jouer_bot(adv):
            spec, nom_adv = adv
            if self.pool is not None:
                n_eval = cfg.travailleurs_evaluation or cfg.travailleurs
                return match_parallele(chemin, spec, cfg.eval_paires, 10_000 + it, self.pool,
                                       n_morceaux=max(1, n_eval // max(len(bots), 1)),
                                       simulations=cfg.eval_simulations, dispositif=self.dev_auto,
                                       nom_a=nom, nom_b=nom_adv, releves=cfg.releves_evaluation,
                                       protocole=cfg.eval_protocole)
            a = agent_depuis_spec(chemin, cfg.eval_simulations, str(self.dev), nom)
            b = agent_depuis_spec(spec, cfg.eval_simulations, str(self.dev), nom_adv)
            return match(a, b, paires=cfg.eval_paires, seed=10_000 + it,
                         releves=cfg.releves_evaluation, protocole=cfg.eval_protocole)

        # les matchs contre les bots (moteur Python, limités par le CPU) se jouent en même temps
        # que ceux contre les réseaux (moteur Rust, GPU de ce processus)
        with ThreadPoolExecutor(max(1, len(bots))) as tex:
            futurs = [tex.submit(jouer_bot, adv) for adv in bots] if self.pool is not None else []
            resultats = matchs_contre(
                chemin, [(*spec_reseau(spec), nom_adv) for spec, nom_adv in reseaux],
                cfg.eval_paires_reseaux, 10_000 + it, cfg.eval_simulations, str(self.dev),
                cfg.eval_protocole, cfg.releves_evaluation, cfg.inference_compilee, nom)
            for (spec, nom_adv), f in zip(bots, futurs):
                resultats[nom_adv] = f.result()
            for spec, nom_adv in bots[len(futurs):]:
                resultats[nom_adv] = jouer_bot((spec, nom_adv))
        for spec, nom_adv in adversaires:
            r = resultats[nom_adv]
            court = "".join(c if c.isalnum() else "_" for c in nom_adv)
            for k, texte in enumerate(r.pop("releves", [])):
                self.ecrire_releve(f"{it:05d}-eval-{court}-{k + 1}", texte, {"Iteration": it})
            classement.ajouter(nom, nom_adv, r["victoires"] + 0.5 * r["nulles"], r["parties"])
            rapports[nom_adv] = r
        classement.sauver()
        elo = classement.ajuster()
        self.etat["dernier_evalue"] = nom
        if meilleur is None or meilleur == nom:
            promu = True
        elif cfg.eval_promotion == "ic":
            r = rapports.get(meilleur)
            promu = r is not None and r.get("elo_bas", -1.0) > 0
        else:
            promu = elo.get(nom, -1e9) >= elo.get(meilleur, -1e9)
        if promu:
            self.etat["meilleur"] = nom
            shutil.copy(self.dir / "modeles" / f"{nom}.pt", self.dir / "modeles" / "meilleur.tmp")
            os.replace(self.dir / "modeles" / "meilleur.tmp", self.dir / "modeles" / "meilleur.pt")
        self._purger_evalues()
        cles = ("victoires", "nulles", "defaites", "score", "elo", "elo_bas", "elo_haut", "pentanomial")
        return {"elo": round(elo.get(nom, 0.0), 1), "meilleur": self.etat["meilleur"],
                "matchs": {k: {c: v.get(c) for c in cles} for k, v in rapports.items()}}

    def valeurs_unites(self, it: int, data: dict) -> dict:
        """Valeur dynamique des unités (points du scoreur) avec le réseau publié (dernier.pt)."""
        from ..score import K
        from .evaluateurs import EvaluateurReseau
        from .valeurs import etape
        ev = EvaluateurReseau.depuis_fichier(str(self.dir / "modeles" / "dernier.pt"), str(self.dev))
        try:
            return etape(self.dir, it, ev, self.fenetre, data, self.cfg.unites, K)
        except Exception as e:  # noqa: BLE001 — une mesure ne doit jamais arrêter l'entraînement
            print(f"Valeur des unités : échec ({e})", flush=True)
            return {"erreur": str(e)}

    def _purger_evalues(self) -> None:
        """Mode continu : ne garde que les `modeles_gardes` derniers modèles évalués."""
        n = self.cfg.modeles_gardes
        if n <= 0 or not self.cfg.eval_tous:
            return
        garder = {self.etat.get("meilleur"), self.etat.get("dernier_evalue")}
        evalues = sorted(f for f in (self.dir / "modeles").glob("iter_*.pt")
                         if int(f.stem[5:]) % self.cfg.eval_tous == 0 and f.stem not in garder)
        for f in evalues[: max(0, len(evalues) - n)]:
            f.unlink(missing_ok=True)

    # ---------------------------------------------------------- boucle
    def sauver(self, it: int) -> None:
        from .modele import sauver
        m = self.dir / "modeles"
        sauver(m / "dernier.tmp", self.ema if self.ema is not None else self.model, {"iteration": it})
        os.replace(m / "dernier.tmp", m / "dernier.pt")
        if self.ema is not None:   # poids bruts (reprise de l'apprentissage)
            sauver(m / "brut.tmp", self.model, {"iteration": it})
            os.replace(m / "brut.tmp", m / "brut.pt")
        shutil.copy(m / "dernier.pt", m / f"iter_{it:04d}.pt")
        if self.cfg.purger_modeles and it > 1:
            prec = it - 1
            garde = prec == 0 or (self.cfg.eval_tous and prec % self.cfg.eval_tous == 0)
            if not garde and f"iter_{prec:04d}" not in (self.etat.get("meilleur"), self.etat.get("dernier_evalue")):
                (m / f"iter_{prec:04d}.pt").unlink(missing_ok=True)
        self.torch.save(self.opt.state_dict(), self.dir / "optim.tmp")
        os.replace(self.dir / "optim.tmp", self.dir / "optim.pt")
        ecrire_atomique(self.dir / "etat.json", json.dumps(self.etat, indent=1))

    def executer(self) -> None:
        def arret(signum, frame):   # docker stop envoie SIGTERM : même effet que Ctrl-C
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, arret)
        try:
            self._boucle()
        except KeyboardInterrupt:
            print("\nInterruption : l'itération en cours est abandonnée ; relancez la même "
                  "commande pour reprendre depuis la dernière itération terminée.")
        finally:
            if self.pool is not None:
                self.pool.shutdown(cancel_futures=True)
            if self.tb is not None:
                self.tb.close()

    def _attendre_reprise(self) -> None:
        """Pause demandée depuis l'interface web (controle.json, {"pause": true}) : prise en compte
        entre deux itérations complètes (auto-jeu, apprentissage, évaluation) ; attend la reprise."""
        f = self.dir / "controle.json"

        def en_pause() -> bool:
            try:
                return bool(json.loads(f.read_text(encoding="utf-8")).get("pause"))
            except (OSError, ValueError):
                return False

        if not en_pause():
            return
        it = self.etat["iteration"]
        print(f"Pause demandée : arrêt après l'itération {it} (reprise depuis l'interface web)", flush=True)
        if self.cfg.direct:
            ecrire_phase(self.dir, "pause", it)
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        while en_pause():
            time.sleep(2)
        print("Reprise de l'entraînement", flush=True)

    def _tensorboard(self, l: dict) -> None:
        if self.tb is None:
            return
        it = l["iteration"]
        st, ap = l["autojeu"], l["apprentissage"]
        w = self.tb.add_scalar
        w("autojeu/manches_moyennes", st["manches"] / max(st["parties"], 1), it)
        w("autojeu/taux_nulles", st["nulles"] / max(st["parties"], 1), it)
        w("autojeu/parties_par_heure", st["parties_par_heure"], it)
        w("donnees/exemples", l["exemples"], it)
        w("donnees/fenetre", l["fenetre"], it)
        for k in ("perte_politique", "perte_valeur", "entropie", "precision_valeur", "lr"):
            if k in ap:
                w(f"apprentissage/{k}", ap[k], it)
        for k, v in (l.get("validation") or {}).items():
            w(f"validation/{k}", v, it)
        if "evaluation" in l:
            w("evaluation/elo", l["evaluation"]["elo"], it)
            for adv, r in l["evaluation"]["matchs"].items():
                w(f"evaluation/score_contre_{adv}", r["score"], it)
        self.tb.flush()

    def _boucle(self) -> None:
        cfg = self.cfg
        journal = open(self.dir / "journal.jsonl", "a", encoding="utf-8")
        if not (self.dir / "modeles" / "dernier.pt").exists():
            self.sauver(0)   # réseau initial (aléatoire)
        while cfg.iterations <= 0 or self.etat["iteration"] < cfg.iterations:
            self._attendre_reprise()
            it = self.etat["iteration"] + 1
            t0 = time.time()
            data, st = self.autojeu(it)
            n = len(data.get("z", []))
            if n:
                tmp = self.dir / "donnees" / f"tmp_{it:04d}.npz"   # hors du motif iter_*.npz
                np.savez_compressed(tmp, **data)
                os.replace(tmp, self.dir / "donnees" / f"iter_{it:04d}.npz")
                self._ajouter(data)
            t1 = time.time()
            if cfg.direct:
                ecrire_phase(self.dir, "apprentissage", it)
            val = self.valider(data) if n and not st.get("amorce") else {}
            self.etat["non_appris"] = self.etat.get("non_appris", 0) + n
            if (cfg.reinit_tous and it - self.etat.get("derniere_reinit", 0) >= cfg.reinit_tous
                    and self.taille_fenetre() >= cfg.fenetre_min):
                print(f"Réinitialisation : réseau neuf entraîné sur la fenêtre ({self.taille_fenetre()} "
                      f"exemples)…", flush=True)
                ap = self.reinitialiser(it)
                ap["reinitialisation"] = True
                self.etat["derniere_reinit"] = it
            else:
                ap = self.apprendre(it, self.etat["non_appris"])
            if ap.get("pas"):
                self.etat["non_appris"] = 0
            if it == cfg.amorce_iterations and cfg.amorce_pas > 0 and self.taille_fenetre() >= cfg.lot:
                # pré-entraînement supervisé : le réseau imite d'abord la recherche heuristique
                print(f"Pré-entraînement supervisé sur les données d'amorçage : {cfg.amorce_pas} pas…",
                      flush=True)
                pre = self.apprendre(it, 0, n_pas=cfg.amorce_pas)
                pre["pas"] += ap.get("pas", 0)
                ap = pre
            t2 = time.time()
            self.etat["iteration"] = it
            self.sauver(it)
            ligne = {"iteration": it, "exemples": n, "fenetre": self.taille_fenetre(),
                     "autojeu": st, "apprentissage": ap, "validation": val,
                     "temps": {"autojeu": round(t1 - t0, 1), "apprentissage": round(t2 - t1, 1)}}
            if (cfg.unites or {}).get("actif") and not st.get("amorce"):
                ligne["unites"] = self.valeurs_unites(it, data)
                ligne["temps"]["unites"] = round(time.time() - t2, 1)
                t2 = time.time()
            if cfg.eval_tous and (it % cfg.eval_tous == 0 or it == cfg.iterations) and ap.get("pas"):
                if cfg.direct:
                    ecrire_phase(self.dir, "evaluation", it)
                ligne["evaluation"] = self.evaluer(it)
                ecrire_atomique(self.dir / "etat.json", json.dumps(self.etat, indent=1))
                ligne["temps"]["evaluation"] = round(time.time() - t2, 1)
            journal.write(json.dumps(ligne, ensure_ascii=False) + "\n")
            journal.flush()
            self._tensorboard(ligne)
            print(_resume(ligne), flush=True)
        journal.close()


def _resume(l: dict) -> str:
    st, ap = l["autojeu"], l["apprentissage"]
    s = (f"[it {l['iteration']:4d}] {'amorce ' if st.get('amorce') else ''}"
         f"{st['parties']} parties (B {st['victoires_blanc']} / N {st['victoires_noir']} / "
         f"= {st['nulles']}, {st['manches'] / max(st['parties'], 1):.1f} manches) "
         f"{l['exemples']} ex. | fenêtre {l['fenetre']}")
    if ap.get("pas"):
        s += (f" | {ap['pas']} pas, pol {ap['perte_politique']:.3f} val {ap['perte_valeur']:.3f}"
              f" préc {ap['precision_valeur']:.2f}")
    v = l.get("validation")
    if v:
        s += f"\n           parties inédites : précision valeur {v['precision_valeur']:.2f}, " \
             f"1er coup = cible {v['premier_coup']:.2f}, KL {v['kl']:.3f}"
        if "logloss_valeur" in v:
            s += f", log-loss valeur {v['logloss_valeur']:.3f} (recherche {v['logloss_q']:.3f}, " \
                 f"constante {v['logloss_base']:.3f})"
    t = l["temps"]
    s += f" | {t['autojeu']:.0f}s + {t['apprentissage']:.0f}s"
    if "evaluation" in l:
        e = l["evaluation"]
        m = " ".join(f"{k}:{v['score']:.2f}" for k, v in e["matchs"].items())
        s += f"\n           évaluation : Elo {e['elo']:+.0f} (glouton = 0) | {m} | meilleur = {e['meilleur']}"
    u = l.get("unites")
    if u and "top" in u:
        s += (f"\n           unités : K {u['K']:.2f}, meilleures {' '.join(u['top'])}, "
              f"avantage du 1er choix {u['avantage_premier_choix']:+.1f} pts, R² {u['r2']:.2f}")
    return s


def entrainer(cfg: ConfigEntrainement) -> None:
    Entraineur(cfg).executer()
