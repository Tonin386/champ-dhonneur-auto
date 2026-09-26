"""Évaluateurs de positions pour la recherche.

Un évaluateur reçoit une liste de requêtes (partie, actions légales) et
renvoie pour chacune (logits des actions, valeur) — la valeur est exprimée du
point de vue de l'équipe du joueur au trait, dans [-1, 1].
"""
from __future__ import annotations

import numpy as np

from ..bots.heuristic import score, value
from ..engine import Action, Game


class Evaluateur:
    def evaluer(self, requetes: list[tuple[Game, list[Action]]]) -> list[tuple[np.ndarray, float]]:
        raise NotImplementedError


class EvaluateurUniforme(Evaluateur):
    """Politique uniforme et valeur nulle (utile pour les tests)."""

    def evaluer(self, requetes):
        return [(np.zeros(len(legal), np.float32), 0.0) for _, legal in requetes]


class EvaluateurHeuristique(Evaluateur):
    """Amorçage : a priori = gain heuristique à 1 coup, valeur = heuristique.

    Sert à la génération 0 de l'auto-jeu pour que le réseau démarre d'un
    niveau raisonnable au lieu de parties aléatoires quasi toutes nulles.
    """

    def __init__(self, temperature: float = 5.0):
        self.temperature = temperature

    def evaluer(self, requetes):
        out = []
        for g, legal in requetes:
            team = g.team(g.to_move)
            base = score(g, team)
            logits = np.empty(len(legal), np.float32)
            for i, a in enumerate(legal):
                h = g.copy()
                h.apply(a)
                s = score(h, team)
                logits[i] = max(-10.0, min(10.0, (s - base) / self.temperature))
            out.append((logits, value(g, team)))
        return out


class EvaluateurReseau(Evaluateur):
    """Évaluation par lots avec le réseau PyTorch (CPU ou GPU).

    Sur GPU, les poids sont convertis une fois pour toutes en bf16 (fp16 si le GPU ne gère
    pas bf16) : moins de noyaux lancés qu'avec autocast, ce qui compte pour les petits lots
    de l'auto-jeu. L'évaluateur prend possession du modèle qu'on lui confie.

    `compiler` (GPU) : réseau compilé (`torch.compile`, graphes CUDA). Les lots sont complétés
    jusqu'à des tailles fixes (`PALIERS_LOT` × `PALIERS_ACTIONS`), une compilation par palier ;
    mesuré sur RTX 3070 : +40 % de positions/s, sorties aussi proches du calcul fp32 qu'en
    exécution directe. `recharger` copie de nouveaux poids en place, sans recompiler.
    """

    PALIERS_LOT = (128, 256, 512, 1024, 2048, 4096)
    PALIERS_ACTIONS = (16, 32, 64)
    LOT_MAX = 8192   # au-delà (draft exact de centaines de parties à la fois…), évaluation par morceaux

    def __init__(self, model, device: str = "cpu", demi_precision: bool | None = None,
                 compiler: bool = False):
        import torch

        from .encodage import collate, encode_actions, encode_state
        self.torch = torch
        self._enc = (encode_state, encode_actions, collate)
        self.device = torch.device(device)
        if demi_precision is None:
            demi_precision = self.device.type == "cuda"   # demi-précision seulement sur GPU
        self.dtype = torch.float32
        if self.device.type == "cuda":
            activer_tf32()
            if demi_precision:
                self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = model.to(self.device, dtype=self.dtype).eval()
        self.compile = None
        if compiler and self.device.type == "cuda":
            self.compile = torch.compile(self.model, mode="reduce-overhead", dynamic=False)
        self.n_evals = 0
        self.n_lots = 0

    @classmethod
    def depuis_fichier(cls, path, device: str = "cpu", compiler: bool = False):
        from .modele import charger
        model, _ = charger(path, "cpu")
        return cls(model, device, compiler=compiler)

    def recharger(self, path) -> bool:
        """Copie en place les poids d'un autre fichier de même architecture (graphes compilés
        conservés). Renvoie faux si l'architecture diffère : il faut alors un nouvel évaluateur."""
        from .modele import charger
        model, _ = charger(path, "cpu")
        if model.cfg != self.model.cfg:
            return False
        with self.torch.no_grad():
            self.model.load_state_dict(model.state_dict())
        return True

    def evaluer(self, requetes):
        if not requetes:
            return []
        torch = self.torch
        encode_state, encode_actions, collate = self._enc
        states = [encode_state(g) for g, _ in requetes]
        actions = [encode_actions(g, legal) for g, legal in requetes]
        x, mask = collate(states, actions)
        logits, v = self.evaluer_lot(x)
        return [(logits[i, :len(actions[i])], float(v[i])) for i in range(len(requetes))]

    def evaluer_lot(self, x: dict) -> tuple[np.ndarray, np.ndarray]:
        """Entrées déjà encodées (numpy) -> (logits (B, A) float32, valeurs (B,) dans [-1, 1]).

        Utilisé directement par l'auto-jeu du moteur Rust, qui encode lui-même les positions.
        """
        b, a_len = x["acts"].shape[:2]
        if b > self.LOT_MAX:     # mémoire bornée : les très grands lots sont évalués par morceaux
            n = self.LOT_MAX
            parts = [self.evaluer_lot({k: v[i:i + n] for k, v in x.items()}) for i in range(0, b, n)]
            return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        fwd = self.model
        if self.compile is not None and a_len <= self.PALIERS_ACTIONS[-1]:
            if b > self.PALIERS_LOT[-1]:     # lot trop grand : par morceaux
                n = self.PALIERS_LOT[-1]
                parts = [self.evaluer_lot({k: v[i:i + n] for k, v in x.items()}) for i in range(0, b, n)]
                return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
            x = _completer(x, next(p for p in self.PALIERS_LOT if p >= b),
                           next(p for p in self.PALIERS_ACTIONS if p >= a_len))
            fwd = self.compile
        torch = self.torch
        with torch.inference_mode():
            xt = {}
            for k, v in x.items():
                v = np.ascontiguousarray(v)
                t = torch.from_numpy(v if v.flags.writeable else v.copy())
                if t.is_floating_point():
                    t = t.to(self.dtype)
                xt[k] = t.to(self.device, non_blocking=True)
            logits, vlog = fwd(xt)
            p = torch.softmax(vlog[:b].float(), dim=-1)
            v = (p[:, 0] - p[:, 2]).cpu().numpy()
            logits = logits[:b, :a_len].float().cpu().numpy()
        self.n_evals += len(v)
        self.n_lots += 1
        return logits, v


def _completer(x: dict, b: int, a_len: int) -> dict:
    """Lot complété jusqu'à b positions et a_len actions (lignes et actions factices, ignorées)."""
    from .encodage import NONE_CELL
    out = {}
    for k, v in x.items():
        forme = (b, a_len, *v.shape[2:]) if k == "acts" else (b, *v.shape[1:])
        p = np.zeros(forme, v.dtype)
        if k == "acts":
            p[..., 4:] = NONE_CELL
            p[:len(v), :v.shape[1]] = v
        else:
            p[:len(v)] = v
        out[k] = p
    return out


def activer_tf32() -> None:
    """Ampere et suivants : multiplications fp32 en TF32 (bien plus rapides, précision suffisante)."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def softmax(x: np.ndarray) -> np.ndarray:
    m = x.max()
    e = np.exp(x - m)
    return e / e.sum()
