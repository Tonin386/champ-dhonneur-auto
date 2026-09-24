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
    """

    def __init__(self, model, device: str = "cpu", demi_precision: bool | None = None):
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
        self.n_evals = 0
        self.n_lots = 0

    @classmethod
    def depuis_fichier(cls, path, device: str = "cpu"):
        from .modele import charger
        model, _ = charger(path, "cpu")
        return cls(model, device)

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
        torch = self.torch
        with torch.inference_mode():
            xt = {}
            for k, v in x.items():
                v = np.ascontiguousarray(v)
                t = torch.from_numpy(v if v.flags.writeable else v.copy())
                if t.is_floating_point():
                    t = t.to(self.dtype)
                xt[k] = t.to(self.device, non_blocking=True)
            logits, vlog = self.model(xt)
            p = torch.softmax(vlog.float(), dim=-1)
            v = (p[:, 0] - p[:, 2]).cpu().numpy()
            logits = logits.float().cpu().numpy()
        self.n_evals += len(v)
        self.n_lots += 1
        return logits, v


def activer_tf32() -> None:
    """Ampere et suivants : multiplications fp32 en TF32 (bien plus rapides, précision suffisante)."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def softmax(x: np.ndarray) -> np.ndarray:
    m = x.max()
    e = np.exp(x - m)
    return e / e.sum()
