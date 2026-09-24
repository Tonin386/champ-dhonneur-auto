"""Réseau politique/valeur de Champ d'honneur.

Architecture (≈ 1 à 5 M de paramètres selon la configuration) :

  jetons d'entrée : 37 cases + 8 unités (4 à moi, 4 adverses) + 1 jeton global
        │  plongements (type d'unité, camp, contrôle, pile) + position apprise
        ▼
  transformeur pré-normalisé (attention multi-têtes sur tout le plateau)
        │
        ├── tête valeur : jeton global → MLP → Victoire / Nulle / Défaite
        └── tête politique « pointeur » : pour chaque action légale,
            h[case1] + h[case2] + h[case3] + plongements(type, pièce, unité)
            + projection du jeton global → MLP → logit

Les types d'unités partagent une même table de plongements dans l'état et
dans les actions, ce qui permet au réseau de relier « ma Cavalerie en c3 » et
« l'action : Cavalerie c3 se déplace puis attaque ».
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encodage import (ACT_F, CELL_F, GLOB_F, MAX_COINS_EMB, N_CELLS, N_COIN_TYPES, N_KINDS,
                       N_UNIT_TOKENS, PENDING_KINDS, UNIT_F)


@dataclass
class ConfigModele:
    d: int = 192          # dimension des jetons
    couches: int = 6
    tetes: int = 6
    ffn: int = 4          # facteur d'expansion du MLP
    dropout: float = 0.0

    @classmethod
    def depuis(cls, d: dict | None) -> "ConfigModele":
        return cls(**(d or {}))


class ReseauChamp(nn.Module):
    def __init__(self, cfg: ConfigModele | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ConfigModele()
        d = cfg.d
        self.emb_type = nn.Embedding(N_COIN_TYPES, d)          # partagé état / actions
        self.emb_pos = nn.Parameter(torch.randn(N_CELLS, d) * 0.02)
        self.emb_owner = nn.Embedding(3, d)
        self.emb_ctrl = nn.Embedding(4, d)
        self.emb_coins = nn.Embedding(MAX_COINS_EMB + 1, d)
        self.lin_cell = nn.Linear(CELL_F, d)
        self.emb_side = nn.Embedding(2, d)
        self.lin_unit = nn.Linear(UNIT_F, d)
        self.emb_slot = nn.Parameter(torch.randn(N_UNIT_TOKENS, d) * 0.02)
        self.emb_pending = nn.Embedding(len(PENDING_KINDS), d)
        self.lin_glob = nn.Linear(GLOB_F, d)
        self.glob_tok = nn.Parameter(torch.randn(d) * 0.02)

        layer = nn.TransformerEncoderLayer(d, cfg.tetes, cfg.ffn * d, cfg.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.tronc = nn.TransformerEncoder(layer, cfg.couches, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

        # tête valeur (V/N/D)
        self.val = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))
        # tête politique pointeur
        self.none_cell = nn.Parameter(torch.zeros(d))
        self.p_cells = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(3)])
        self.p_kind = nn.Embedding(N_KINDS, d)
        self.p_coin = nn.Linear(d, d, bias=False)
        self.p_unit = nn.Linear(d, d, bias=False)
        self.p_extra = nn.Linear(d, d, bias=False)
        self.p_glob = nn.Linear(d, d)
        self.p_mlp = nn.Sequential(nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x: dict[str, torch.Tensor]):
        ci, cf = x["cell_i"], x["cell_f"]
        b = ci.shape[0]
        cells = (self.emb_pos + self.emb_type(ci[..., 0]) + self.emb_owner(ci[..., 1])
                 + self.emb_ctrl(ci[..., 2]) + self.emb_coins(ci[..., 3]) + self.lin_cell(cf))
        ui = x["unit_i"]
        units = self.emb_slot + self.emb_type(ui[..., 0]) + self.emb_side(ui[..., 1]) + self.lin_unit(x["unit_f"])
        gi = x["glob_i"]
        glob = (self.glob_tok + self.emb_pending(gi[:, 0]) + self.emb_type(gi[:, 1])
                + self.lin_glob(x["glob_f"])).unsqueeze(1)
        h = self.norm(self.tronc(torch.cat([cells, units, glob], dim=1)))
        g = h[:, -1]
        value_logits = self.val(g)

        acts = x["acts"]                                    # (B, A, 7)
        hc = torch.cat([h[:, :N_CELLS], self.none_cell.expand(b, 1, -1)], dim=1)  # (B, 38, d)
        z = self.p_glob(g).unsqueeze(1) + self.p_kind(acts[..., 0])
        z = z + self.p_coin(self.emb_type(acts[..., 1])) + self.p_unit(self.emb_type(acts[..., 2]))
        z = z + self.p_extra(self.emb_type(acts[..., 3]))
        for k in range(3):
            idx = acts[..., 4 + k].unsqueeze(-1).expand(-1, -1, hc.shape[-1])
            z = z + self.p_cells[k](torch.gather(hc, 1, idx))
        logits = self.p_mlp(z).squeeze(-1)                  # (B, A)
        return logits, value_logits


def valeur_scalaire(value_logits: torch.Tensor) -> torch.Tensor:
    """Espérance V - D dans [-1, 1] à partir des logits V/N/D."""
    p = F.softmax(value_logits.float(), dim=-1)
    return p[:, 0] - p[:, 2]


def sauver(path, model: ReseauChamp, extra: dict | None = None) -> None:
    sd = {k: v.detach().cpu() for k, v in _nu(model).state_dict().items()}
    torch.save({"config": asdict(_nu(model).cfg), "etat": sd, **(extra or {})}, path)


def charger(path, device: str | torch.device = "cpu") -> tuple[ReseauChamp, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ReseauChamp(ConfigModele.depuis(ck["config"]))
    model.load_state_dict(ck["etat"])
    model.to(device).eval()
    return model, ck


def _nu(model):
    """Modèle sous-jacent (hors torch.compile)."""
    return getattr(model, "_orig_mod", model)


def nb_parametres(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


assert ACT_F == 7
