"""Réseau politique/valeur de Champ d'honneur.

Architecture (≈ 1 à 5 M de paramètres selon la configuration) :

  jetons d'entrée : 37 cases + 8 cartes Unité (à moi, adverses, encore disponibles
  pendant le draft) + 1 jeton global
        │  plongements (type d'unité, camp, contrôle, pile) + position apprise ; les jetons
        │  d'unité n'ont pas de position : l'ensemble est invariant par permutation
        ▼
  transformeur pré-normalisé (attention multi-têtes sur tout le plateau)
        │
        ├── tête valeur : jeton global → MLP → Victoire / Nulle / Défaite
        ├── tête politique « pointeur » : pour chaque action légale,
        │   h[case1] + h[case2] + h[case3] + plongements(type, pièce, unité)
        │   + h[jeton de l'unité concernée] + projection du jeton global → MLP → logit
        └── têtes auxiliaires (apprentissage seulement, à la KataGo) : contrôle final de
            chaque Lieu, marge finale de marqueurs, main de l'adversaire

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
                       N_UNIT_TOKENS, PENDING_KINDS, UNIT_F, VERSION)

N_MARGE = 9          # marge finale de marqueurs, de -4 à +4
MAX_MAIN = 3         # pièces d'un même type dans la main adverse : 0 à 3


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
        self.emb_side = nn.Embedding(3, d)                     # à moi, adverse, disponible
        self.lin_unit = nn.Linear(UNIT_F, d)
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
        self.p_unit_ctx = nn.Linear(d, d, bias=False)          # jeton contextualisé de l'unité
        self.p_extra_ctx = nn.Linear(d, d, bias=False)
        self.p_glob = nn.Linear(d, d)
        self.p_mlp = nn.Sequential(nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # têtes auxiliaires
        self.aux_lieux = nn.Linear(d, 3)                       # neutre / à moi / adverse
        self.aux_marge = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_MARGE))
        self.aux_main = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_COIN_TYPES * (MAX_MAIN + 1)))

    def forward(self, x: dict[str, torch.Tensor], aux: bool = False):
        ci, cf = x["cell_i"], x["cell_f"]
        b = ci.shape[0]
        cells = (self.emb_pos + self.emb_type(ci[..., 0]) + self.emb_owner(ci[..., 1])
                 + self.emb_ctrl(ci[..., 2]) + self.emb_coins(ci[..., 3]) + self.lin_cell(cf))
        ui = x["unit_i"]
        units = self.emb_type(ui[..., 0]) + self.emb_side(ui[..., 1]) + self.lin_unit(x["unit_f"])
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
        # unité et unité « extra » de l'action -> leur jeton contextualisé (hors jetons adverses :
        # les deux camps peuvent avoir le même type, les actions portent sur les miens ou les
        # cartes disponibles)
        hu = h[:, N_CELLS:N_CELLS + N_UNIT_TOKENS]           # (B, 8, d)
        pas_adv = (ui[..., 1] != 1).unsqueeze(1)             # (B, 1, 8)
        for k, lin in ((2, self.p_unit_ctx), (3, self.p_extra_ctx)):
            a = acts[..., k]
            m = (a.unsqueeze(-1) == ui[..., 0].unsqueeze(1)) & pas_adv & (a > 0).unsqueeze(-1)
            z = z + lin(torch.matmul(m.to(hu.dtype), hu))
        logits = self.p_mlp(z).squeeze(-1)                  # (B, A)
        if not aux:
            return logits, value_logits
        return logits, value_logits, {
            "lieux": self.aux_lieux(h[:, :N_CELLS]),                          # (B, 37, 3)
            "marge": self.aux_marge(g),                                       # (B, 9)
            "main": self.aux_main(g).view(b, N_COIN_TYPES, MAX_MAIN + 1),     # (B, 18, 4)
        }


def valeur_scalaire(value_logits: torch.Tensor) -> torch.Tensor:
    """Espérance V - D dans [-1, 1] à partir des logits V/N/D."""
    p = F.softmax(value_logits.float(), dim=-1)
    return p[:, 0] - p[:, 2]


def sauver(path, model: ReseauChamp, extra: dict | None = None) -> None:
    sd = {k: v.detach().cpu() for k, v in _nu(model).state_dict().items()}
    torch.save({"config": asdict(_nu(model).cfg), "etat": sd, "version_encodage": VERSION,
                **(extra or {})}, path)


class ModeleIncompatible(ValueError):
    pass


def charger(path, device: str | torch.device = "cpu") -> tuple[ReseauChamp, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("version_encodage", 1) != VERSION:
        raise ModeleIncompatible(f"{path} : encodage v{ck.get('version_encodage', 1)}, "
                                 f"v{VERSION} attendu (modèle d'une génération précédente)")
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
