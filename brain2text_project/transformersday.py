# -*- coding: utf-8 -*-
"""
transformer_predict.py (version compatible avec transformer_ctc_train.py)

- charge un modèle Transformer-CTC à partir d'un config.json + checkpoint,
- gère :
    * DayInputLayer (day-specific input)
    * patching temporel (patch_size, patch_stride)
- fournit une fonction :
      predict_ids_from_tensor(x, config_json_path, checkpoint_path, session=None)
  qui renvoie une séquence d'IDs de phonèmes (CTC greedy).

Usage typique :

    from transformer_predict import predict_ids_from_tensor

    x = ...        # torch.Tensor [T, 512] ou [1, T, 512]
    session = "t15.2023.08.18"   # optionnel, pour choisir le bon day_idx
    pred_ids = predict_ids_from_tensor(
        x,
        config_json_path=r"...\\config.json",
        checkpoint_path=r"...\\model_bestPER.pt",
        session=session,
    )
"""

import os
import json
from typing import List, Optional, Tuple, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# Utils CTC
# =====================================================

def _norm_path(p: str) -> str:
    return os.path.normpath(p) if p else p


@torch.no_grad()
def ctc_greedy_ids_from_logits(logits: torch.Tensor, blank_id: int = 0) -> List[int]:
    """
    logits : [1, T, C] ou [T, C]
    renvoie une liste d'IDs après argmax + collapse CTC.
    """
    if logits.dim() == 3:
        logits = logits.squeeze(0)  # [T, C]
    ids = torch.argmax(logits, dim=-1).tolist()
    out, prev = [], None
    for i in ids:
        if i == blank_id:
            prev = i
            continue
        if prev is None or i != prev:
            out.append(int(i))
        prev = i
    return out


# =====================================================
# Modules (copiés du script d'entraînement)
# =====================================================

class DayInputLayer(nn.Module):
    """
    Projection spécifique par jour :
    x -> W_day x + b_day, avec Softsign + Dropout
    """
    def __init__(self, neural_dim: int, n_days: int, input_dropout: float = 0.2):
        super().__init__()
        self.neural_dim = neural_dim
        self.n_days = n_days

        self.day_activation = nn.Softsign()
        self.day_dropout = nn.Dropout(input_dropout)

        # W_d ~ identité, b_d ~ 0 pour chaque jour
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

    def forward(self, x: torch.Tensor, day_idx: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D] (D = neural_dim)
        day_idx: [B] (index du jour pour chaque élément du batch)
        """
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)  # [B,D,D]
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)  # [B,1,D]
        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_activation(x)
        x = self.day_dropout(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 20000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):  # [B,T,D]
        T = x.size(1)
        x = x + self.pe[:T].unsqueeze(0)
        return self.dropout(x)


class TransformerCTC(nn.Module):
    """
    Même architecture que dans transformer_ctc_train.py :
      - DayInputLayer
      - patching temporel
    """
    def __init__(self,
                 neural_dim=512,
                 n_classes=41,
                 d_model=256,
                 nhead=8,
                 num_layers=6,
                 dim_ff=1024,
                 dropout=0.1,
                 blank_id=0,
                 n_days=1,
                 input_dropout=0.2,
                 patch_size=14,
                 patch_stride=4):
        super().__init__()
        self.blank_id = int(blank_id)
        self.neural_dim = neural_dim
        self.n_days = n_days
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # --- Day-specific input layer ---
        self.day_input = DayInputLayer(
            neural_dim=neural_dim,
            n_days=n_days,
            input_dropout=input_dropout,
        )

        # Dimension d’entrée du Transformer après patching
        if self.patch_size > 0:
            in_dim = neural_dim * self.patch_size
        else:
            in_dim = neural_dim

        self.in_norm = nn.LayerNorm(in_dim)
        self.proj_in = nn.Linear(in_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.posenc = PositionalEncoding(d_model, dropout=dropout)
        self.head = nn.Linear(d_model, n_classes)
        with torch.no_grad():
            if self.head.bias is not None:
                self.head.bias.zero_()

    def _patch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Patching temporel interne.
        x: [B,T,D] (D = neural_dim)
        -> [B,T_p,D * patch_size]
        """
        if self.patch_size <= 0:
            return x

        B, T, C = x.shape
        if T < self.patch_size:
            return x  # trop court, on laisse comme ça

        x_u = x.unsqueeze(1)           # [B,1,T,C]
        x_u = x_u.permute(0, 3, 1, 2)  # [B,C,1,T]
        x_u = x_u.unfold(3, self.patch_size, self.patch_stride)  # [B,C,1,num_patches,patch_size]
        x_u = x_u.squeeze(2)           # [B,C,num_patches,patch_size]
        x_u = x_u.permute(0, 2, 3, 1)  # [B,num_patches,patch_size,C]
        x_p = x_u.reshape(B, x_u.size(1), self.patch_size * C)  # [B,T_p,C*patch_size]
        return x_p

    def forward(self, x: torch.Tensor, day_idx: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, neural_dim]
        day_idx: [B]
        retourne: logits [B, T_p, n_classes]
        """
        x = self.day_input(x, day_idx)
        x = self._patch(x)
        x = self.in_norm(x)
        x = self.proj_in(x)
        x = self.posenc(x)
        y = self.encoder(x)
        logits = self.head(y)
        return logits


# =====================================================
# Chargement des poids
# =====================================================

def _load_state_safely(model: nn.Module, ckpt_path: str, device: torch.device):
    """
    Charge les poids d'un checkpoint dans le modèle (strict=False).
    Gère le cas 'module.' (DDP).
    """
    ckpt_path = _norm_path(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint introuvable : {ckpt_path}")

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)  # PyTorch >= 2.4
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)  # fallback

    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]

    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print("[PREDICT] Clés manquantes (extrait) :", missing[:5])
    if unexpected:
        print("[PREDICT] Clés inattendues (extrait) :", unexpected[:5])


# =====================================================
# Cache de modèles
# =====================================================

# key = (config_json_path_norm, checkpoint_path_norm, str(device))
# value = (model, device, blank_id, n_features, sessions_list)
_LOADED_MODELS: Dict[
    Tuple[str, str, str],
    Tuple[nn.Module, torch.device, int, int, List[str]]
] = {}


def _load_transformer_from_config(
    config_json_path: str,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, torch.device, int, int, List[str]]:
    """
    Charge (si nécessaire) :
      - la config JSON,
      - le modèle TransformerCTC (day-specific + patching),
      - les poids du checkpoint.

    Retourne : (model, device, blank_id, n_features, sessions_list)
    """
    cfg_path = _norm_path(config_json_path)
    ckpt_path = _norm_path(checkpoint_path)

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config introuvable : {cfg_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    key = (cfg_path, ckpt_path, str(device))

    if key in _LOADED_MODELS:
        return _LOADED_MODELS[key]

    # Lecture du JSON
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Hyperparams (support ancien format + nouveau format nested)
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}

    def _get_cfg(keys, default=None):
        for k in keys:
            if k in model_cfg:
                return model_cfg[k]
            if k in cfg:
                return cfg[k]
        return default

    n_features = int(_get_cfg(["N_FEATURES", "n_features"], 512))
    n_classes  = int(_get_cfg(["N_CLASSES", "n_classes"], 41))
    blank_id   = int(_get_cfg(["BLANK_ID", "blank_id"], 0))

    d_model    = int(_get_cfg(["TF_D_MODEL", "d_model"], 256))
    nhead      = int(_get_cfg(["TF_NHEAD", "nhead"], 4))
    nlayer     = int(_get_cfg(["TF_LAYERS", "num_layers"], 6))
    d_ff       = int(_get_cfg(["TF_FF_DIM", "dim_ff"], 1024))
    tf_drop    = float(_get_cfg(["TF_DROPOUT", "dropout"], 0.1))

    patch_size   = int(_get_cfg(["PATCH_SIZE", "patch_size"], 0))
    patch_stride = int(_get_cfg(["PATCH_STRIDE", "patch_stride"], 1))
    n_days       = int(_get_cfg(["N_DAYS", "n_days"], 1))
    input_dropout = float(_get_cfg(["DAY_INPUT_DROPOUT", "day_input_dropout"], 0.0))

    sessions = cfg.get("SESSIONS", cfg.get("sessions", []))
    if not isinstance(sessions, list):
        sessions = []

    # Construction du modèle
    model = TransformerCTC(
        neural_dim=n_features,
        n_classes=n_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=nlayer,
        dim_ff=d_ff,
        dropout=tf_drop,
        blank_id=blank_id,
        n_days=n_days,
        input_dropout=input_dropout,
        patch_size=patch_size,
        patch_stride=patch_stride,
    ).to(device)

    # Chargement des poids
    _load_state_safely(model, ckpt_path, device)
    model.eval()

    _LOADED_MODELS[key] = (model, device, blank_id, n_features, sessions)
    return model, device, blank_id, n_features, sessions


# =====================================================
# Fonction publique : prédiction à partir d'un tenseur
# =====================================================

@torch.no_grad()
def predict_ids_from_tensor(
    x: torch.Tensor,
    config_json_path: str,
    checkpoint_path: str,
    session: Optional[str] = None,
    return_logits: bool = False,
    device: Optional[torch.device] = None,
) -> Union[List[int], Tuple[List[int], torch.Tensor]]:
    """
    x : torch.Tensor [T, n_features] ou [1, T, n_features].

    config_json_path : chemin vers le config.json du run
    checkpoint_path  : chemin vers le checkpoint (.pt)
    session          : nom de la session (ex: 't15.2023.08.18') pour choisir le bon day_idx.
                       Si None, on utilise day_idx=0.
    return_logits    : si True, renvoie aussi les logits pour la confiance.

    Renvoie : liste d'IDs de phonèmes prédits (CTC greedy + collapse),
              ou (ids, logits) si return_logits=True.
    """
    model, device, blank_id, n_features, sessions = _load_transformer_from_config(
        config_json_path,
        checkpoint_path,
        device=device,
    )

    # Mise en forme du tenseur
    if x.dim() == 2:
        # [T, C] -> [1, T, C]
        x = x.unsqueeze(0)
    elif x.dim() == 3:
        if x.size(0) != 1:
            raise ValueError(
                f"predict_ids_from_tensor attend B=1 ou [T,C], reçu shape {tuple(x.shape)}"
            )
    else:
        raise ValueError(
            f"predict_ids_from_tensor attend un tensor 2D ou 3D, reçu dim={x.dim()}"
        )

    if x.size(-1) != n_features:
        raise ValueError(
            f"Mauvais nombre de features : attendu {n_features}, reçu {x.size(-1)}"
        )

    x = x.to(device=device, dtype=torch.float32)

    # day_idx en fonction de la session (si dispo)
    if session is None or not sessions:
        day_idx = torch.zeros(1, dtype=torch.long, device=device)
    else:
        try:
            idx = sessions.index(session)
        except ValueError:
            raise ValueError(
                f"Session '{session}' introuvable dans SESSIONS du config.json : {sessions}"
            )
        day_idx = torch.tensor([idx], dtype=torch.long, device=device)

    logits = model(x, day_idx)  # [1, T_p, C]
    ids = ctc_greedy_ids_from_logits(logits, blank_id=blank_id)
    if return_logits:
        return ids, logits
    return ids
