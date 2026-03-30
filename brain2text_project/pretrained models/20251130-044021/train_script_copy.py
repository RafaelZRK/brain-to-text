# -*- coding: utf-8 -*-
# transformer_ctc_train.py — Transformer-CTC: entraînement + éval greedy CTC
# Augmentations: smoothing, bruit blanc, offset, random cut
# Optim: AdamW + Cosine LR + grad accumulation
# Ajouts: day-specific input layers + patching temporel façon GRU baseline

import os
import json
import time
import shutil
import math
from datetime import datetime
from typing import List, Dict, Optional
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import h5py
from scipy.ndimage import gaussian_filter1d


# ===================== CONFIG & CHEMINS =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RUNS_ROOT = os.path.join(BASE_DIR, "outputs", "runs")

DATA_ROOT = os.path.join(
    BASE_DIR,
    "data",
    "t15_copyTask_neuralData",
    "hdf5_data_final",
)

SPLIT_CSV = os.path.join(
    BASE_DIR,
    "t15_sessions_random_split.csv",
)


def load_train_eval_dates_from_csv(split_csv: str):
    """
    Lit le CSV (date, type) et renvoie deux listes IDENTIQUES :
      - train_dates : ['t15.2023.08.11', ...]
      - eval_dates  : ['t15.2023.08.11', ...]
    On ne garde que les lignes avec type == 'train'.
    """
    train_dates = []
    eval_dates = []

    with open(split_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row["date"].strip()          # ex: '2023-08-11'
            typ = row["type"].strip().lower()   # 'train', 'eval', etc.

            if typ != "train":
                continue
            if not date:
                continue

            folder_name = "t15." + date.replace("-", ".")  # 't15.2023.08.11'

            # même liste pour train et eval
            train_dates.append(folder_name)
            eval_dates.append(folder_name)

    return train_dates, eval_dates


def build_h5_paths_from_dates(root: str, date_folders, filename: str = "data_train.hdf5"):
    """
    À partir d'une liste de dossiers de dates (ex: 't15.2023.08.11'),
    construit la liste des chemins complets vers filename
    (par défaut data_train.hdf5)
    """
    paths = []
    for folder in date_folders:
        full_path = os.path.join(root, folder, filename)
        if not os.path.isfile(full_path):
            print(f"[WARN] Fichier absent : {full_path}")
            continue
        paths.append(full_path)
    return paths


# --- Construction des listes de fichiers HDF5 à partir du CSV ---
TRAIN_DATES, EVAL_DATES = load_train_eval_dates_from_csv(SPLIT_CSV)

# Train : fichiers data_train.hdf5
H5_PATHS = build_h5_paths_from_dates(DATA_ROOT, TRAIN_DATES, filename="data_train.hdf5")
# Val : fichiers data_val.hdf5 sur les mêmes jours
EVAL_H5_PATHS = build_h5_paths_from_dates(DATA_ROOT, EVAL_DATES, filename="data_val.hdf5")

print("=== SPLIT DEPUIS CSV ===")
print(f"[TRAIN DATES] {len(TRAIN_DATES)} -> {len(H5_PATHS)} fichiers HDF5")
print(f"[EVAL  DATES] {len(EVAL_DATES)} -> {len(EVAL_H5_PATHS)} fichiers HDF5")

# Sessions ↔ day index
SESSIONS = TRAIN_DATES  # mêmes jours pour train/val
SESSION_TO_ID = {name: i for i, name in enumerate(SESSIONS)}
N_DAYS = len(SESSIONS)
print(f"[SESSIONS] {N_DAYS} jours : {SESSIONS}")

# ===================== PARAMS DATA =====================
NEURAL_KEY = "input_features"   # [T,512] float32
LABELS_KEY = "seq_class_ids"    # [L] int64 (padding 0 à droite)

# Modèle / dataset
N_FEATURES = 512
N_CLASSES = 41
BLANK_ID = 0

TF_D_MODEL = 512
TF_LAYERS = 10
TF_FF_DIM = 2360   # ← augmenté
TF_NHEAD  = 16     # ou 8, au choix
TF_DROPOUT = 0.3

# Patching temporel (comme GRU baseline)
PATCH_SIZE = 14
PATCH_STRIDE = 4

# Dropout sur les day-layers (comme input_layer_dropout)
DAY_INPUT_DROPOUT = 0.2

# ===================== TRAINING CONFIG =====================
BASE_LR = 2e-4
WEIGHT_DECAY = 1e-5
CLIP_NORM = 1.0
EPOCHS = 80
PRINT_EVERY = 50
USE_AMP = True
SEED = 1337
PRELOAD_RAM = False
PRELOAD_VAL = False

BATCH_SIZE = 8  # nombre de trials par batch

# AMP warmup
AMP_WARMUP_EPOCHS = 2

# Évaluation
EVAL_EVERY = 2

# Sauvegardes / nettoyage
SAVE_LAST_EVERY = 5
EMPTY_CACHE_EVERY = 10

# >>> Smoothing (train)
AUG_SMOOTH_ENABLE = True
AUG_SMOOTH_PROB = 0.5
AUG_SMOOTH_STD = 2.0
AUG_SMOOTH_SIZE = 100
AUG_SMOOTH_PADDING = "same"   # 'same' ou 'valid'

# >>> Smoothing (eval)
AUG_SMOOTH_EVAL_ENABLE = True

# Data aug supplémentaire
RANDOM_CUT_MAX = 5
NOISE_STD = 0.5
OFFSET_STD = 0.1

# Gradient accumulation
ACCUM_STEPS = 4

# Cosine scheduler warmup
WARMUP_FRAC = 0.05   # 5% des steps de warmup


# ===================== MODULES =====================
class DayInputLayer(nn.Module):
    """
    Projection spécifique par jour (comme dans GRUDecoder):
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
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):  # [B,T,D]
        T = x.size(1)
        x = x + self.pe[:T].unsqueeze(0)
        return self.dropout(x)


class TransformerCTC(nn.Module):
    """
    Transformer CTC avec :
      - day-specific input layer
      - patching temporel (patch_size, patch_stride)
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
        x: [B, T, neural_dim]   (features brutes + data-aug)
        day_idx: [B]            (index du jour pour chaque trial)
        """
        # 1) jour spécifique
        x = self.day_input(x, day_idx)   # [B,T,neural_dim]

        # 2) patching
        x = self._patch(x)               # [B,T_p,neural_dim*patch_size]

        # 3) transformer
        x = self.in_norm(x)
        x = self.proj_in(x)
        x = self.posenc(x)
        y = self.encoder(x)
        logits = self.head(y)           # [B,T_p,C]
        return logits


# ===================== CTC & METRICS =====================
@torch.no_grad()
def ctc_greedy_ids_from_logits(logits: torch.Tensor, blank_id: int = 0) -> List[int]:
    if logits.dim() == 3:
        logits = logits.squeeze(0)  # [T,C]
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


def levenshtein(a: List[int], b: List[int]) -> int:
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, y in enumerate(b, start=1):
            cur = dp[j]
            dp[j] = prev if x == y else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def phoneme_error_rate(pred: List[int], ref: List[int], blank_id: int = 0) -> float:
    r = [int(i) for i in ref if int(i) != 0]  # padding 0 retiré
    if not r:
        return 0.0 if not pred else float("nan")
    return levenshtein(pred, r) / len(r)


# ===================== GAUSSIAN SMOOTH =====================
def gauss_smooth(inputs: torch.Tensor,
                 device,
                 smooth_kernel_std: float = 2.0,
                 smooth_kernel_size: int = 100,
                 padding: str = "same") -> torch.Tensor:
    """
    Lissage gaussien 1D le long du temps.
    inputs: [B, T, C] renvoie [B, T, C] (si padding='same').
    """
    assert inputs.dim() == 3, "gauss_smooth attend un tenseur [B,T,C]"
    _, _, C = inputs.shape

    impulse = np.zeros(smooth_kernel_size, dtype=np.float32)
    impulse[smooth_kernel_size // 2] = 1.0
    g = gaussian_filter1d(impulse, smooth_kernel_std)
    keep = np.argwhere(g > 0.01)
    g = np.squeeze(g[keep])
    g = (g / np.sum(g)).astype(np.float32)
    K = int(g.shape[0])

    kern = torch.from_numpy(g).to(device=device, dtype=torch.float32).view(1, 1, K)  # [1,1,K]
    x = inputs.permute(0, 2, 1).contiguous()  # [B,C,T]
    kern = kern.repeat(C, 1, 1)  # [C,1,K]

    if padding == "same":
        pad_left = K // 2
        pad_right = K - 1 - pad_left
        x = F.pad(x, (pad_left, pad_right), mode="reflect")
        y = F.conv1d(x, kern, padding=0, groups=C)
    elif padding == "valid":
        y = F.conv1d(x, kern, padding=0, groups=C)
    else:
        try:
            y = F.conv1d(x, kern, padding=padding, groups=C)
        except Exception:
            pad_left = K // 2
            pad_right = K - 1 - pad_left
            x = F.pad(x, (pad_left, pad_right), mode="reflect")
            y = F.conv1d(x, kern, padding=0, groups=C)

    return y.permute(0, 2, 1).contiguous()  # [B,T,C]


def patched_lengths(input_lengths: torch.Tensor,
                    patch_size: int,
                    patch_stride: int) -> torch.Tensor:
    """Convertit les longueurs temporelles brutes en longueurs après patching."""
    if patch_size <= 0:
        return input_lengths
    L = (input_lengths - patch_size) // patch_stride + 1
    L = torch.clamp(L, min=1)
    return L


# ===================== VALIDATION (greedy only) =====================
@torch.no_grad()
def eval_on_validation(
    model: nn.Module,
    val_index: List[Dict],
    device: torch.device,
    blank_id: int,
    preload_val: Optional[List[Dict]] = None,
) -> float:
    model.eval()
    pers: List[float] = []
    sample_idx = int(np.random.randint(0, len(val_index))) if len(val_index) > 0 else None
    sample = None

    for k, item in enumerate(val_index):
        batch = preload_val[k] if preload_val is not None else load_trial(item)
        x = batch["x"].unsqueeze(0).to(device)   # [1, T, 512]
        y = batch["y"].to(device)                # [L]
        day_idx = torch.tensor([batch["day_idx"]], dtype=torch.int64, device=device)

        targets = y[y != 0].tolist()
        if not targets:
            continue

        if AUG_SMOOTH_EVAL_ENABLE:
            x = gauss_smooth(
                x, device=device,
                smooth_kernel_std=AUG_SMOOTH_STD,
                smooth_kernel_size=AUG_SMOOTH_SIZE,
                padding=AUG_SMOOTH_PADDING
            )

        logits = model(x, day_idx)              # [1, T_p, C]
        pred_ids = ctc_greedy_ids_from_logits(logits, blank_id=blank_id)
        per = phoneme_error_rate(pred_ids, targets, blank_id=blank_id)
        if per == per:
            pers.append(per)

        if k == sample_idx:
            trial_num = batch.get("trial_num", item.get("trial_num", -1))
            sample = {
                "trial_num": int(trial_num),
                "ref_ids": targets,
                "pred_ids": pred_ids,
                "per": float(per),
            }

    if sample is not None:
        print("\n───────────── SAMPLE (IDs) ─────────────")
        print(f"trial={sample['trial_num']} | PER={sample['per']:.3f}")
        print("REF :", " ".join(map(str, sample['ref_ids'])))
        print("PRED:", " ".join(map(str, sample['pred_ids'])))

    model.train()
    return float(np.mean(pers)) if pers else float("nan")


# ===================== DATA I/O HELPERS =====================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _list_trial_keys(file_path: str) -> List[str]:
    with h5py.File(file_path, "r") as f:
        keys = [k for k in f.keys() if k.startswith("trial_") and k[6:].isdigit()]
    keys.sort()
    return keys


def build_train_index_from_hdf5() -> List[Dict]:
    idx = []
    gid = 0
    for file_path in H5_PATHS:
        keys = _list_trial_keys(file_path)
        # dossier = .../t15.2023.08.11/data_train.hdf5 → "t15.2023.08.11"
        day_folder = os.path.basename(os.path.dirname(file_path))
        day_idx = SESSION_TO_ID[day_folder]

        for k in keys:
            idx.append({
                "global_id": gid,
                "file_path": file_path,
                "trial_key": k,
                "trial_num": int(k[6:]),
                "day_idx": day_idx,
            })
            gid += 1
    print(f"[TRAIN] Trials total indexés: {len(idx)}")
    return idx


def build_eval_index_half_per_file() -> List[Dict]:
    val_idx = []
    gid = 0
    for file_path in EVAL_H5_PATHS:
        keys = _list_trial_keys(file_path)
        n = len(keys)
        m = max(1, n // 2)
        sub = keys[:m]

        day_folder = os.path.basename(os.path.dirname(file_path))
        day_idx = SESSION_TO_ID[day_folder]

        for k in sub:
            val_idx.append({
                "global_id": gid,
                "file_path": file_path,
                "trial_key": k,
                "trial_num": int(k[6:]),
                "day_idx": day_idx,
            })
            gid += 1
        print(f"[EVAL SPLIT] {os.path.basename(file_path)} -> HALF={m} / {n}")
    print(f"[EVAL] Trials total (tous fichiers): {len(val_idx)}")
    return val_idx


def load_trial(item: Dict) -> Dict[str, torch.Tensor]:
    file_path = item["file_path"]
    trial_key = item["trial_key"]
    day_idx = item["day_idx"]
    with h5py.File(file_path, "r") as f:
        g = f[trial_key]
        x_np = g[NEURAL_KEY][:]
        y_np = g[LABELS_KEY][:]
    # remet en [T, 512] si besoin
    x_np = x_np.T if x_np.shape[0] == N_FEATURES else x_np
    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(y_np).long()
    return {"trial_num": item["trial_num"], "x": x, "y": y, "day_idx": day_idx}


def make_batch_from_items(items: List[Dict[str, torch.Tensor]], device: torch.device):
    """
    items: liste de dicts {"trial_num": int, "x": Tensor[T_i,512], "y": Tensor[L_i], "day_idx": int}
    On construit un batch:
      - x_batch: [B, T_max, 512]
      - targets: concaténation de toutes les séquences de labels
      - input_lengths: [B] longueurs temporelles (T_i)
      - target_lengths: [B] longueurs des séquences de labels (L_i sans padding)
      - day_indices: [B]
    """
    B = len(items)
    assert B > 0, "make_batch_from_items appelé avec B=0"

    lengths = [it["x"].shape[0] for it in items]
    T_max = max(lengths)

    x_batch = torch.zeros(B, T_max, N_FEATURES, dtype=torch.float32)
    targets_list = []
    target_lengths = []
    day_indices = []

    for b, it in enumerate(items):
        x = it["x"]              # [T_i, 512]
        T_i = x.shape[0]
        x_batch[b, :T_i] = x
        y = it["y"]              # [L_raw]
        y_nopad = y[y != 0]      # enlève padding 0
        targets_list.append(y_nopad)
        target_lengths.append(y_nopad.numel())
        day_indices.append(it["day_idx"])

    targets = torch.cat(targets_list, dim=0)  # [sum_L]

    x_batch = x_batch.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    input_lengths = torch.tensor(lengths, dtype=torch.int64, device=device)
    target_lengths = torch.tensor(target_lengths, dtype=torch.int64, device=device)
    day_indices = torch.tensor(day_indices, dtype=torch.int64, device=device)

    return x_batch, targets, input_lengths, target_lengths, day_indices


# ===================== MAIN TRAIN LOOP =====================
def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        print(f"🚀 {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}")
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        print("⚠️  GPU non disponible, CPU utilisé")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RUNS_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)

    train_index = build_train_index_from_hdf5()
    print(f"[SPLIT] TRAIN={len(train_index)} | VAL=will be built from EVAL_H5_PATHS")

    val_index = build_eval_index_half_per_file()

    preload = None
    if PRELOAD_RAM:
        preload = []
        total_bytes = 0
        for item in train_index:
            b = load_trial(item)  # tensors CPU classiques
            total_bytes += b["x"].numel() * 4 + b["y"].numel() * 8  # float32 + int64
            preload.append(b)
        print(f"[PRELOAD] approx RAM utilisée: {total_bytes/1e9:.2f} GB")

    print('preloaded')

    preload_val = None
    if PRELOAD_VAL:
        preload_val = []
        for item in val_index:
            b = load_trial(item)
            if device.type == "cuda":
                b["x"] = b["x"].pin_memory()
                b["y"] = b["y"].pin_memory()
            preload_val.append(b)

    model = TransformerCTC(
        neural_dim=N_FEATURES,
        n_classes=N_CLASSES,
        d_model=TF_D_MODEL,
        nhead=TF_NHEAD,
        num_layers=TF_LAYERS,
        dim_ff=TF_FF_DIM,
        dropout=TF_DROPOUT,
        blank_id=BLANK_ID,
        n_days=N_DAYS,
        input_dropout=DAY_INPUT_DROPOUT,
        patch_size=PATCH_SIZE,
        patch_stride=PATCH_STRIDE,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")

    ctc = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True, reduction="mean")

    # AdamW + Cosine LR Scheduler
    optim = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    num_batches_per_epoch = max(1, math.ceil(len(train_index) / BATCH_SIZE))
    steps_per_epoch = max(1, math.ceil(num_batches_per_epoch / ACCUM_STEPS))
    total_steps = max(1, EPOCHS * steps_per_epoch)
    warmup_steps = max(1, int(WARMUP_FRAC * total_steps))

    def lr_lambda(step: int):
        # step commence à 0 dans LambdaLR
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
    global_step = 0

    scaler = torch.amp.GradScaler("cuda", enabled=False)

    os.makedirs(RUNS_ROOT, exist_ok=True)
    with open(os.path.join(RUNS_ROOT, "latest.txt"), "w", encoding="utf-8") as f:
        f.write(run_id + "\n")

    # Save cfg
    cfg = {
        "H5_PATHS": H5_PATHS,
        "EVAL_H5_PATHS": EVAL_H5_PATHS,
        "SESSIONS": SESSIONS,
        "N_DAYS": N_DAYS,
        "NEURAL_KEY": NEURAL_KEY,
        "LABELS_KEY": LABELS_KEY,
        "MODEL": "TransformerCTC",
        "TF_D_MODEL": TF_D_MODEL,
        "TF_NHEAD": TF_NHEAD,
        "TF_LAYERS": TF_LAYERS,
        "TF_FF_DIM": TF_FF_DIM,
        "TF_DROPOUT": TF_DROPOUT,
        "PATCH_SIZE": PATCH_SIZE,
        "PATCH_STRIDE": PATCH_STRIDE,
        "DAY_INPUT_DROPOUT": DAY_INPUT_DROPOUT,
        "N_FEATURES": N_FEATURES,
        "N_CLASSES": N_CLASSES,
        "BLANK_ID": BLANK_ID,
        "BASE_LR": BASE_LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "CLIP_NORM": CLIP_NORM,
        "EPOCHS": EPOCHS,
        "USE_AMP": USE_AMP,
        "SEED": SEED,
        "PRELOAD_RAM": PRELOAD_RAM,
        "PRELOAD_VAL": PRELOAD_VAL,
        "AMP_WARMUP_EPOCHS": AMP_WARMUP_EPOCHS,
        "AUG_SMOOTH_ENABLE": AUG_SMOOTH_ENABLE,
        "AUG_SMOOTH_PROB": AUG_SMOOTH_PROB,
        "AUG_SMOOTH_STD": AUG_SMOOTH_STD,
        "AUG_SMOOTH_SIZE": AUG_SMOOTH_SIZE,
        "AUG_SMOOTH_PADDING": AUG_SMOOTH_PADDING,
        "AUG_SMOOTH_EVAL_ENABLE": AUG_SMOOTH_EVAL_ENABLE,
        "BATCH_SIZE": BATCH_SIZE,
        "RANDOM_CUT_MAX": RANDOM_CUT_MAX,
        "ACCUM_STEPS": ACCUM_STEPS,
        "WARMUP_FRAC": WARMUP_FRAC,
        "TRAIN_TRIALS": [tr["trial_num"] for tr in train_index],
        "VAL_TRIALS": [tr["trial_num"] for tr in val_index],
        "RUN_ID": run_id,
        "N_PARAMS": total,
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("\n===== TRAIN =====")
    best_per_g = float("inf")
    best_per_path = os.path.join(run_dir, "model_bestPER.pt")
    last_path = os.path.join(run_dir, "model_last.pt")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        losses_epoch = []

        # AMP warmup
        amp_enabled = (USE_AMP and (epoch > AMP_WARMUP_EPOCHS) and device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        order = np.random.permutation(len(train_index))
        num_batches = int(np.ceil(len(order) / BATCH_SIZE))

        optim.zero_grad(set_to_none=True)
        accum_counter = 0  # compte les mini-batchs *utilisés* dans l'accumulation

        for b_idx in range(num_batches):
            start = b_idx * BATCH_SIZE
            end = min((b_idx + 1) * BATCH_SIZE, len(order))
            batch_ids = order[start:end]

            items = []
            for j in batch_ids:
                b = preload[j] if (PRELOAD_RAM and preload is not None) else load_trial(train_index[j])
                y = b["y"]
                if (y != 0).sum().item() == 0:
                    continue  # trial sans labels non nuls
                items.append(b)

            if len(items) == 0:
                continue

            x_batch, targets, il, tl, day_idx = make_batch_from_items(items, device)
            B = x_batch.shape[0]
            T_max = x_batch.shape[1]
            if B == 0:
                continue

            accum_counter += 1

            # Data augmentation: smoothing
            if AUG_SMOOTH_ENABLE and (np.random.rand() < AUG_SMOOTH_PROB):
                x_batch = gauss_smooth(
                    x_batch,
                    device=device,
                    smooth_kernel_std=AUG_SMOOTH_STD,
                    smooth_kernel_size=AUG_SMOOTH_SIZE,
                    padding=AUG_SMOOTH_PADDING
                )

            # Bruit blanc
            if np.random.rand() < 0.5:
                x_batch = x_batch + torch.randn_like(x_batch) * NOISE_STD

            # Offset constant par trial
            if np.random.rand() < 0.5:
                B_aug, T_aug, C_aug = x_batch.shape
                offset = torch.randn(B_aug, 1, C_aug, device=x_batch.device) * OFFSET_STD
                x_batch = x_batch + offset

            # Random cut léger au début de la séquence
            if RANDOM_CUT_MAX > 0:
                cut = np.random.randint(0, RANDOM_CUT_MAX + 1)
                if cut > 0:
                    x_batch = x_batch[:, cut:, :]  # [B, T_max-cut, C]
                    il = il - cut                   # cohérence pour la CTC

            T_max = x_batch.shape[1]

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(x_batch, day_idx)  # [B,T_p,C]

                il_patched = patched_lengths(il, PATCH_SIZE, PATCH_STRIDE)

                logp_TBC = F.log_softmax(logits.transpose(0, 1), dim=-1)  # [T_p,B,C]
                loss_ctc = ctc(logp_TBC, targets, il_patched, tl)
                # scaling pour accumulation
                loss = loss_ctc / ACCUM_STEPS

            scaler.scale(loss).backward()

            # Step d'optimisation quand on a accumulé assez de mini-batchs
            if (accum_counter % ACCUM_STEPS) == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            losses_epoch.append(float(loss_ctc.item()))

            if (b_idx % PRINT_EVERY) == 0:
                gpu_mem = f" | GPU mem={torch.cuda.memory_allocated(0)/1e9:.2f} GB" if device.type == "cuda" else ""
                print(
                    f"[epoch {epoch:02d}] batch {b_idx:06d} | B={B} | "
                    f"loss_ctc {loss_ctc.item():.4f} | T_max={T_max}{gpu_mem}"
                )

        # Flush si des gradients restent accumulés
        if accum_counter > 0 and (accum_counter % ACCUM_STEPS) != 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

        # ===== Validation (greedy only) =====
        do_eval = ((epoch % EVAL_EVERY) == 0)
        mean_loss = float(np.mean(losses_epoch)) if losses_epoch else float("nan")
        cur_lr = optim.param_groups[0]["lr"]

        if do_eval:
            per_g = eval_on_validation(
                model=model,
                val_index=val_index,
                device=device,
                blank_id=BLANK_ID,
                preload_val=preload_val,
            )
            dt = (time.time() - t0) / 60.0
            print(f"[Epoch {epoch:02d} | EVAL] train_loss={mean_loss:.4f}  "
                  f"PER_g={per_g:.3f}  lr={cur_lr:.6f} | {dt:.1f} min")

            if per_g == per_g and per_g < best_per_g:
                best_per_g = per_g
                torch.save(model.state_dict(), best_per_path)
                print(f"  ✓ Nouveau meilleur (PER greedy) → {best_per_path} [{per_g:.3f}]")
        else:
            dt = (time.time() - t0) / 60.0
            print(f"[Epoch {epoch:02d}] train_loss={mean_loss:.4f} "
                  f"lr={cur_lr:.6f} | {dt:.1f} min (pas d'éval)")

        if (epoch % SAVE_LAST_EVERY) == 0:
            torch.save(model.state_dict(), last_path)
            print(f"  ↺ Sauvegarde périodique (last) → {last_path}")

        if device.type == "cuda" and (epoch % EMPTY_CACHE_EVERY) == 0:
            torch.cuda.empty_cache()

    try:
        shutil.copyfile(__file__, os.path.join(run_dir, "train_script_copy.py"))
    except Exception:
        pass

    print("\n===== FIN =====")
    print(f"Best (PER greedy) : {best_per_path}")
    print(f"Last              : {last_path}")
    print(f"Run dir           : {run_dir}")


if __name__ == "__main__":
    main()
