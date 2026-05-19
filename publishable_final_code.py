
"""
publishable_final_code.py

Clean Hierarchical Cluster and Boundary Voting Pipeline for Reference-Based
Temporal Localization in CNC Multivariate Time-Series Data.

Purpose
-------
Locate a target machining stage in CNC multivariate time-series data using a
leave-one-experiment-out evaluation protocol.

Main components
---------------
1. Context-aware segment construction.
2. TCN + squeeze-and-excitation + attention encoder.
3. Supervised contrastive representation learning.
4. Target reference-bank similarity computation.
5. Candidate-level IoU ranker/regressor.
6. Diversified retrieval pool construction.
7. Hierarchical temporal clustering.
8. Dense boundary voting and local refinement.

Reports
-------
All outputs are written under ``publishable_final_code_outputs`` using the
``publishable_final_code`` filename prefix:

- publishable_final_code_results.csv
- publishable_final_code_summary.csv
- publishable_final_code_timing.csv

Notes
-----
The evaluation diagnostics named Oracle_ALL, Oracle_POOL, and TopN_Oracle are
computed only after candidate generation for analysis. They are not used for
model selection or final prediction.
"""

import time
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================================
# CONFIGURATION
# =========================================
DATA_DIR = Path("cncmtw")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_LABEL = "Layer 2 Down"
OUTPUT_DIR = Path("publishable_final_code_outputs")
OUTPUT_PREFIX = "publishable_final_code"
VERBOSE_TABLES = False  # Set True only when detailed candidate/cluster debug tables are needed.

# ============================================================
# ABLATION / PARAMETER ANALYSIS CONFIG
# ============================================================
# Default values define the baseline behavior.
# Change ONE option at a time for controlled ablation studies.
EXPERIMENT_TAG = "baseline_safe"
HANDCRAFTED_FEATURE_MODE = "baseline"      # "baseline" or "compact"
RETRIEVAL_SCORE_MODE = "baseline"          # "baseline" or "simplified"
BOUNDARY_PRE_SCORE_MODE = "baseline"       # "baseline" or "decorrelated"
CLUSTER_SCORE_MODE = "baseline"            # "baseline" or "simplified"
VERIFY_CANDIDATE_MODE = "fixed"            # "fixed", "ratio", or "elbow"

VERIFY_TOP_M_FIXED = 320
VERIFY_TOP_RATIO = 0.35
VERIFY_TOP_MIN = 160
VERIFY_TOP_MAX = 420

BASE_CHANNELS = [
    "X1_ActualVelocity",
    "Y1_ActualVelocity",
    "Z1_ActualVelocity",
    "X1_CurrentFeedback",
    "Y1_CurrentFeedback",
    "X1_ActualAcceleration",
    "Y1_ActualAcceleration",
    "Z1_ActualAcceleration",
]

ACC_CHANNELS = [
    "X1_ActualAcceleration",
    "Y1_ActualAcceleration",
    "Z1_ActualAcceleration",
]

SMOOTH_W_DEFAULT = 11
SMOOTH_W_ACC = 5

SEG_LEN = 192
CTX_BEFORE = 128
CTX_AFTER = 128
USE_POSITION_CHANNEL = True

EMB_DIM = 128

ENC_EPOCHS = 30
ENC_LR = 1e-3
SUPCON_TEMPERATURE = 0.07

RANK_EPOCHS = 40
RANK_LR = 1e-3
BATCH_SIZE = 64

TEST_STRIDE = 8
TRAIN_STRIDE = 16

LEN_TOL = 0.45
TOP_K_RETRIEVAL = 350
TOP_N_CONSENSUS = 80

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def fmt_sec(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    sec = seconds - 60 * minutes
    if minutes < 60:
        return f"{minutes}m {sec:.1f}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes}m {sec:.1f}s"


def log(msg):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


RUN_T0 = time.perf_counter()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================
# HELPERS
# =========================================
def smooth(x, w):
    return pd.Series(x).rolling(w, center=True).mean().bfill().ffill().values


def minmax_norm(x):
    x = np.asarray(x, dtype=np.float32)
    mn = np.min(x)
    mx = np.max(x)
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def compute_verify_top_m(pool_df):
    """Select how many candidates enter boundary verification.

    fixed: preserves the original baseline.
    ratio: uses a fixed fraction of current pool size.
    elbow: chooses the largest drop in sorted boundary_pre_score, with safe bounds.
    """
    n = len(pool_df)
    if n == 0:
        return 0
    if VERIFY_CANDIDATE_MODE == "fixed":
        return int(min(VERIFY_TOP_M_FIXED, n))
    if VERIFY_CANDIDATE_MODE == "ratio":
        k = int(round(VERIFY_TOP_RATIO * n))
        return int(max(min(k, VERIFY_TOP_MAX, n), min(VERIFY_TOP_MIN, n)))
    if VERIFY_CANDIDATE_MODE == "elbow":
        col = "boundary_pre_score" if "boundary_pre_score" in pool_df.columns else "pool_score"
        scores = np.sort(pd.to_numeric(pool_df[col], errors="coerce").fillna(0.0).values.astype(float))[::-1]
        if len(scores) < 3:
            return int(len(scores))
        diffs = np.abs(np.diff(scores))
        raw_k = int(np.argmax(diffs) + 1)
        return int(max(min(raw_k, VERIFY_TOP_MAX, n), min(VERIFY_TOP_MIN, n)))
    raise ValueError(f"Unknown VERIFY_CANDIDATE_MODE={VERIFY_CANDIDATE_MODE}")


def boundary_pre_score(df, mode="baseline"):
    """Pre-score for candidates before boundary voting.

    baseline uses the full weighted candidate pre-score.
    decorrelated reduces repeated use of sim_mean through pool/retrieval/sim channels.
    """
    pred = minmax_norm(df["pred_iou"].values)
    pool = minmax_norm(df["pool_score"].values)
    sim = minmax_norm(df["sim_mean"].values)
    length = df["len_good_norm"].values
    retr = minmax_norm(df["retrieval_score"].values)
    if mode == "baseline":
        return 0.06 * pred + 0.40 * pool + 0.24 * sim + 0.16 * length + 0.14 * retr
    if mode == "decorrelated":
        return 0.20 * pred + 0.42 * pool + 0.23 * length + 0.15 * retr
    raise ValueError(f"Unknown BOUNDARY_PRE_SCORE_MODE={mode}")


def cluster_quality_score(stats, mode="baseline"):
    """Cluster-level score.

    baseline uses the full weighted cluster score.
    simplified removes the size bonus and reduces duplicated similarity/retrieval terms.
    """
    if mode == "baseline":
        return (
            0.18 * stats["max_pool"] +
            0.14 * stats["mean_pool"] +
            0.14 * stats["best_pre"] +
            0.10 * stats["max_retr"] +
            0.08 * stats["mean_retr"] +
            0.08 * stats["max_sim"] +
            0.06 * stats["mean_sim"] +
            0.08 * stats["mean_len_good"] +
            0.06 * stats["compactness"] +
            0.04 * stats["endpoint_compactness"] +
            0.03 * stats["size_bonus"] +
            0.01 * stats["pos_good"]
        )
    if mode == "simplified":
        return (
            0.30 * stats["best_pre"] +
            0.25 * stats["max_pool"] +
            0.15 * stats["mean_len_good"] +
            0.15 * stats["compactness"] +
            0.10 * stats["endpoint_compactness"] +
            0.05 * stats["purity"]
        )
    raise ValueError(f"Unknown CLUSTER_SCORE_MODE={mode}")


def normalize_labels(df):
    df = df.copy()
    df["Machining_Process"] = df["Machining_Process"].astype(str).str.strip()
    return df


def preprocess_df(df):
    df = normalize_labels(df).copy()

    for c in BASE_CHANNELS:
        if c in ACC_CHANNELS:
            df[c] = smooth(df[c].values, SMOOTH_W_ACC)
        else:
            df[c] = smooth(df[c].values, SMOOTH_W_DEFAULT)

    return df


def extract_segments(df):
    labels = df["Machining_Process"].tolist()
    segments = []
    current = labels[0]
    start = 0

    for i in range(1, len(labels)):
        if labels[i] != current:
            segments.append((current, start, i - 1))
            current = labels[i]
            start = i

    segments.append((current, start, len(labels) - 1))
    return segments


def get_target_segment(df, target_label=TARGET_LABEL):
    segs = extract_segments(df)
    target_segs = [(lab, s, e) for lab, s, e in segs if lab == target_label]

    if len(target_segs) == 0:
        return None

    return max(target_segs, key=lambda x: x[2] - x[1])


def overlap(a_s, a_e, b_s, b_e):
    return max(0, min(a_e, b_e) - max(a_s, b_s) + 1)


def iou_score(a_s, a_e, b_s, b_e):
    inter = overlap(a_s, a_e, b_s, b_e)
    union = (a_e - a_s + 1) + (b_e - b_s + 1) - inter
    return inter / union if union > 0 else 0.0


def safe_slice_with_edge_pad(X, start, end):
    T, C = X.shape

    if end < 0:
        return np.repeat(X[0:1], max(1, end - start + 1), axis=0)

    if start >= T:
        return np.repeat(X[-1:], max(1, end - start + 1), axis=0)

    s = max(0, start)
    e = min(T - 1, end)
    out = X[s:e + 1]

    if out.shape[0] == 0:
        return np.repeat(X[0:1], max(1, end - start + 1), axis=0)

    left_pad = max(0, -start)
    right_pad = max(0, end - (T - 1))

    if left_pad > 0:
        out = np.vstack([np.repeat(out[:1], left_pad, axis=0), out])

    if right_pad > 0:
        out = np.vstack([out, np.repeat(out[-1:], right_pad, axis=0)])

    return out


def z_norm(arr):
    if arr.shape[0] == 0:
        return np.zeros((1, arr.shape[1]), dtype=np.float32)

    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1e-8

    return (arr - mean) / std


def resample_multivariate(arr, target_len):
    if arr.shape[0] == 1:
        return np.repeat(arr, target_len, axis=0)

    old_x = np.linspace(0, 1, arr.shape[0])
    new_x = np.linspace(0, 1, target_len)

    out = np.zeros((target_len, arr.shape[1]), dtype=np.float32)

    for c in range(arr.shape[1]):
        out[:, c] = np.interp(new_x, old_x, arr[:, c])

    return out


def add_position_channel(arr):
    pos = np.linspace(0, 1, arr.shape[0], dtype=np.float32).reshape(-1, 1)
    return np.concatenate([arr, pos], axis=1)


def build_context_segment(X, s, e):
    seg = X[s:e + 1]
    seg = z_norm(seg)
    seg = resample_multivariate(seg, SEG_LEN)

    before = safe_slice_with_edge_pad(X, s - CTX_BEFORE, s - 1)
    after = safe_slice_with_edge_pad(X, e + 1, e + CTX_AFTER)

    before = z_norm(before)
    after = z_norm(after)

    before = resample_multivariate(before, CTX_BEFORE)
    after = resample_multivariate(after, CTX_AFTER)

    full = np.vstack([before, seg, after]).astype(np.float32)

    if USE_POSITION_CHANNEL:
        full = add_position_channel(full)

    return full


def local_consistency_score(X, start, end):
    seg = z_norm(X[start:end + 1])
    before = z_norm(safe_slice_with_edge_pad(X, start - 32, start - 1))
    after = z_norm(safe_slice_with_edge_pad(X, end + 1, end + 32))

    seg_mean = seg.mean(axis=0)
    before_mean = before.mean(axis=0)
    after_mean = after.mean(axis=0)

    boundary_shift = np.linalg.norm(seg_mean - before_mean) + np.linalg.norm(seg_mean - after_mean)
    energy = np.mean(seg ** 2)
    duration = end - start + 1

    return float(0.5 * boundary_shift + 0.5 * energy + 0.002 * duration)


def boundary_score(X, s, e):
    T = len(X)
    left_jump = np.linalg.norm(X[s] - X[s - 1]) if s > 0 else 0.0
    right_jump = np.linalg.norm(X[e + 1] - X[e]) if e < T - 1 else 0.0
    return float(left_jump + right_jump)


def variance_shift_score(X, s, e, w=24):
    left_before = z_norm(safe_slice_with_edge_pad(X, s - w, s - 1))
    left_after = z_norm(safe_slice_with_edge_pad(X, s, s + w - 1))

    right_before = z_norm(safe_slice_with_edge_pad(X, e - w + 1, e))
    right_after = z_norm(safe_slice_with_edge_pad(X, e + 1, e + w))

    return float(
        np.linalg.norm(np.var(left_after, axis=0) - np.var(left_before, axis=0)) +
        np.linalg.norm(np.var(right_after, axis=0) - np.var(right_before, axis=0))
    )


def boundary_context_score(X, s, e, w=16):
    seg = z_norm(X[s:e + 1])
    left = z_norm(safe_slice_with_edge_pad(X, s - w, s - 1))
    right = z_norm(safe_slice_with_edge_pad(X, e + 1, e + w))

    seg_mean = seg.mean(axis=0)
    left_mean = left.mean(axis=0)
    right_mean = right.mean(axis=0)

    d1 = np.linalg.norm(seg_mean - left_mean)
    d2 = np.linalg.norm(seg_mean - right_mean)
    d3 = np.linalg.norm(left_mean - right_mean)

    return float(d1 + d2 - 0.3 * d3)


def segment_energy_stats(X, s, e):
    seg = z_norm(X[s:e + 1])
    power = np.mean(seg ** 2, axis=0)
    return float(np.mean(power)), float(np.std(power))


def get_position_features(s, e, T):
    start_norm = s / max(1, T)
    end_norm = e / max(1, T)
    center_norm = ((s + e) / 2.0) / max(1, T)
    length_norm = (e - s + 1) / max(1, T)
    return start_norm, end_norm, center_norm, length_norm


def gaussian_penalty(x, mu, std):
    std = max(std, 1e-6)
    return abs(x - mu) / std


# =========================================
# MODEL
# =========================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden = max(4, channels // reduction)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t = x.shape
        z = self.pool(x).view(b, c)
        w = self.fc(z).view(b, c, 1)
        return x * w


class TemporalBlockSE(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1, dropout=0.15):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.se = SEBlock1D(out_ch, reduction=8)

        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        out = self.se(out)

        res = x if self.downsample is None else self.downsample(x)

        return self.final_relu(out + res)


class AttentionPooling(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Conv1d(in_ch, in_ch // 2, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(in_ch // 2, 1, kernel_size=1)
        )

    def forward(self, x):
        weights = self.attn(x)
        weights = torch.softmax(weights, dim=-1)
        return torch.sum(x * weights, dim=-1)


class Encoder(nn.Module):
    def __init__(self, in_ch, emb_dim=128):
        super().__init__()

        self.input_se = SEBlock1D(in_ch, reduction=4)

        self.tcn = nn.Sequential(
            TemporalBlockSE(in_ch, 64, kernel_size=5, dilation=1, dropout=0.10),
            TemporalBlockSE(64, 96, kernel_size=5, dilation=2, dropout=0.10),
            TemporalBlockSE(96, 128, kernel_size=5, dilation=4, dropout=0.15),
            TemporalBlockSE(128, 128, kernel_size=5, dilation=8, dropout=0.15),
        )

        self.pool = AttentionPooling(128)

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, emb_dim)
        )

    def forward(self, x):
        x = self.input_se(x)
        h = self.tcn(x)
        h = self.pool(h)
        z = self.fc(h)
        return nn.functional.normalize(z, dim=1)


class RankRegressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x).squeeze(-1))


# =========================================
# DATASETS
# =========================================
class SupConSegmentDataset(Dataset):
    def __init__(self, train_items):
        self.items = train_items

        labels = sorted(list(set(item["label"] for item in train_items)))
        self.label_to_id = {lab: i for i, lab in enumerate(labels)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        x = torch.tensor(item["array"].T, dtype=torch.float32)
        y = torch.tensor(self.label_to_id[item["label"]], dtype=torch.long)
        return x, y


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        logits = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positive_count = mask.sum(dim=1)
        valid = positive_count > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (positive_count + 1e-12)

        return -mean_log_prob_pos[valid].mean()


class RankDataset(Dataset):
    def __init__(self, X, y, weights):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weights[idx]


# =========================================
# FEATURE BUILDER
# =========================================
def build_candidate_features(
    z,
    sims,
    X,
    s,
    e,
    total_len,
    target_len_mean,
    target_len_std,
    pos_mean,
    pos_std,
):
    L = e - s + 1

    sim_max = float(np.max(sims))
    sim_mean = float(np.mean(sims))
    sim_std = float(np.std(sims))
    sim_top3_mean = float(np.mean(np.sort(sims)[-3:])) if len(sims) >= 3 else sim_mean

    local_score = local_consistency_score(X, s, e)
    b_score = boundary_score(X, s, e)
    b_ctx = boundary_context_score(X, s, e)
    var_shift = variance_shift_score(X, s, e)
    eng_mean, eng_std = segment_energy_stats(X, s, e)

    len_penalty = abs(L - target_len_mean) / max(target_len_std, 1e-6)

    start_norm, end_norm, center_norm, length_norm = get_position_features(s, e, total_len)
    pos_penalty = gaussian_penalty(center_norm, pos_mean, pos_std)

    if HANDCRAFTED_FEATURE_MODE == "baseline":
        handcrafted = np.array([
            sim_max, sim_mean, sim_std, sim_top3_mean,
            local_score, b_score, b_ctx, var_shift,
            eng_mean, eng_std, len_penalty,
            start_norm, end_norm, center_norm, length_norm, pos_penalty,
        ], dtype=np.float32)
    elif HANDCRAFTED_FEATURE_MODE == "compact":
        # Compact ablation: keep only the least redundant, most interpretable features.
        handcrafted = np.array([
            sim_mean, sim_top3_mean, sim_std,
            len_penalty, local_score, b_ctx,
        ], dtype=np.float32)
    else:
        raise ValueError(f"Unknown HANDCRAFTED_FEATURE_MODE={HANDCRAFTED_FEATURE_MODE}")

    feat = np.concatenate([z.astype(np.float32), handcrafted], axis=0).astype(np.float32)

    if RETRIEVAL_SCORE_MODE == "baseline":
        retrieval_raw = (
            0.15 * sim_max +
            0.35 * sim_mean +
            0.25 * sim_top3_mean
            - 0.10 * sim_std
            - 0.15 * len_penalty
            - 0.06 * pos_penalty
            + 0.02 * local_score
            + 0.01 * b_ctx
        )
    elif RETRIEVAL_SCORE_MODE == "simplified":
        # Reduced redundancy: fewer repeated similarity terms, stronger context term.
        retrieval_raw = (
            0.45 * sim_top3_mean +
            0.25 * sim_mean
            - 0.10 * sim_std
            - 0.15 * len_penalty
            + 0.15 * b_ctx
        )
    else:
        raise ValueError(f"Unknown RETRIEVAL_SCORE_MODE={RETRIEVAL_SCORE_MODE}")

    meta = {
        "len": L,
        "sim_max": sim_max,
        "sim_mean": sim_mean,
        "sim_std": sim_std,
        "sim_top3_mean": sim_top3_mean,
        "local_score": local_score,
        "boundary_score": b_score,
        "boundary_ctx": b_ctx,
        "variance_shift": var_shift,
        "eng_mean": eng_mean,
        "eng_std": eng_std,
        "len_penalty": len_penalty,
        "start_norm": start_norm,
        "end_norm": end_norm,
        "center_norm": center_norm,
        "length_norm": length_norm,
        "pos_penalty": pos_penalty,
        "retrieval_score": retrieval_raw,
    }

    return feat, meta


# =========================================
# LOAD DATA
# =========================================
log(f"publishable_final_code started | tag={EXPERIMENT_TAG} | target={TARGET_LABEL} | device={DEVICE}")
log(f"Data dir: {DATA_DIR.resolve() if DATA_DIR.exists() else DATA_DIR}")
all_dfs = {}
load_t0 = time.perf_counter()

for exp_id in range(1, 19):
    file_path = DATA_DIR / f"experiment_{exp_id:02d}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} bulunamadı.")

    df = pd.read_csv(file_path)
    df = preprocess_df(df)
    all_dfs[exp_id] = df
    log(f"Loaded EXP {exp_id:02d}: rows={len(df)} file={file_path.name}")

log(f"Data loaded in {fmt_sec(time.perf_counter() - load_t0)}")


# =========================================
# MAIN LOOP
# =========================================
results = []
timing_rows = []

for test_id in range(1, 19):
    exp_t0 = time.perf_counter()
    log("=" * 100)
    log(f"EXP {test_id:02d} START")

    train_items = []
    target_train_lengths = []
    target_train_centers = []

    for exp_id in range(1, 19):
        if exp_id == test_id:
            continue

        df = all_dfs[exp_id]
        X = df[BASE_CHANNELS].to_numpy(dtype=np.float32)
        T = len(X)

        for lab, s, e in extract_segments(df):
            arr = build_context_segment(X, s, e)

            train_items.append({
                "label": lab,
                "array": arr,
                "exp_id": exp_id,
                "start": s,
                "end": e,
                "length": e - s + 1,
            })

            if lab == TARGET_LABEL:
                target_train_lengths.append(e - s + 1)
                target_train_centers.append(((s + e) / 2.0) / max(1, T))

    test_df = all_dfs[test_id]
    test_X = test_df[BASE_CHANNELS].to_numpy(dtype=np.float32)
    test_T = len(test_X)

    gt = get_target_segment(test_df)

    if gt is None:
        exp_elapsed = time.perf_counter() - exp_t0
        log(f"EXP {test_id:02d}: target label not found -> skipped | elapsed={fmt_sec(exp_elapsed)}")
        results.append({
            "Experiment": test_id,
            "Pred_Start": None,
            "Pred_End": None,
            "True_Start": None,
            "True_End": None,
            "IoU": None,
            "PreBoundary_IoU": None,
            "Weighted_IoU": None,
            "TopN_Oracle_IoU": None,
            "Oracle_ALL_IoU": None,
            "Oracle_POOL_IoU": None,
            "Pool_Loss": None,
            "Selection_Loss": None,
            "Experiment_Time_Sec": exp_elapsed,
        "Experiment_Tag": EXPERIMENT_TAG,
        "Handcrafted_Feature_Mode": HANDCRAFTED_FEATURE_MODE,
        "Retrieval_Score_Mode": RETRIEVAL_SCORE_MODE,
        "Boundary_Pre_Score_Mode": BOUNDARY_PRE_SCORE_MODE,
        "Cluster_Score_Mode": CLUSTER_SCORE_MODE,
        "Verify_Candidate_Mode": VERIFY_CANDIDATE_MODE,
        "Verify_Top_M": None,
        })
        timing_rows.append({
            "Experiment": test_id,
            "Status": "no_target",
            "Experiment_Time_Sec": exp_elapsed,
        "Experiment_Tag": EXPERIMENT_TAG,
        "Handcrafted_Feature_Mode": HANDCRAFTED_FEATURE_MODE,
        "Retrieval_Score_Mode": RETRIEVAL_SCORE_MODE,
        "Boundary_Pre_Score_Mode": BOUNDARY_PRE_SCORE_MODE,
        "Cluster_Score_Mode": CLUSTER_SCORE_MODE,
        "Verify_Candidate_Mode": VERIFY_CANDIDATE_MODE,
        "Verify_Top_M": None,
            "Experiment_Time": fmt_sec(exp_elapsed),
        })
        continue

    _, ts, te = gt

    # =========================================
    # TRAIN ENCODER
    # =========================================
    enc_t0 = time.perf_counter()
    ds = SupConSegmentDataset(train_items)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    in_ch = len(BASE_CHANNELS) + (1 if USE_POSITION_CHANNEL else 0)

    encoder = Encoder(in_ch, EMB_DIM).to(DEVICE)
    opt = torch.optim.Adam(encoder.parameters(), lr=ENC_LR)
    loss_fn = SupConLoss(temperature=SUPCON_TEMPERATURE)

    encoder.train()

    for epoch in range(ENC_EPOCHS):
        losses = []

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            z = encoder(xb)
            loss = loss_fn(z, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        print(
            f"Encoder Epoch {epoch + 1} SupCon loss {np.mean(losses):.4f}"
            if len(losses)
            else f"Encoder Epoch {epoch + 1} skipped"
        )

    log(f"EXP {test_id:02d} encoder trained in {fmt_sec(time.perf_counter() - enc_t0)}")

    # =========================================
    # TARGET BANK
    # =========================================
    bank_t0 = time.perf_counter()
    encoder.eval()
    target_bank = []

    with torch.no_grad():
        for item in train_items:
            if item["label"] != TARGET_LABEL:
                continue

            x = torch.tensor(item["array"].T, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            z = encoder(x).cpu().numpy()[0]
            target_bank.append(z)


    if len(target_bank) == 0:
        log("No target prototype")
        timing_rows.append({"Experiment": test_id, "Status": "no_target_prototype", "Experiment_Time_Sec": time.perf_counter() - exp_t0})
        continue

    target_bank = np.stack(target_bank, axis=0)

    log(f"EXP {test_id:02d} target bank built in {fmt_sec(time.perf_counter() - bank_t0)} | refs={len(target_bank)}")

    target_len_mean = float(np.mean(target_train_lengths))
    target_len_std = float(np.std(target_train_lengths) + 1e-6)

    pos_mean = float(np.mean(target_train_centers))
    pos_std = float(np.std(target_train_centers) + 1e-6)

    min_len = max(20, int(target_len_mean * (1 - LEN_TOL)))
    max_len = int(target_len_mean * (1 + LEN_TOL))

    # =========================================
    # TRAIN RANKER - stable regression
    # =========================================
    rank_data_t0 = time.perf_counter()
    all_rank_rows = []

    with torch.no_grad():
        for exp_id in range(1, 19):
            if exp_id == test_id:
                continue

            df = all_dfs[exp_id]
            X = df[BASE_CHANNELS].to_numpy(dtype=np.float32)
            T = len(X)

            gt_seg = get_target_segment(df)
            if gt_seg is None:
                continue

            _, gts, gte = gt_seg

            exp_rows = []

            for L in range(min_len, max_len + 1, 8):
                for s in range(0, len(X) - L + 1, TRAIN_STRIDE):
                    e = s + L - 1

                    arr = build_context_segment(X, s, e)
                    x = torch.tensor(arr.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    z = encoder(x).cpu().numpy()[0]

                    sims = np.dot(target_bank, z)

                    feat, meta = build_candidate_features(
                        z=z,
                        sims=sims,
                        X=X,
                        s=s,
                        e=e,
                        total_len=T,
                        target_len_mean=target_len_mean,
                        target_len_std=target_len_std,
                        pos_mean=pos_mean,
                        pos_std=pos_std,
                    )

                    cand_iou = iou_score(s, e, gts, gte)

                    exp_rows.append({
                        "feat": feat,
                        "iou": cand_iou,
                        "retrieval_score": meta["retrieval_score"],
                        "sim_mean": meta["sim_mean"],
                        "sim_max": meta["sim_max"],
                        "len_penalty": meta["len_penalty"],
                        "start": s,
                        "end": e,
                    })

            if len(exp_rows) == 0:
                continue

            exp_df = pd.DataFrame([
                {k: v for k, v in r.items() if k != "feat"}
                for r in exp_rows
            ])

            pos_idx = exp_df[exp_df["iou"] >= 0.50].index.tolist()

            mid_idx = exp_df[
                (exp_df["iou"] >= 0.20) & (exp_df["iou"] < 0.50)
            ].index.tolist()

            hard_neg_df = exp_df[
                exp_df["iou"] <= 0.10
            ].sort_values(
                ["retrieval_score", "sim_mean", "sim_max"],
                ascending=False
            )

            hard_neg_idx = hard_neg_df.head(
                max(100, len(pos_idx))
            ).index.tolist()

            easy_neg_df = exp_df[
                exp_df["iou"] <= 0.05
            ].sort_values(
                ["retrieval_score", "sim_mean"],
                ascending=True
            )

            easy_neg_idx = easy_neg_df.head(
                max(80, len(pos_idx) // 2)
            ).index.tolist()

            oracle_near_idx = exp_df.sort_values(
                "iou", ascending=False
            ).head(120).index.tolist()

            selected_idx = set()
            selected_idx.update(pos_idx)
            selected_idx.update(mid_idx)
            selected_idx.update(hard_neg_idx)
            selected_idx.update(easy_neg_idx)
            selected_idx.update(oracle_near_idx)

            retr_q75 = exp_df["retrieval_score"].quantile(0.75)

            for idx in selected_idx:
                row = exp_rows[idx]
                iou = row["iou"]

                is_hard_negative = (
                    iou <= 0.10 and
                    row["retrieval_score"] >= retr_q75
                )

                if iou >= 0.70:
                    w = 7.0
                elif iou >= 0.50:
                    w = 5.0
                elif iou >= 0.20:
                    w = 3.0
                elif is_hard_negative:
                    w = 2.5
                else:
                    w = 1.0

                all_rank_rows.append({
                    "feat": row["feat"],
                    "iou": iou,
                    "weight": w,
                    "hard_negative": is_hard_negative,
                })

    if len(all_rank_rows) == 0:
        log("No rank data")
        timing_rows.append({"Experiment": test_id, "Status": "no_rank_data", "Experiment_Time_Sec": time.perf_counter() - exp_t0})
        continue

    log(f"EXP {test_id:02d} rank training data generated in {fmt_sec(time.perf_counter() - rank_data_t0)}")

    rank_X = np.array([r["feat"] for r in all_rank_rows], dtype=np.float32)
    rank_y = np.array([r["iou"] for r in all_rank_rows], dtype=np.float32)
    rank_w = np.array([r["weight"] for r in all_rank_rows], dtype=np.float32)

    hard_neg_count = sum(1 for r in all_rank_rows if r["hard_negative"])

    print("Rank samples:", len(rank_X))
    print("Rank hard negatives:", hard_neg_count)
    print("Rank target mean IoU:", float(rank_y.mean()))
    print("Rank target max IoU:", float(rank_y.max()))
    print("Rank positives IoU>=0.5:", int(np.sum(rank_y >= 0.5)))
    print("Rank medium IoU>=0.2:", int(np.sum(rank_y >= 0.2)))

    rank_ds = RankDataset(rank_X, rank_y, rank_w)
    rank_loader = DataLoader(rank_ds, batch_size=BATCH_SIZE, shuffle=True)

    ranker = RankRegressor(rank_X.shape[1]).to(DEVICE)
    rank_opt = torch.optim.AdamW(ranker.parameters(), lr=RANK_LR, weight_decay=1e-4)

    ranker.train()
    rank_train_t0 = time.perf_counter()

    for epoch in range(RANK_EPOCHS):
        losses = []

        for xb, yb, wb in rank_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            wb = wb.to(DEVICE)

            pred_iou = ranker(xb)
            pred_iou = torch.clamp(pred_iou, 0.01, 0.99)

            loss_raw = nn.functional.smooth_l1_loss(
                pred_iou,
                yb,
                reduction="none",
                beta=0.10
            )

            loss = (loss_raw * wb).mean()

            rank_opt.zero_grad()
            loss.backward()
            rank_opt.step()

            losses.append(loss.item())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Ranker Epoch {epoch + 1} weighted SmoothL1 loss {np.mean(losses):.4f}")

    log(f"EXP {test_id:02d} ranker trained in {fmt_sec(time.perf_counter() - rank_train_t0)}")

    # =========================================
    # TEST CANDIDATES
    # =========================================
    test_cand_t0 = time.perf_counter()
    test_candidates = []

    with torch.no_grad():
        for L in range(min_len, max_len + 1, 8):
            for s in range(0, len(test_X) - L + 1, TEST_STRIDE):
                e = s + L - 1

                arr = build_context_segment(test_X, s, e)
                x = torch.tensor(arr.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                z = encoder(x).cpu().numpy()[0]

                sims = np.dot(target_bank, z)

                feat, meta = build_candidate_features(
                    z=z,
                    sims=sims,
                    X=test_X,
                    s=s,
                    e=e,
                    total_len=test_T,
                    target_len_mean=target_len_mean,
                    target_len_std=target_len_std,
                    pos_mean=pos_mean,
                    pos_std=pos_std,
                )

                test_candidates.append({
                    "start": s,
                    "end": e,
                    "feat": feat,
                    **meta,
                })

    cand_df = pd.DataFrame([
        {k: v for k, v in c.items() if k != "feat"}
        for c in test_candidates
    ])
    log(f"EXP {test_id:02d} test candidates generated in {fmt_sec(time.perf_counter() - test_cand_t0)} | candidates={len(cand_df)}")

    # =========================================
    # EXPERIMENT-WISE NORMALIZED POOL SCORE
    # =========================================
    cand_df["retr_norm"] = minmax_norm(cand_df["retrieval_score"].values)
    cand_df["sim_mean_norm"] = minmax_norm(cand_df["sim_mean"].values)
    cand_df["sim_top3_norm"] = minmax_norm(cand_df["sim_top3_mean"].values)
    cand_df["len_good_norm"] = 1.0 - minmax_norm(cand_df["len_penalty"].values)

    cand_df["pool_score"] = (
        0.45 * cand_df["retr_norm"] +
        0.25 * cand_df["sim_mean_norm"] +
        0.15 * cand_df["sim_top3_norm"] +
        0.15 * cand_df["len_good_norm"]
    )

    cand_df["oracle_iou"] = cand_df.apply(
        lambda r: iou_score(int(r["start"]), int(r["end"]), ts, te),
        axis=1
    )

    oracle_all_row = cand_df.loc[cand_df["oracle_iou"].idxmax()]
    oracle_all_iou = float(oracle_all_row["oracle_iou"])
    oracle_all_start = int(oracle_all_row["start"])
    oracle_all_end = int(oracle_all_row["end"])

    # =========================================
    # DIVERSIFIED RETRIEVAL POOL
    # =========================================
    pool_indices = set()

    sort_keys = [
        ["pool_score", "retrieval_score", "sim_mean"],
        ["retrieval_score", "sim_mean", "sim_top3_mean"],
        ["sim_mean", "retrieval_score", "sim_max"],
        ["sim_top3_mean", "sim_mean", "retrieval_score"],
        ["len_good_norm", "retrieval_score", "sim_mean"],
    ]

    for keys in sort_keys:
        idxs = cand_df.sort_values(keys, ascending=False).head(TOP_K_RETRIEVAL).index.tolist()
        pool_indices.update(idxs)

    pool_indices = sorted(list(pool_indices))
    pool_df = cand_df.loc[pool_indices].copy()

    oracle_pool_row = pool_df.loc[pool_df["oracle_iou"].idxmax()]
    oracle_pool_iou = float(oracle_pool_row["oracle_iou"])
    oracle_pool_start = int(oracle_pool_row["start"])
    oracle_pool_end = int(oracle_pool_row["end"])

    print("\n========== ORACLE ==========")
    print(f"GT: {ts}-{te}")
    print(f"[ALL ]  {oracle_all_start}-{oracle_all_end} | IoU={oracle_all_iou:.4f}")
    print(f"[POOL]  {oracle_pool_start}-{oracle_pool_end} | IoU={oracle_pool_iou:.4f}")
    print(f"Pool Loss = {oracle_all_iou - oracle_pool_iou:.4f}")
    print(f"Pool size = {len(pool_indices)}")
    print("============================\n")

    # =========================================
    # RANKER OVER POOL
    # =========================================
    ranker.eval()

    pool_feats = np.stack([test_candidates[idx]["feat"] for idx in pool_indices], axis=0)
    pool_tensor = torch.tensor(pool_feats, dtype=torch.float32)

    pred_ious = []

    with torch.no_grad():
        for i in range(0, len(pool_tensor), 512):
            xb = pool_tensor[i:i + 512].to(DEVICE)
            out = ranker(xb).cpu().numpy()
            out = np.clip(out, 0.01, 0.99)
            pred_ious.extend(out.tolist())

    pool_df["pred_iou"] = pred_ious

    # =========================================
    # PRE-BOUNDARY SELECTION: CONSENSUS + WEIGHTED ENDPOINT
    # =========================================
    pred_iou_norm = minmax_norm(pool_df["pred_iou"].values)
    pool_score_norm = minmax_norm(pool_df["pool_score"].values)
    sim_norm = minmax_norm(pool_df["sim_mean"].values)
    len_good_norm = pool_df["len_good_norm"].values

    pool_df["pre_score"] = (
        0.10 * pred_iou_norm +
        0.45 * pool_score_norm +
        0.25 * sim_norm +
        0.20 * len_good_norm
    )

    top_df = pool_df.sort_values(
        ["pre_score", "pool_score", "retrieval_score", "sim_mean"],
        ascending=False
    ).head(TOP_N_CONSENSUS).copy()

    top_df["vote_weight"] = (
        0.10 * minmax_norm(top_df["pred_iou"].values) +
        0.45 * minmax_norm(top_df["pool_score"].values) +
        0.25 * minmax_norm(top_df["sim_mean"].values) +
        0.20 * top_df["len_good_norm"].values
    )

    top_df["vote_weight"] = top_df["vote_weight"] + 1e-6

    timeline_vote = np.zeros(test_T, dtype=np.float32)

    for _, r in top_df.iterrows():
        s = int(r["start"])
        e = int(r["end"])
        w = float(r["vote_weight"])

        s = max(0, min(test_T - 1, s))
        e = max(0, min(test_T - 1, e))

        if e >= s:
            timeline_vote[s:e + 1] += w

    timeline_vote_smooth = pd.Series(timeline_vote).rolling(
        window=21,
        center=True,
        min_periods=1
    ).mean().values.astype(np.float32)

    consensus_scores = []
    target_len = float(target_len_mean)

    for _, r in top_df.iterrows():
        s = int(r["start"])
        e = int(r["end"])
        L = e - s + 1

        s = max(0, min(test_T - 1, s))
        e = max(0, min(test_T - 1, e))

        if e < s:
            consensus_scores.append(-1e9)
            continue

        segment_support = float(np.mean(timeline_vote_smooth[s:e + 1]))
        center_support = float(np.max(timeline_vote_smooth[s:e + 1]))
        len_pen = abs(L - target_len) / max(target_len, 1e-6)

        consensus_score = (
            0.55 * segment_support +
            0.20 * center_support +
            0.15 * float(r["pool_score"]) +
            0.10 * float(r["pre_score"])
            - 0.10 * len_pen
        )

        consensus_scores.append(consensus_score)

    top_df["consensus_score"] = consensus_scores

    # Pre-boundary candidate selection
    best = top_df.sort_values(
        ["consensus_score", "vote_weight", "pool_score"],
        ascending=False
    ).iloc[0]

    cand_ps = int(best["start"])
    cand_pe = int(best["end"])
    cand_iou = iou_score(cand_ps, cand_pe, ts, te)

    # =========================================
    # FINAL SELECTION: Hierarchical Cluster Selection + Dense Boundary Voting
    # =========================================
    # First, a temporal candidate cluster is selected; then dense boundary voting
    # is applied only inside the selected cluster.

    # 1) Build a broad and diversified verification set from the retrieval pool
    pool_df["boundary_pre_score"] = boundary_pre_score(pool_df, BOUNDARY_PRE_SCORE_MODE)
    VERIFY_TOP_M = compute_verify_top_m(pool_df)

    verify_indices = set()
    v_sort_keys = [
        ["boundary_pre_score", "pool_score", "sim_mean"],
        ["pool_score", "retrieval_score", "sim_mean"],
        ["sim_mean", "pool_score", "retrieval_score"],
        ["len_good_norm", "pool_score", "sim_mean"],
        ["retrieval_score", "pool_score", "sim_mean"],
    ]

    per_key = max(70, VERIFY_TOP_M // len(v_sort_keys))

    for keys in v_sort_keys:
        idxs = pool_df.sort_values(keys, ascending=False).head(per_key).index.tolist()
        verify_indices.update(idxs)

    if len(verify_indices) < VERIFY_TOP_M:
        extra = pool_df.sort_values(
            ["boundary_pre_score", "pool_score", "sim_mean"],
            ascending=False
        ).head(VERIFY_TOP_M).index.tolist()
        verify_indices.update(extra)

    verify_indices = list(verify_indices)
    verify_df = pool_df.loc[verify_indices].copy()
    verify_df = verify_df.sort_values(
        ["boundary_pre_score", "pool_score", "sim_mean"],
        ascending=False
    ).head(VERIFY_TOP_M).copy()

    top_df = verify_df.copy()

    # 2) Candidate vote weight
    verify_df["vote_weight"] = boundary_pre_score(verify_df, BOUNDARY_PRE_SCORE_MODE)
    verify_df["vote_weight"] = verify_df["vote_weight"] + 1e-6

    q = verify_df["vote_weight"].quantile(0.20)
    candidate_df = verify_df[verify_df["vote_weight"] >= q].copy()
    if len(candidate_df) < 40:
        candidate_df = verify_df.copy()

    target_len_int = int(round(target_len_mean))

    # 3) Temporal clustering by candidate center
    candidate_df["center"] = (candidate_df["start"] + candidate_df["end"]) / 2.0
    cluster_gap = max(int(0.45 * target_len_int), 48)

    sorted_df = candidate_df.sort_values("center").copy()
    cluster_ids = []
    current_cluster = 0
    prev_center = None
    for c in sorted_df["center"].values:
        if prev_center is None:
            cluster_ids.append(current_cluster)
        else:
            if abs(c - prev_center) > cluster_gap:
                current_cluster += 1
            cluster_ids.append(current_cluster)
        prev_center = c

    sorted_df["cluster_id"] = cluster_ids
    candidate_df = sorted_df.sort_index().copy()

    # 4) Cluster scoring
    cluster_rows = []
    for cid, cdf in candidate_df.groupby("cluster_id"):
        cdf = cdf.copy()
        if len(cdf) == 0:
            continue

        total_vote = float(cdf["vote_weight"].sum())
        mean_vote = float(cdf["vote_weight"].mean())
        max_vote = float(cdf["vote_weight"].max())
        mean_pool = float(cdf["pool_score"].mean())
        max_pool = float(cdf["pool_score"].max())
        mean_retr = float(cdf["retrieval_score"].mean())
        max_retr = float(cdf["retrieval_score"].max())
        mean_sim = float(cdf["sim_mean"].mean())
        max_sim = float(cdf["sim_mean"].max())
        mean_len_good = float(cdf["len_good_norm"].mean())
        max_len_good = float(cdf["len_good_norm"].max())
        center_mean = float(cdf["center"].mean())
        center_std = float(cdf["center"].std()) if len(cdf) > 1 else 0.0
        start_std = float(cdf["start"].std()) if len(cdf) > 1 else 0.0
        end_std = float(cdf["end"].std()) if len(cdf) > 1 else 0.0
        compactness = 1.0 / (1.0 + center_std / max(target_len_int, 1))
        endpoint_compactness = 1.0 / (1.0 + (start_std + end_std) / max(2 * target_len_int, 1))
        size_bonus = float(np.log1p(len(cdf)))

        best_row = cdf.sort_values(
            ["boundary_pre_score", "pool_score", "sim_mean"],
            ascending=False
        ).iloc[0]
        best_pre = float(best_row["boundary_pre_score"])
        best_pool = float(best_row["pool_score"])
        best_sim = float(best_row["sim_mean"])
        mean_pos_pen = float(cdf["pos_penalty"].mean())
        pos_good = 1.0 / (1.0 + max(0.0, mean_pos_pen))

        top_quality = float(cdf.sort_values("boundary_pre_score", ascending=False).head(min(10, len(cdf)))["boundary_pre_score"].mean())
        mean_quality = float(cdf["boundary_pre_score"].mean()) + 1e-8
        purity = float(np.clip((top_quality / mean_quality) / 1.25, 0.0, 1.0))

        cluster_stats = {
            "max_pool": max_pool,
            "mean_pool": mean_pool,
            "best_pre": best_pre,
            "max_retr": max_retr,
            "mean_retr": mean_retr,
            "max_sim": max_sim,
            "mean_sim": mean_sim,
            "mean_len_good": mean_len_good,
            "compactness": compactness,
            "endpoint_compactness": endpoint_compactness,
            "size_bonus": size_bonus,
            "pos_good": pos_good,
            "purity": purity,
        }
        cluster_score = cluster_quality_score(cluster_stats, CLUSTER_SCORE_MODE)

        cluster_rows.append({
            "cluster_id": int(cid),
            "cluster_score": float(cluster_score),
            "n": int(len(cdf)),
            "total_vote": total_vote,
            "mean_vote": mean_vote,
            "max_vote": max_vote,
            "mean_pool": mean_pool,
            "max_pool": max_pool,
            "mean_retr": mean_retr,
            "max_retr": max_retr,
            "mean_sim": mean_sim,
            "max_sim": max_sim,
            "mean_len_good": mean_len_good,
            "max_len_good": max_len_good,
            "center_mean": center_mean,
            "center_std": center_std,
            "compactness": compactness,
            "endpoint_compactness": endpoint_compactness,
            "purity": purity,
            "size_bonus": size_bonus,
            "best_pre": best_pre,
            "best_pool": best_pool,
            "best_sim": best_sim,
            "mean_pos_penalty": mean_pos_pen,
            "pos_good": pos_good,
        })

    cluster_df = pd.DataFrame(cluster_rows)
    if len(cluster_df) == 0:
        chosen_cluster_id = -1
        cluster_vote_df = candidate_df.copy()
        num_clusters = 0
        best_cluster_score = np.nan
    else:
        cluster_df = cluster_df.sort_values(
            ["cluster_score", "max_pool", "best_pre", "total_vote"],
            ascending=False
        ).reset_index(drop=True)
        chosen_cluster_id = int(cluster_df.iloc[0]["cluster_id"])
        best_cluster_score = float(cluster_df.iloc[0]["cluster_score"])
        num_clusters = int(len(cluster_df))
        cluster_vote_df = candidate_df[candidate_df["cluster_id"] == chosen_cluster_id].copy()

    if len(cluster_vote_df) < 12:
        if len(cluster_df) > 0:
            c_mean = float(cluster_df.iloc[0]["center_mean"])
        else:
            c_mean = float(candidate_df["center"].mean())
        radius = max(int(0.65 * target_len_int), 64)
        fallback_df = candidate_df[np.abs(candidate_df["center"] - c_mean) <= radius].copy()
        if len(fallback_df) >= 12:
            cluster_vote_df = fallback_df.copy()

    # 5) Dense Boundary Voting only inside selected cluster
    cq = cluster_vote_df["vote_weight"].quantile(0.15)
    vote_df = cluster_vote_df[cluster_vote_df["vote_weight"] >= cq].copy()
    if len(vote_df) < 8:
        vote_df = cluster_vote_df.copy()

    start_vote = np.zeros(test_T, dtype=np.float32)
    end_vote = np.zeros(test_T, dtype=np.float32)
    segment_vote = np.zeros(test_T, dtype=np.float32)

    sigma = max(6.0, 0.075 * target_len_int)
    radius = int(max(12, 3 * sigma))

    def add_gaussian_vote(arr, center, weight):
        center = int(center)
        lo = max(0, center - radius)
        hi = min(test_T - 1, center + radius)
        xs = np.arange(lo, hi + 1)
        vals = np.exp(-0.5 * ((xs - center) / sigma) ** 2).astype(np.float32)
        arr[lo:hi + 1] += float(weight) * vals

    for _, r in vote_df.iterrows():
        s = int(r["start"])
        e = int(r["end"])
        w = float(r["vote_weight"])
        s = max(0, min(test_T - 1, s))
        e = max(0, min(test_T - 1, e))
        if e < s:
            continue
        add_gaussian_vote(start_vote, s, w)
        add_gaussian_vote(end_vote, e, w)
        segment_vote[s:e + 1] += w

    smooth_w = max(7, int(0.075 * target_len_int))
    if smooth_w % 2 == 0:
        smooth_w += 1

    start_vote_s = pd.Series(start_vote).rolling(
        window=smooth_w, center=True, min_periods=1
    ).mean().values.astype(np.float32)
    end_vote_s = pd.Series(end_vote).rolling(
        window=smooth_w, center=True, min_periods=1
    ).mean().values.astype(np.float32)
    segment_vote_s = pd.Series(segment_vote).rolling(
        window=smooth_w, center=True, min_periods=1
    ).mean().values.astype(np.float32)

    def select_peaks(order, vote_arr, min_gap, max_peaks=16):
        peaks = []
        for idx in order:
            idx = int(idx)
            if vote_arr[idx] <= 0:
                break
            if all(abs(idx - p) >= min_gap for p in peaks):
                peaks.append(idx)
            if len(peaks) >= max_peaks:
                break
        return peaks

    start_order = np.argsort(start_vote_s)[::-1]
    end_order = np.argsort(end_vote_s)[::-1]
    min_peak_gap = max(8, int(0.12 * target_len_int))
    start_peaks = select_peaks(start_order, start_vote_s, min_peak_gap, max_peaks=16)
    end_peaks = select_peaks(end_order, end_vote_s, min_peak_gap, max_peaks=16)

    boundary_rows = []
    for s in start_peaks:
        for e in end_peaks:
            if e <= s:
                continue
            L = e - s + 1
            len_ratio = L / max(target_len_mean, 1e-6)
            if len_ratio < 0.55 or len_ratio > 1.65:
                continue
            len_pen = abs(L - target_len_mean) / max(target_len_mean, 1e-6)
            sv = float(start_vote_s[s])
            ev = float(end_vote_s[e])
            occ = float(np.mean(segment_vote_s[s:e + 1]))
            local_support_df = vote_df[
                (np.abs(vote_df["start"] - s) <= int(0.22 * target_len_int)) &
                (np.abs(vote_df["end"] - e) <= int(0.22 * target_len_int))
            ]
            if len(local_support_df) > 0:
                local_support = float(local_support_df["vote_weight"].sum())
                local_pool = float(local_support_df["pool_score"].mean())
                local_sim = float(local_support_df["sim_mean"].mean())
                local_retr = float(local_support_df["retrieval_score"].mean())
            else:
                local_support = 0.0
                local_pool = 0.0
                local_sim = 0.0
                local_retr = 0.0
            bscore = float(boundary_score(test_X, s, e))
            lscore = float(local_consistency_score(test_X, s, e))
            boundary_rows.append({
                "start": s, "end": e, "len": L,
                "start_vote": sv, "end_vote": ev, "occupancy_vote": occ,
                "local_support": local_support, "local_pool": local_pool,
                "local_sim": local_sim, "local_retr": local_retr,
                "len_pen": len_pen, "boundary_score": bscore, "local_score": lscore,
            })

    boundary_df = pd.DataFrame(boundary_rows)
    if len(boundary_df) == 0:
        fallback_df = vote_df.copy()
        fallback_df["endpoint_weight"] = fallback_df["vote_weight"]
        ps = int(round(np.average(fallback_df["start"].values, weights=fallback_df["endpoint_weight"].values)))
        pe = int(round(np.average(fallback_df["end"].values, weights=fallback_df["endpoint_weight"].values)))
        ps = max(0, min(test_T - 1, ps))
        pe = max(0, min(test_T - 1, pe))
        if pe < ps:
            ps, pe = pe, ps
        boundary_best_score = np.nan
    else:
        boundary_df["start_vote_n"] = minmax_norm(boundary_df["start_vote"].values)
        boundary_df["end_vote_n"] = minmax_norm(boundary_df["end_vote"].values)
        boundary_df["occupancy_n"] = minmax_norm(boundary_df["occupancy_vote"].values)
        boundary_df["local_support_n"] = minmax_norm(boundary_df["local_support"].values)
        boundary_df["local_pool_n"] = minmax_norm(boundary_df["local_pool"].values)
        boundary_df["local_sim_n"] = minmax_norm(boundary_df["local_sim"].values)
        boundary_df["local_retr_n"] = minmax_norm(boundary_df["local_retr"].values)
        boundary_df["len_good_n"] = 1.0 - minmax_norm(boundary_df["len_pen"].values)
        boundary_df["boundary_n"] = minmax_norm(boundary_df["boundary_score"].values)
        boundary_df["local_score_n"] = minmax_norm(boundary_df["local_score"].values)
        boundary_df["boundary_vote_score"] = (
            0.18 * boundary_df["start_vote_n"] +
            0.18 * boundary_df["end_vote_n"] +
            0.16 * boundary_df["occupancy_n"] +
            0.16 * boundary_df["local_support_n"] +
            0.12 * boundary_df["len_good_n"] +
            0.07 * boundary_df["local_pool_n"] +
            0.05 * boundary_df["local_retr_n"] +
            0.04 * boundary_df["local_sim_n"] +
            0.02 * boundary_df["boundary_n"] +
            0.02 * boundary_df["local_score_n"]
        )
        best_b = boundary_df.sort_values(
            ["boundary_vote_score", "local_support_n", "occupancy_n"], ascending=False
        ).iloc[0]
        ps = int(best_b["start"])
        pe = int(best_b["end"])
        boundary_best_score = float(best_b["boundary_vote_score"])

    # 6) Local refinement
    refine_rows = []
    step = 4
    max_shift = 24
    for ds_shift in range(-max_shift, max_shift + 1, step):
        for de_shift in range(-max_shift, max_shift + 1, step):
            rs = ps + ds_shift
            re = pe + de_shift
            rs = max(0, min(test_T - 2, rs))
            re = max(rs + 1, min(test_T - 1, re))
            L = re - rs + 1
            len_ratio = L / max(target_len_mean, 1e-6)
            if len_ratio < 0.55 or len_ratio > 1.65:
                continue
            len_pen = abs(L - target_len_mean) / max(target_len_mean, 1e-6)
            sv = float(start_vote_s[rs])
            ev = float(end_vote_s[re])
            occ = float(np.mean(segment_vote_s[rs:re + 1]))
            local_support_df = vote_df[
                (np.abs(vote_df["start"] - rs) <= int(0.22 * target_len_int)) &
                (np.abs(vote_df["end"] - re) <= int(0.22 * target_len_int))
            ]
            local_support = float(local_support_df["vote_weight"].sum()) if len(local_support_df) else 0.0
            refine_rows.append({
                "start": rs, "end": re, "len_pen": len_pen,
                "start_vote": sv, "end_vote": ev, "occupancy": occ,
                "local_support": local_support,
                "boundary": float(boundary_score(test_X, rs, re)),
                "local": float(local_consistency_score(test_X, rs, re)),
            })

    refine_df = pd.DataFrame(refine_rows)
    if len(refine_df) > 0:
        refine_df["start_vote_n"] = minmax_norm(refine_df["start_vote"].values)
        refine_df["end_vote_n"] = minmax_norm(refine_df["end_vote"].values)
        refine_df["occupancy_n"] = minmax_norm(refine_df["occupancy"].values)
        refine_df["support_n"] = minmax_norm(refine_df["local_support"].values)
        refine_df["len_good_n"] = 1.0 - minmax_norm(refine_df["len_pen"].values)
        refine_df["boundary_n"] = minmax_norm(refine_df["boundary"].values)
        refine_df["local_n"] = minmax_norm(refine_df["local"].values)
        refine_df["refine_vote_score"] = (
            0.20 * refine_df["start_vote_n"] +
            0.20 * refine_df["end_vote_n"] +
            0.18 * refine_df["occupancy_n"] +
            0.18 * refine_df["support_n"] +
            0.14 * refine_df["len_good_n"] +
            0.06 * refine_df["boundary_n"] +
            0.04 * refine_df["local_n"]
        )
        best_r = refine_df.sort_values(
            ["refine_vote_score", "support_n", "occupancy_n"], ascending=False
        ).iloc[0]
        ps = int(best_r["start"])
        pe = int(best_r["end"])
        boundary_best_score = float(best_r["refine_vote_score"])

    selected_iou = iou_score(ps, pe, ts, te)
    weighted_start = ps
    weighted_end = pe
    weighted_iou = selected_iou

    cand_row = verify_df.sort_values(
        ["boundary_pre_score", "pool_score", "sim_mean"], ascending=False
    ).iloc[0]
    cand_ps = int(cand_row["start"])
    cand_pe = int(cand_row["end"])
    cand_iou = iou_score(cand_ps, cand_pe, ts, te)

    robust_count = len(vote_df)
    robust_start_mean = float(np.mean(vote_df["start"].values))
    robust_end_mean = float(np.mean(vote_df["end"].values))

    final_boundary_score = boundary_best_score
    final_num_start_peaks = len(start_peaks)
    final_num_end_peaks = len(end_peaks)
    final_chosen_cluster_size = len(cluster_vote_df)
    preboundary_score_alias = final_boundary_score
    preboundary_num_start_peaks_alias = final_num_start_peaks
    preboundary_num_end_peaks_alias = final_num_end_peaks

    topn_oracle_row = top_df.loc[top_df["oracle_iou"].idxmax()]
    topn_oracle_iou = float(topn_oracle_row["oracle_iou"])
    topn_oracle_start = int(topn_oracle_row["start"])
    topn_oracle_end = int(topn_oracle_row["end"])

    if VERBOSE_TABLES and len(cluster_df) > 0:
        print("Cluster scores:")
        print(cluster_df.head(12)[[
            "cluster_id", "cluster_score", "n",
            "center_mean", "center_std",
            "max_pool", "mean_pool",
            "max_retr", "mean_retr",
            "max_sim", "mean_sim",
            "mean_len_good",
            "compactness",
            "endpoint_compactness"
        ]])

    if VERBOSE_TABLES:
        print("Top Boundary-Voting candidates:")
        print(verify_df.sort_values(
        ["boundary_pre_score", "pool_score", "sim_mean"],
        ascending=False
    ).head(40)[[
        "start", "end", "len",
        "sim_max", "sim_mean",
        "len_penalty", "pos_penalty",
        "retrieval_score",
        "pool_score",
        "pred_iou",
        "boundary_pre_score",
        "vote_weight",
        "oracle_iou"
    ]])

    if VERBOSE_TABLES and len(boundary_df) > 0:
        print("Top Boundary hypotheses:")
        print(boundary_df.sort_values(
            ["boundary_vote_score", "local_support_n", "occupancy_n"],
            ascending=False
        ).head(20)[[
            "start", "end", "len",
            "start_vote", "end_vote", "occupancy_vote",
            "local_support", "len_pen",
            "boundary_vote_score"
        ]])

    print(f"[BOUNDARY-SET ORACLE] {topn_oracle_start}-{topn_oracle_end} | IoU={topn_oracle_iou:.4f}")
    print(f"[PRE-BOUNDARY] Pred: {cand_ps}-{cand_pe} | GT: {ts}-{te} | IoU={cand_iou:.4f}")
    print(f"[FINAL CLUSTER+BOUNDARY] Pred: {ps}-{pe} | GT: {ts}-{te} | IoU={selected_iou:.4f}")
    print(f"[FINAL DEBUG] clusters={num_clusters}, chosen_cluster={chosen_cluster_id}, cluster_size={final_chosen_cluster_size}, voted_candidates={robust_count}, start_peaks={final_num_start_peaks}, end_peaks={final_num_end_peaks}, boundary_score={final_boundary_score:.4f}, verify_top_m={VERIFY_TOP_M}")
    print(f"Selection Loss = {oracle_pool_iou - selected_iou:.4f}")

    exp_elapsed = time.perf_counter() - exp_t0
    log(f"EXP {test_id:02d} DONE | IoU={selected_iou:.4f} | elapsed={fmt_sec(exp_elapsed)} | total={fmt_sec(time.perf_counter() - RUN_T0)}")

    results.append({
        "Experiment": test_id,
        "Pred_Start": ps,
        "Pred_End": pe,
        "True_Start": ts,
        "True_End": te,
        "IoU": selected_iou,

        "PreBoundary_Start": cand_ps,
        "PreBoundary_End": cand_pe,
        "PreBoundary_IoU": cand_iou,

        "Weighted_Start": weighted_start,
        "Weighted_End": weighted_end,
        "Weighted_IoU": weighted_iou,
        "PreBoundary_Candidate_Start": cand_ps,
        "PreBoundary_Candidate_End": cand_pe,
        "PreBoundary_Candidate_IoU": cand_iou,
        "PreBoundary_Score": preboundary_score_alias,
        "PreBoundary_Num_Start_Peaks": preboundary_num_start_peaks_alias,
        "PreBoundary_Num_End_Peaks": preboundary_num_end_peaks_alias,
        "Final_Chosen_Cluster_ID": chosen_cluster_id,
        "Final_Num_Clusters": num_clusters,
        "Final_Best_Cluster_Score": best_cluster_score,
        "Final_Chosen_Cluster_Size": final_chosen_cluster_size,
        "Robust_Candidate_Count": robust_count,
        "Robust_Start_Mean": robust_start_mean,
        "Robust_End_Mean": robust_end_mean,
        "Chosen_Cluster_ID": chosen_cluster_id,
        "Num_Clusters": num_clusters,
        "Best_Cluster_Score": best_cluster_score,

        "TopN_Oracle_Start": topn_oracle_start,
        "TopN_Oracle_End": topn_oracle_end,
        "TopN_Oracle_IoU": topn_oracle_iou,

        "Oracle_ALL_Start": oracle_all_start,
        "Oracle_ALL_End": oracle_all_end,
        "Oracle_ALL_IoU": oracle_all_iou,

        "Oracle_POOL_Start": oracle_pool_start,
        "Oracle_POOL_End": oracle_pool_end,
        "Oracle_POOL_IoU": oracle_pool_iou,

        "Pool_Loss": oracle_all_iou - oracle_pool_iou,
        "Selection_Loss": oracle_pool_iou - selected_iou,
        "Pool_Size": len(pool_indices),
        "Experiment_Time_Sec": exp_elapsed,
        "Experiment_Tag": EXPERIMENT_TAG,
        "Handcrafted_Feature_Mode": HANDCRAFTED_FEATURE_MODE,
        "Retrieval_Score_Mode": RETRIEVAL_SCORE_MODE,
        "Boundary_Pre_Score_Mode": BOUNDARY_PRE_SCORE_MODE,
        "Cluster_Score_Mode": CLUSTER_SCORE_MODE,
        "Verify_Candidate_Mode": VERIFY_CANDIDATE_MODE,
        "Verify_Top_M": VERIFY_TOP_M,
    })

    timing_rows.append({
        "Experiment": test_id,
        "Status": "ok",
        "Experiment_Time_Sec": exp_elapsed,
        "Experiment_Tag": EXPERIMENT_TAG,
        "Handcrafted_Feature_Mode": HANDCRAFTED_FEATURE_MODE,
        "Retrieval_Score_Mode": RETRIEVAL_SCORE_MODE,
        "Boundary_Pre_Score_Mode": BOUNDARY_PRE_SCORE_MODE,
        "Cluster_Score_Mode": CLUSTER_SCORE_MODE,
        "Verify_Candidate_Mode": VERIFY_CANDIDATE_MODE,
        "Verify_Top_M": VERIFY_TOP_M,
        "Experiment_Time": fmt_sec(exp_elapsed),
        "Cumulative_Time_Sec": time.perf_counter() - RUN_T0,
        "Cumulative_Time": fmt_sec(time.perf_counter() - RUN_T0),
        "IoU": selected_iou,
        "Pool_Oracle_IoU": oracle_pool_iou,
        "Selection_Loss": oracle_pool_iou - selected_iou,
        "Pool_Size": len(pool_indices),
        "Num_Clusters": num_clusters,
        "Chosen_Cluster_ID": chosen_cluster_id,
    })


# =========================================
# FINAL REPORT
# =========================================
df = pd.DataFrame(results)
timing_df = pd.DataFrame(timing_rows)

print("\n===== FINAL =====")
print(df)

valid_df = df.dropna(subset=["IoU"])
total_elapsed = time.perf_counter() - RUN_T0

summary = {
    "Mean_Selected_IoU": float(valid_df["IoU"].mean()) if len(valid_df) else np.nan,
    "Mean_PreBoundary_IoU": float(valid_df["PreBoundary_IoU"].mean()) if len(valid_df) else np.nan,
    "Mean_Final_IoU": float(valid_df["IoU"].mean()) if len(valid_df) else np.nan,
    "Mean_PreBoundary_Candidate_IoU": float(valid_df["PreBoundary_Candidate_IoU"].mean()) if len(valid_df) else np.nan,
    "Mean_TopN_Oracle_IoU": float(valid_df["TopN_Oracle_IoU"].mean()) if len(valid_df) else np.nan,
    "Mean_Oracle_POOL_IoU": float(valid_df["Oracle_POOL_IoU"].mean()) if len(valid_df) else np.nan,
    "Mean_Oracle_ALL_IoU": float(valid_df["Oracle_ALL_IoU"].mean()) if len(valid_df) else np.nan,
    "Mean_Pool_Loss": float(valid_df["Pool_Loss"].mean()) if len(valid_df) else np.nan,
    "Mean_Selection_Loss": float(valid_df["Selection_Loss"].mean()) if len(valid_df) else np.nan,
    "Valid_Experiments": int(len(valid_df)),
    "Total_Time_Sec": float(total_elapsed),
    "Total_Time": fmt_sec(total_elapsed),
    "Experiment_Tag": EXPERIMENT_TAG,
    "Handcrafted_Feature_Mode": HANDCRAFTED_FEATURE_MODE,
    "Retrieval_Score_Mode": RETRIEVAL_SCORE_MODE,
    "Boundary_Pre_Score_Mode": BOUNDARY_PRE_SCORE_MODE,
    "Cluster_Score_Mode": CLUSTER_SCORE_MODE,
    "Verify_Candidate_Mode": VERIFY_CANDIDATE_MODE,
}
summary_df = pd.DataFrame([summary])

print("\n===== SUMMARY =====")
for k, v in summary.items():
    print(f"{k}: {v}")

# Persist reproducible reports.
results_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_results.csv"
summary_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.csv"
timing_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_timing.csv"
df.to_csv(results_path, index=False)
summary_df.to_csv(summary_path, index=False)
timing_df.to_csv(timing_path, index=False)

log(f"Saved results: {results_path}")
log(f"Saved summary: {summary_path}")
log(f"Saved timing : {timing_path}")
log(f"Total elapsed: {fmt_sec(total_elapsed)}")

print("\n--- DIAGNOSIS ---")

for _, r in valid_df.iterrows():
    exp_id = int(r["Experiment"])
    selected = r["IoU"]
    cand = r["PreBoundary_IoU"]
    weighted = r["Weighted_IoU"]
    topn = r["TopN_Oracle_IoU"]
    pool = r["Oracle_POOL_IoU"]
    all_iou = r["Oracle_ALL_IoU"]

    if all_iou < 0.5:
        reason = "Candidate generation / full search space problem"
    elif pool < 0.5:
        reason = "Retrieval pool problem"
    elif topn < 0.5:
        reason = "Top-N filtering problem"
    elif selected < 0.5:
        reason = "Weighted endpoint / consensus problem"
    else:
        reason = "OK"

    print(
        f"EXP {exp_id:02d} | "
        f"Selected={selected:.3f} | "
        f"PreBoundary={cand:.3f} | "
        f"Weighted={weighted:.3f} | "
        f"TopN Oracle={topn:.3f} | "
        f"Pool Oracle={pool:.3f} | "
        f"All Oracle={all_iou:.3f} | "
        f"{reason}"
    )
