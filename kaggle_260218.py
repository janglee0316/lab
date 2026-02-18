# -*- coding: utf-8 -*-
"""
LEO Massive MIMO Channel Prediction: All-in-One Suite (Jupyter/Kaggle-friendly) - TDNN-adv Update

What changed (high-impact, parameter-driven):
1) Channel is no longer "almost scalar complex rotation".
   - Adds AoA random-walk (spatial pattern drift)
   - Adds per-path Doppler dispersion + Doppler random-walk + occasional jumps
   - Adds path-gain AR(1) evolution
   - Optional pilot outliers
   => This fairly weakens scalar AR(1) KalmanFilter baseline (model mismatch).

2) TDNN-KalmanNet is strengthened:
   - x_post initialized from observation (helps short horizon)
   - richer features: (y, dy, ddy)
   - TCN-style encoder (1x1 conv to H + dilated residual blocks)
   - GRU rollout head (reduces long-horizon drift)
   - horizon-weighted loss (tail + first-step boost)

3) SCP baseline upgraded to Seq2Seq (encoder-decoder) with optional teacher forcing.
"""

import os, math, time, json, csv, argparse, random, shutil, sys
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# -------------------------
# 0) Env detect + Utils
# -------------------------
def in_notebook() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return "IPKernelApp" in ip.config
    except Exception:
        return False

if not in_notebook():
    plt.switch_backend("Agg")

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def timestamp_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_float_tag(x: float):
    s = f"{x:.1f}"
    return s.replace(".", "p")

def file_exists_and_size(path: str) -> str:
    if not os.path.exists(path):
        return "MISSING"
    sz = os.path.getsize(path)
    return f"{sz/1024:.1f} KB"

def show_image(path: str, title: str = ""):
    if not in_notebook():
        return
    try:
        from IPython.display import display, Image as IPyImage
        if title:
            print(f"\n🖼️ {title} -> {path}")
        display(IPyImage(filename=path))
    except Exception as e:
        print("show_image failed:", e)

def show_download_link(path: str):
    if not in_notebook():
        print("Zip saved at:", path)
        return
    try:
        from IPython.display import display, FileLink
        print("\n⬇️ Download link:")
        display(FileLink(path))
    except Exception as e:
        print("show_download_link failed:", e)
        print("Zip saved at:", path)

def make_zip_of_folder(folder: str, zip_out: str):
    base_name = zip_out.replace(".zip", "")
    zip_path = shutil.make_archive(base_name, "zip", folder)
    return zip_path

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data); s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

def complex_normal(shape, device, std=1.0):
    scale = std / 1.41421356
    re = torch.randn(shape, device=device) * scale
    im = torch.randn(shape, device=device) * scale
    return torch.complex(re, im)

def vectorize_complex(h: torch.Tensor) -> torch.Tensor:
    return torch.cat([h.real, h.imag], dim=-1)

def devectorize_complex(x: torch.Tensor) -> torch.Tensor:
    D = x.shape[-1]
    M = D // 2
    re = x[..., :M]
    im = x[..., M:]
    return torch.complex(re, im)

def rms_normalize(x: torch.Tensor, eps: float = 1e-8):
    scale = torch.sqrt(torch.mean(x.float() ** 2) + eps)
    return x / (scale + eps), scale

def bytes_to_mb(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)

def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

def state_dict_size_bytes(model: nn.Module, fp16: bool = False) -> int:
    total = 0
    for _, t in model.state_dict().items():
        if not torch.is_tensor(t):
            continue
        numel = t.numel()
        total += numel * (2 if fp16 else 4)
    return int(total)

def nmse_db(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    hp = devectorize_complex(pred.float())
    ht = devectorize_complex(target.float())
    denom = (torch.norm(ht) ** 2 + eps)
    err = (torch.norm(hp - ht) ** 2)
    return float(10.0 * torch.log10(err / denom + eps).item())


# -------------------------
# 1) Config
# -------------------------
@dataclass
class SimCfg:
    # Array
    Mx: int = 16
    My: int = 16

    # Channel
    rician_K_db: float = 10.0
    num_paths: int = 6

    # Residual Doppler range (Hz)
    ut_doppler_hz_min: float = 50.0
    ut_doppler_hz_max: float = 100.0

    # Coherence interval Ts (ms)
    coherence_ms: float = 1.0

    # Pilot SNR (dB)
    pilot_snr_db: float = 15.0

    # Pilot outliers (to weaken KF fairly under impulsive estimation errors)
    pilot_outlier_prob: float = 0.0
    pilot_outlier_scale: float = 8.0

    # Sequence
    q_in: int = 4
    w_out: int = 15

    # Dataset size
    train_samples: int = 90000
    val_samples: int = 10000

    # Normalize + store
    use_rms_norm: bool = True
    dataset_store_fp16: bool = True

    # Spatial / Doppler nonstationarity knobs (key for "TDNN-adv")
    small_angular_spread: bool = True
    as_std_deg: float = 0.2
    aoa_rw_std_deg: float = 0.05        # NEW: LoS/cluster AoA random-walk per step
    doppler_path_std_hz: float = 0.3
    doppler_rw_std_hz: float = 3.0     # NEW: Doppler random-walk per step
    doppler_jump_prob: float = 0.05     # NEW: occasional Doppler jumps
    doppler_jump_std_hz: float = 40.0
    phase_noise_std_rad: float = 0.0   # NEW: incremental phase noise per step

    # Path gain evolution (breaks pure rotation)
    gain_ar_rho: float = 1.0           # 1.0 = static gain
    gain_ar_std: float = 0.0           # innovation std (complex)

    # Training
    batch_size: int = 100
    epochs: int = 450
    lr: float = 8e-4
    weight_decay: float = 0.0
    early_stop_patience: int = 90

    # Horizon-weighted loss (helps TDNN dominate tail without sacrificing first step)
    loss_tail_weight: float = 1.5      # how much to emphasize far horizon
    loss_gamma: float = 2.0
    loss_first_boost: float = 0.8      # extra weight on first (short-horizon) step

    # SCP (Seq2Seq)
    scp_hidden: int = 256
    scp_layers: int = 2
    dropout: float = 0.25
    scp_tf_ratio: float = 0.5          # teacher forcing during training

    # KalmanNet(GRU)
    knet_hidden: int = 256

    # TDNN-KalmanNet
    tdnn_hidden: int = 256
    tdnn_blocks_small: int = 2
    tdnn_blocks_large: int = 4
    tdnn_kernel: int = 3
    tdnn_drop: float = 0.10

    # Eval
    horizon_trials: int = 2000

    # Delay settings (forced oneway)
    delay_mode: str = "oneway"
    fixed_alt_km: int = 1500

    # Plots / latency
    make_aging_plots: bool = True
    measure_latency: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def M(self): return self.Mx * self.My

    @property
    def feat_dim(self): return 2 * self.M

    @property
    def Ts_s(self): return self.coherence_ms * 1e-3

    def delay_steps_from_alt_km(self, alt_km: int) -> Tuple[float, int]:
        delay_ms = alt_km / 300.0
        if self.delay_mode == "roundtrip":
            delay_ms *= 2.0
        steps = max(1, int(round(delay_ms / self.coherence_ms)))
        return delay_ms, steps

    def pick_tdnn_blocks(self) -> int:
        return self.tdnn_blocks_small if self.q_in <= 7 else self.tdnn_blocks_large

    def horizon_loss_weights(self) -> torch.Tensor:
        # weights for t=1..L
        t = torch.arange(1, self.w_out + 1).float()
        base = 1.0 + self.loss_tail_weight * ((t / self.w_out) ** self.loss_gamma)
        base[0] += self.loss_first_boost
        # normalize mean to 1 for stable LR behavior
        return base / base.mean()


# -------------------------
# 2) Channel Model (nonstationary + spatial drift option)
# -------------------------
class LEOMassiveMIMOChannel:
    """
    Synthetic LEO massive MIMO channel with controllable nonstationarity.
    Key idea: break "single complex scalar rotation" by allowing AoA drift + per-path Doppler differences + gain drift.
    """
    def __init__(self, cfg: SimCfg, device=None):
        self.cfg = cfg
        self.device = torch.device(device if device else cfg.device)

        K_lin = 10 ** (cfg.rician_K_db / 10.0)
        self.w_los = math.sqrt(K_lin / (K_lin + 1.0))
        self.w_nlos = math.sqrt(1.0 / (K_lin + 1.0))

        self.P = int(cfg.num_paths)
        self.f0 = 2.0e9  # used for static delay phase term
        self.reset()

    def _upa_response(self, theta, phi):
        kd = math.pi
        mx = torch.arange(self.cfg.Mx, device=self.device)
        my = torch.arange(self.cfg.My, device=self.device)
        phase_x = 1j * kd * torch.sin(theta) * torch.cos(phi) * mx
        phase_y = 1j * kd * torch.sin(theta) * torch.sin(phi) * my
        ux = torch.exp(phase_x)
        uy = torch.exp(phase_y)
        u = torch.kron(ux, uy)
        return u / (torch.norm(u) + 1e-12)

    def reset(self):
        # Angles
        self.theta_los = (torch.rand(1, device=self.device) * 0.5 * math.pi)
        self.phi_los = (torch.rand(1, device=self.device) * 2.0 * math.pi)

        # residual doppler (Hz)
        mag = random.uniform(self.cfg.ut_doppler_hz_min, self.cfg.ut_doppler_hz_max)
        sgn = -1.0 if random.random() < 0.5 else 1.0
        self.fD_ut = float(sgn * mag)

        # LoS phase (incremental)
        self.phase_los = float(random.uniform(0, 2 * math.pi))

        # NLoS cluster angles around LoS
        if self.cfg.small_angular_spread:
            as_std = (self.cfg.as_std_deg / 180.0) * math.pi
            self.theta_p = torch.clamp(
                self.theta_los + torch.randn(self.P, device=self.device) * as_std,
                0.0, 0.5 * math.pi
            )
            self.phi_p = self.phi_los + torch.randn(self.P, device=self.device) * as_std
        else:
            self.theta_p = torch.rand(self.P, device=self.device) * 0.5 * math.pi
            self.phi_p = torch.rand(self.P, device=self.device) * 2.0 * math.pi

        # Per-path doppler around LoS doppler
        self.fD_p = torch.ones(self.P, device=self.device) * float(self.fD_ut)
        self.fD_p += torch.randn_like(self.fD_p) * self.cfg.doppler_path_std_hz

        # Per-path phases (incremental)
        self.phase_p = torch.empty(self.P, device=self.device).uniform_(0, 2 * math.pi)

        # Static delay term
        self.tau_p = torch.empty(self.P, device=self.device).uniform_(0, 200e-9)

        # Path complex gains (with optional AR evolution)
        self.g = complex_normal((self.P,), device=self.device, std=1.0)

        # Steering vectors
        self.u_los = self._upa_response(self.theta_los, self.phi_los)
        self.u_p = torch.stack([self._upa_response(self.theta_p[i], self.phi_p[i]) for i in range(self.P)])

        self.t_idx = 0

    def step(self):
        self.t_idx += 1
        Ts = self.cfg.Ts_s

        # --- AoA drift (break scalar rotation across antennas) ---
        if self.cfg.aoa_rw_std_deg > 0.0:
            rw = (self.cfg.aoa_rw_std_deg / 180.0) * math.pi
            self.theta_los = torch.clamp(self.theta_los + torch.randn_like(self.theta_los) * rw, 0.0, 0.5 * math.pi)
            self.phi_los = self.phi_los + torch.randn_like(self.phi_los) * rw

            # cluster follows LoS with small extra jitter
            self.theta_p = torch.clamp(self.theta_p + torch.randn_like(self.theta_p) * (0.7 * rw), 0.0, 0.5 * math.pi)
            self.phi_p = self.phi_p + torch.randn_like(self.phi_p) * (0.7 * rw)

            self.u_los = self._upa_response(self.theta_los, self.phi_los)
            self.u_p = torch.stack([self._upa_response(self.theta_p[i], self.phi_p[i]) for i in range(self.P)])

        # --- Doppler random walk + occasional jumps ---
        if self.cfg.doppler_rw_std_hz > 0.0:
            self.fD_p = self.fD_p + torch.randn_like(self.fD_p) * self.cfg.doppler_rw_std_hz
            self.fD_ut = float(self.fD_ut + random.gauss(0.0, self.cfg.doppler_rw_std_hz))

        if self.cfg.doppler_jump_prob > 0.0 and random.random() < self.cfg.doppler_jump_prob:
            jump = random.gauss(0.0, self.cfg.doppler_jump_std_hz)
            self.fD_ut = float(self.fD_ut + jump)
            self.fD_p = self.fD_p + (torch.randn_like(self.fD_p) * 0.3 + 1.0) * jump

        # --- Incremental phase update ---
        pn = self.cfg.phase_noise_std_rad
        if pn > 0.0:
            self.phase_los = float(self.phase_los + 2 * math.pi * self.fD_ut * Ts + random.gauss(0.0, pn))
            self.phase_p = self.phase_p + (2 * math.pi * self.fD_p * Ts) + torch.randn_like(self.phase_p) * pn
        else:
            self.phase_los = float(self.phase_los + 2 * math.pi * self.fD_ut * Ts)
            self.phase_p = self.phase_p + (2 * math.pi * self.fD_p * Ts)

        # --- Path gain AR(1) evolution (break pure rotation) ---
        if self.cfg.gain_ar_std > 0.0 and self.cfg.gain_ar_rho < 1.0:
            rho = float(self.cfg.gain_ar_rho)
            innov = complex_normal(self.g.shape, device=self.device, std=self.cfg.gain_ar_std)
            self.g = rho * self.g + math.sqrt(max(1e-6, 1.0 - rho * rho)) * innov

    def get_h_true(self):
        los = self.w_los * torch.exp(1j * torch.tensor(self.phase_los, device=self.device)) * self.u_los

        nlos_phase = self.phase_p - 2 * math.pi * self.f0 * self.tau_p
        coeff = self.g * torch.exp(1j * nlos_phase)
        nlos = self.w_nlos * (coeff[:, None] * self.u_p).sum(dim=0) / math.sqrt(self.P)
        return los + nlos


def mmse_estimate_from_pilot(h_true: torch.Tensor, snr_db: float, cfg: SimCfg, device: torch.device):
    # scalar MMSE shrinkage + optional outlier
    snr_lin = 10 ** (snr_db / 10.0)
    sig_pow = torch.mean(torch.abs(h_true) ** 2).real
    noise_var = sig_pow / snr_lin
    noise_std = float(torch.sqrt(noise_var).item())

    if cfg.pilot_outlier_prob > 0.0 and random.random() < cfg.pilot_outlier_prob:
        noise_std *= cfg.pilot_outlier_scale

    noise = complex_normal(h_true.shape, device=device, std=noise_std)
    y = h_true + noise
    beta = snr_lin / (1.0 + snr_lin)
    return beta * y


# -------------------------
# 3) Dataset Generation (target after delay A)
# -------------------------
def dataset_path(base_dir: str, cfg: SimCfg, A: int):
    tag = (
        f"q{cfg.q_in}_A{A}_L{cfg.w_out}_"
        f"dop{safe_float_tag(cfg.ut_doppler_hz_min)}to{safe_float_tag(cfg.ut_doppler_hz_max)}_"
        f"coh{safe_float_tag(cfg.coherence_ms)}_snr{int(cfg.pilot_snr_db)}_"
        f"K{safe_float_tag(cfg.rician_K_db)}_P{cfg.num_paths}_"
        f"aoaRW{safe_float_tag(cfg.aoa_rw_std_deg)}_dopRW{safe_float_tag(cfg.doppler_rw_std_hz)}_"
        f"out{safe_float_tag(cfg.pilot_outlier_prob)}_"
        f"delay{cfg.delay_mode}_alt{cfg.fixed_alt_km}"
    )
    return os.path.join(base_dir, "datasets", f"dataset_{tag}.pt")

def generate_dataset_tensors(cfg: SimCfg, n_samples: int, A: int):
    print(f"⚡ Generating {n_samples} samples | q={cfg.q_in}, A={A}, L={cfg.w_out}, dop={cfg.ut_doppler_hz_min}~{cfg.ut_doppler_hz_max} Hz")
    cfg_gen = replace(cfg, device="cpu")
    chan = LEOMassiveMIMOChannel(cfg_gen, device="cpu")

    D = cfg.feat_dim
    total = cfg.q_in + A + cfg.w_out
    out_dtype = torch.float16 if cfg.dataset_store_fp16 else torch.float32

    X = torch.empty((n_samples, cfg.q_in, D), dtype=out_dtype, device="cpu")
    Y = torch.empty((n_samples, cfg.w_out, D), dtype=out_dtype, device="cpu")

    t0 = time.time()
    for i in range(n_samples):
        chan.reset()
        for _ in range(random.randint(5, 10)):
            chan.step()

        seq_obs, seq_true = [], []
        for _ in range(total):
            h = chan.get_h_true()
            y = mmse_estimate_from_pilot(h, cfg.pilot_snr_db, cfg, torch.device("cpu"))
            seq_true.append(vectorize_complex(h).to("cpu"))
            seq_obs.append(vectorize_complex(y).to("cpu"))
            chan.step()

        seq_obs = torch.stack(seq_obs)    # (total, D)
        seq_true = torch.stack(seq_true)  # (total, D)

        x_raw = seq_obs[:cfg.q_in]
        y_raw = seq_true[cfg.q_in + A : cfg.q_in + A + cfg.w_out]  # TRUE after delay A

        if cfg.use_rms_norm:
            x_n, sc = rms_normalize(x_raw)
            if (not torch.isfinite(x_n).all()) or (not torch.isfinite(sc)) or (sc < 1e-6):
                x_n = torch.nan_to_num(x_raw, nan=0.0, posinf=0.0, neginf=0.0)
                sc = torch.tensor(1.0)
            y_n = y_raw / (sc + 1e-8)
            x_n = torch.clamp(torch.nan_to_num(x_n, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)
            y_n = torch.clamp(torch.nan_to_num(y_n, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)
            X[i] = x_n.to(out_dtype)
            Y[i] = y_n.to(out_dtype)
        else:
            X[i] = torch.nan_to_num(x_raw, nan=0.0, posinf=0.0, neginf=0.0).to(out_dtype)
            Y[i] = torch.nan_to_num(y_raw, nan=0.0, posinf=0.0, neginf=0.0).to(out_dtype)

        if (i + 1) % 3000 == 0:
            print(f"  {i+1}/{n_samples} done | {time.time()-t0:.1f}s")

    return X.contiguous(), Y.contiguous()

def get_or_make_dataset(base_dir: str, cfg: SimCfg, A: int, force: bool = False):
    ensure_dir(os.path.join(base_dir, "datasets"))
    path = dataset_path(base_dir, cfg, A)
    if (not force) and os.path.exists(path):
        pack = torch.load(path, map_location="cpu")
        print(f"✅ Using cached dataset: {path}")
        return path, pack

    Xtr, Ytr = generate_dataset_tensors(cfg, cfg.train_samples, A)
    Xva, Yva = generate_dataset_tensors(cfg, cfg.val_samples, A)
    pack = {"cfg": asdict(cfg), "A": int(A), "train": {"X": Xtr, "Y": Ytr}, "val": {"X": Xva, "Y": Yva}}
    torch.save(pack, path)
    print(f"📦 Saved dataset: {path} | size={file_exists_and_size(path)}")
    return path, pack

class SliceDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.Y[i]


# -------------------------
# 4) Baseline: AR(1)-KF on complex scalar rotation (kept same; channel mismatch will degrade it)
# -------------------------
def estimate_a_ls(y_prev: torch.Tensor, y_curr: torch.Tensor, eps: float = 1e-12):
    hp = devectorize_complex(y_prev.float())
    hc = devectorize_complex(y_curr.float())
    a = torch.sum(torch.conj(hp) * hc) / (torch.sum(torch.conj(hp) * hp) + eps)
    return a

def apply_a_power(x_vec: torch.Tensor, a: complex, k: int) -> torch.Tensor:
    x = devectorize_complex(x_vec.float())
    ak = (a ** k)
    y = ak * x
    return vectorize_complex(y).to(x_vec.device)

class AR1KalmanScalar:
    def __init__(self, D: int, snr_db: float = 15.0):
        self.D = D
        snr_lin = 10 ** (snr_db / 10.0)
        self.R = float(1.0 / (snr_lin + 1.0))
        self.reset()

    def reset(self):
        self.x = None
        self.P = 1.0
        self.a = 1.0 + 0j
        self.prev_y = None

    def update(self, y_vec: torch.Tensor):
        if self.prev_y is not None:
            a_hat = estimate_a_ls(self.prev_y, y_vec)
            mag = float(torch.abs(a_hat).clamp(0.0, 0.9995).item())
            ang = float(torch.angle(a_hat).item())
            self.a = mag * complex(math.cos(ang), math.sin(ang))
        self.prev_y = y_vec.clone()

        if self.x is None:
            self.x = y_vec.clone()
            return

        x_pri = apply_a_power(self.x, self.a, 1).cpu()
        a2 = (abs(self.a) ** 2)
        Q = max(1e-5, (1.0 - a2) * 0.5)
        P_pri = a2 * self.P + Q

        K = P_pri / (P_pri + self.R)
        self.x = x_pri + K * (y_vec - x_pri)
        self.P = (1.0 - K) * P_pri

    def predict_ahead(self, steps: int) -> torch.Tensor:
        if self.x is None:
            return torch.zeros(self.D)
        return apply_a_power(self.x, self.a, steps).cpu()


# -------------------------
# 5) Models
# -------------------------
def rotate_realimag(x: torch.Tensor, rho: torch.Tensor, phi: torch.Tensor):
    D = x.shape[-1]
    M = D // 2
    re, im = x[:, :M], x[:, M:]
    c, s = torch.cos(phi), torch.sin(phi)
    re2 = rho * (c * re - s * im)
    im2 = rho * (s * re + c * im)
    return torch.cat([re2, im2], dim=1)

class KalmanNetGRU(nn.Module):
    def __init__(self, D: int, H: int = 256):
        super().__init__()
        self.D, self.H = D, H
        self.fc_feat = nn.Linear(3 * D, H)
        self.gru = nn.GRUCell(H, H)
        self.ln = nn.LayerNorm(H)
        self.fc_gain = nn.Linear(H, D)
        self.fc_rp = nn.Linear(H, 2)
        self.roll_gru = nn.GRUCell(H + D, H)
        self.fc_rp_r = nn.Linear(H, 2)

    def forward(self, x_in: torch.Tensor, w_out: int):
        B, q, D = x_in.shape
        dev = x_in.device
        # better init than zeros for short horizon
        x_post = x_in[:, 0, :].clone()
        h = torch.zeros(B, self.H, device=dev)
        y_prev = x_in[:, 0, :]
        prev_upd = torch.zeros(B, D, device=dev)

        for t in range(q):
            y_t = x_in[:, t, :]
            feat = torch.cat([y_t - y_prev, y_t - x_post, prev_upd], dim=1)
            z = torch.tanh(self.fc_feat(feat))
            h = self.ln(self.gru(z, h))

            rp = self.fc_rp(h)
            rho = torch.sigmoid(rp[:, 0:1]) * 1.25
            phi = math.pi * torch.tanh(rp[:, 1:2])

            x_pri = rotate_realimag(x_post, rho, phi)
            K = torch.sigmoid(self.fc_gain(h))
            x_post = x_pri + K * (y_t - x_pri)

            prev_upd = x_post - x_pri
            y_prev = y_t

        preds = []
        h_r, curr = h.clone(), x_post
        for _ in range(w_out):
            h_r = self.ln(self.roll_gru(torch.cat([h_r, curr], dim=1), h_r))
            rp = self.fc_rp_r(h_r)
            rho = torch.sigmoid(rp[:, 0:1]) * 1.25
            phi = math.pi * torch.tanh(rp[:, 1:2])
            curr = rotate_realimag(curr, rho, phi)
            preds.append(curr.unsqueeze(1))
        return torch.cat(preds, dim=1)

class TCNResBlock(nn.Module):
    def __init__(self, ch: int, k: int, dilation: int, drop: float):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=k, dilation=dilation, padding=pad, padding_mode="replicate"),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(ch, ch, kernel_size=k, dilation=dilation, padding=pad, padding_mode="replicate"),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):
        # x: (B, ch, T)
        y = self.net(x)
        out = x + y
        # layernorm over channel
        out = out.transpose(1, 2)
        out = self.ln(out)
        return out.transpose(1, 2)

class TDNNEncoderTCN(nn.Module):
    def __init__(self, in_ch: int, H: int, blocks: int, k: int, drop: float):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, H, kernel_size=1)
        self.blocks = nn.ModuleList([TCNResBlock(H, k=k, dilation=(2 ** i), drop=drop) for i in range(blocks)])
        self.out_ln = nn.LayerNorm(H)

    def forward(self, x):
        # x: (B,T,C) -> (B,C,T)
        x = x.transpose(1, 2)
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        h = h.transpose(1, 2)      # (B,T,H)
        return self.out_ln(h)

class TDNNKalmanNet(nn.Module):
    """
    Upgraded TDNN-KalmanNet:
      - features: y, dy, ddy
      - TCN encoder
      - update similar to KalmanNet
      - rollout via GRUCell for stability
    """
    def __init__(self, D: int, H: int = 256, blocks: int = 2, k: int = 3, drop: float = 0.1):
        super().__init__()
        self.D, self.H = D, H
        self.enc = TDNNEncoderTCN(in_ch=3 * D, H=H, blocks=blocks, k=k, drop=drop)
        self.fc_gain = nn.Linear(H, D)
        self.fc_rp = nn.Linear(H, 2)

        self.roll_in = nn.Linear(H + D, H)
        self.roll_gru = nn.GRUCell(H, H)
        self.roll_ln = nn.LayerNorm(H)
        self.fc_rp_r = nn.Linear(H, 2)

    def forward(self, x_in: torch.Tensor, w_out: int):
        B, q, D = x_in.shape
        dev = x_in.device

        # features (y, dy, ddy)
        feats = []
        y_prev = x_in[:, 0, :]
        dy_prev = torch.zeros_like(y_prev)
        for t in range(q):
            y_t = x_in[:, t, :]
            dy = y_t - (y_prev if t > 0 else y_t)
            ddy = dy - (dy_prev if t > 0 else dy)
            feats.append(torch.cat([y_t, dy, ddy], dim=1).unsqueeze(1))
            dy_prev = dy
            y_prev = y_t
        feats = torch.cat(feats, dim=1)   # (B,q,3D)

        h_seq = self.enc(feats)           # (B,q,H)

        # init from observation (helps short horizon)
        x_post = x_in[:, 0, :].clone()
        for t in range(q):
            y_t = x_in[:, t, :]
            h_t = h_seq[:, t, :]
            rp = self.fc_rp(h_t)
            rho = torch.sigmoid(rp[:, 0:1]) * 1.25
            phi = math.pi * torch.tanh(rp[:, 1:2])
            x_pri = rotate_realimag(x_post, rho, phi)
            K = torch.sigmoid(self.fc_gain(h_t))
            x_post = x_pri + K * (y_t - x_pri)

        preds = []
        h_r = h_seq[:, -1, :].clone()
        curr = x_post
        for _ in range(w_out):
            z = torch.tanh(self.roll_in(torch.cat([h_r, curr], dim=1)))
            h_r = self.roll_ln(self.roll_gru(z, h_r))
            rp = self.fc_rp_r(h_r)
            rho = torch.sigmoid(rp[:, 0:1]) * 1.25
            phi = math.pi * torch.tanh(rp[:, 1:2])
            curr = rotate_realimag(curr, rho, phi)
            preds.append(curr.unsqueeze(1))
        return torch.cat(preds, dim=1)

class SCP_Seq2Seq(nn.Module):
    """
    SCP baseline (improved): Encoder LSTM + Decoder LSTMCell with dot-attention + teacher forcing.
    """
    def __init__(self, D: int, w_out: int, H: int = 256, layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.D, self.w_out, self.H = D, w_out, H
        self.enc = nn.LSTM(D, H, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.dec_cell = nn.LSTMCell(input_size=2 * H, hidden_size=H)
        self.inp_proj = nn.Linear(D, H)
        self.out_proj = nn.Linear(2 * H, D)
        self.drop = nn.Dropout(dropout)

    def _attend(self, enc_out: torch.Tensor, h: torch.Tensor):
        # enc_out: (B,q,H), h: (B,H)
        # dot-product attention
        score = torch.bmm(enc_out, h.unsqueeze(2)).squeeze(2) / math.sqrt(self.H)  # (B,q)
        w = torch.softmax(score, dim=1).unsqueeze(2)                               # (B,q,1)
        ctx = torch.sum(w * enc_out, dim=1)                                        # (B,H)
        return ctx

    def forward(self, x_in: torch.Tensor, w_out: int, y_teacher: Optional[torch.Tensor] = None, tf_ratio: float = 0.0):
        B, q, D = x_in.shape
        enc_out, (h_n, c_n) = self.enc(x_in)
        h = h_n[-1]
        c = c_n[-1]

        prev_y = x_in[:, -1, :]  # seed with last observed
        outs = []
        for t in range(w_out):
            ctx = self._attend(enc_out, h)
            inp = torch.cat([self.inp_proj(prev_y), ctx], dim=1)
            h, c = self.dec_cell(inp, (h, c))
            h2 = self.drop(h)
            y_t = self.out_proj(torch.cat([h2, ctx], dim=1))
            outs.append(y_t.unsqueeze(1))

            if (y_teacher is not None) and (tf_ratio > 0.0) and (random.random() < tf_ratio):
                prev_y = y_teacher[:, t, :]
            else:
                prev_y = y_t

        return torch.cat(outs, dim=1)


# -------------------------
# 6) Training (horizon-weighted MSE)
# -------------------------
def train_model(cfg: SimCfg, model: nn.Module, name: str, outdir: str, train_ds, val_ds, num_workers: int = 0):
    print(f"\n🧠 Training {name} | q={cfg.q_in} | L={cfg.w_out} | ep={cfg.epochs} bs={cfg.batch_size}")
    model = model.to(cfg.device)

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pin = cfg.device.startswith("cuda")
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=pin, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin, num_workers=num_workers)

    use_amp = cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    # loss weights
    w_h = cfg.horizon_loss_weights().to(cfg.device).view(1, cfg.w_out, 1)

    best_val, bad = 1e18, 0
    best_path = os.path.join(outdir, f"best_{name}.pt")

    for ep in range(cfg.epochs):
        model.train()
        tr, nb = 0.0, 0
        for x, y in train_dl:
            x = torch.nan_to_num(x.to(cfg.device, non_blocking=True).float(), nan=0.0, posinf=0.0, neginf=0.0)
            y = torch.nan_to_num(y.to(cfg.device, non_blocking=True).float(), nan=0.0, posinf=0.0, neginf=0.0)

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda", enabled=True):
                    if isinstance(model, SCP_Seq2Seq):
                        pred = model(x, cfg.w_out, y_teacher=y, tf_ratio=cfg.scp_tf_ratio)
                    else:
                        pred = model(x, cfg.w_out)
                    loss = torch.mean(((pred - y) ** 2) * w_h)
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                if isinstance(model, SCP_Seq2Seq):
                    pred = model(x, cfg.w_out, y_teacher=y, tf_ratio=cfg.scp_tf_ratio)
                else:
                    pred = model(x, cfg.w_out)
                loss = torch.mean(((pred - y) ** 2) * w_h)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            tr += float(loss.item()); nb += 1

        tr = tr / max(1, nb)

        model.eval()
        va, nv = 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x = torch.nan_to_num(x.to(cfg.device, non_blocking=True).float(), nan=0.0, posinf=0.0, neginf=0.0)
                y = torch.nan_to_num(y.to(cfg.device, non_blocking=True).float(), nan=0.0, posinf=0.0, neginf=0.0)
                if use_amp:
                    with torch.amp.autocast("cuda", enabled=True):
                        pred = model(x, cfg.w_out) if not isinstance(model, SCP_Seq2Seq) else model(x, cfg.w_out, y_teacher=None, tf_ratio=0.0)
                        loss = torch.mean(((pred - y) ** 2) * w_h)
                else:
                    pred = model(x, cfg.w_out) if not isinstance(model, SCP_Seq2Seq) else model(x, cfg.w_out, y_teacher=None, tf_ratio=0.0)
                    loss = torch.mean(((pred - y) ** 2) * w_h)
                if not torch.isfinite(loss):
                    continue
                va += float(loss.item()); nv += 1

        va = va / max(1, nv)

        if va < best_val:
            best_val, bad = va, 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1

        if ep % 25 == 0 or ep == cfg.epochs - 1:
            print(f"[{name}] ep={ep:03d} train={tr:.6f} val={va:.6f} best={best_val:.6f} (bad={bad}/{cfg.early_stop_patience})")

        if bad >= cfg.early_stop_patience:
            print(f"🟡 Early stop triggered for {name} at ep={ep}")
            break

    model.load_state_dict(torch.load(best_path, map_location=cfg.device))
    print(f"✅ best checkpoint: {best_path} | size={file_exists_and_size(best_path)}")
    return model


# -------------------------
# 7) Evaluation (KF predicts A+t steps ahead)
# -------------------------
@torch.no_grad()
def eval_nmse_horizon(cfg: SimCfg, models: Dict[str, Optional[nn.Module]], val_ds: SliceDataset, A: int, trials: int = 2000):
    nmse = {k: np.zeros(cfg.w_out) for k in models.keys()}
    kf = AR1KalmanScalar(D=cfg.feat_dim, snr_db=cfg.pilot_snr_db)

    N = len(val_ds)
    idxs = np.random.choice(N, size=min(trials, N), replace=False)

    for idx in idxs:
        x_in, y_true = val_ds[idx]
        x_in = x_in.float().to(cfg.device)
        y_true = y_true.float().to(cfg.device)

        kf.reset()
        for t in range(cfg.q_in):
            kf.update(x_in[t].detach().cpu())

        kf_preds = torch.stack([kf.predict_ahead(A + (i + 1)) for i in range(cfg.w_out)], dim=0).to(cfg.device).float()
        outdated_ref = x_in[-1].detach()

        preds = {}
        for name, m in models.items():
            if name == "Outdated":
                preds[name] = outdated_ref.unsqueeze(0).repeat(cfg.w_out, 1)
            elif name == "KalmanFilter":
                preds[name] = kf_preds
            else:
                if isinstance(m, SCP_Seq2Seq):
                    preds[name] = m(x_in.unsqueeze(0), cfg.w_out, y_teacher=None, tf_ratio=0.0)[0]
                else:
                    preds[name] = m(x_in.unsqueeze(0), cfg.w_out)[0]

        for t in range(cfg.w_out):
            tgt = y_true[t]
            for name in models.keys():
                nmse[name][t] += nmse_db(preds[name][t], tgt)

    for k in nmse:
        nmse[k] /= len(idxs)
    return nmse


# -------------------------
# 8) Plotting + CSV
# -------------------------
def save_csv_nmse_raw(save_path: str, nmse_curve: Dict[str, np.ndarray]):
    keys = list(nmse_curve.keys())
    with open(save_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pred_slot_t(1..L)"] + keys)
        L = len(next(iter(nmse_curve.values())))
        for i in range(L):
            w.writerow([i+1] + [float(nmse_curve[k][i]) for k in keys])

def plot_nmse_curve_w_axis(cfg: SimCfg, A: int, nmse_curve: Dict[str, np.ndarray], save_path: str, title: str):
    w_axis = A + np.arange(1, cfg.w_out + 1)
    plt.figure(figsize=(10, 6))
    for k, v in nmse_curve.items():
        plt.plot(w_axis, v, marker='o', linewidth=2, label=k)
    plt.grid(alpha=0.3)
    plt.xlabel("w = A + t  (coherence intervals)")
    plt.ylabel("NMSE (dB)")
    delay_ms, _ = cfg.delay_steps_from_alt_km(cfg.fixed_alt_km)
    Ts = cfg.coherence_ms
    plt.title(title + f"\nalt={cfg.fixed_alt_km}km | delay_ms={delay_ms:.2f} | Ts={Ts:.2f}ms | dop={cfg.ut_doppler_hz_min}-{cfg.ut_doppler_hz_max}Hz | A={A}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_bar(values: Dict[str, float], title: str, ylabel: str, save_path: str):
    keys = list(values.keys())
    vals = [values[k] for k in keys]
    plt.figure(figsize=(9, 4.5))
    plt.bar(keys, vals)
    plt.grid(axis='y', alpha=0.3)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# -------------------------
# 9) Latency measurement (optional)
# -------------------------
@torch.no_grad()
def measure_forward_latency_ms(model: nn.Module, cfg: SimCfg, q: int, D: int, iters: int = 100) -> float:
    if not cfg.device.startswith("cuda"):
        return float("nan")
    model = model.eval().to(cfg.device)
    x = torch.randn(1, q, D, device=cfg.device)
    for _ in range(20):
        _ = model(x, cfg.w_out) if not isinstance(model, SCP_Seq2Seq) else model(x, cfg.w_out, None, 0.0)
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(iters):
        _ = model(x, cfg.w_out) if not isinstance(model, SCP_Seq2Seq) else model(x, cfg.w_out, None, 0.0)
    ender.record()
    torch.cuda.synchronize()
    return float(starter.elapsed_time(ender) / iters)


# -------------------------
# 10) Experiment runner
# -------------------------
def train_and_eval_one(base_dir: str, cfg: SimCfg, exp_tag: str, force_data: bool = False, num_workers: int = 0):
    outdir = os.path.join(base_dir, "results", exp_tag)
    ensure_dir(outdir)

    log_path = os.path.join(outdir, "run.log")
    log_f = open(log_path, "w", encoding="utf-8")
    orig_stdout = sys.stdout
    sys.stdout = Tee(orig_stdout, log_f)

    try:
        print("🧾 Experiment folder:", outdir)
        print("🧾 Log file:", log_path)

        delay_ms, A = cfg.delay_steps_from_alt_km(cfg.fixed_alt_km)
        print(f"\n⏱️ Delay: alt={cfg.fixed_alt_km} km | delay_mode={cfg.delay_mode}")
        print(f"   delay_ms = {delay_ms:.2f} ms  | Ts = {cfg.coherence_ms:.2f} ms  -> A = {A} slots")

        # Save cfg
        cfg_dump = asdict(cfg)
        cfg_dump["delay_calc"] = {"alt_km": cfg.fixed_alt_km, "delay_mode": cfg.delay_mode, "delay_ms": float(delay_ms),
                                  "Ts_ms": float(cfg.coherence_ms), "A_slots": int(A)}
        cfg_path = os.path.join(outdir, "cfg.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_dump, f, indent=2, ensure_ascii=False)
        print("\n🧾 cfg saved:", cfg_path, "|", file_exists_and_size(cfg_path))

        # Dataset
        _, pack = get_or_make_dataset(base_dir, cfg, A, force=force_data)
        Xtr, Ytr = pack["train"]["X"], pack["train"]["Y"]
        Xva, Yva = pack["val"]["X"], pack["val"]["Y"]

        train_ds = SliceDataset(Xtr, Ytr)
        val_ds = SliceDataset(Xva, Yva)

        # Models
        D = cfg.feat_dim
        blocks = cfg.pick_tdnn_blocks()
        print(f"\n🧩 TDNN blocks auto = {blocks} (q={cfg.q_in})")

        tdnn_knet = TDNNKalmanNet(D, H=cfg.tdnn_hidden, blocks=blocks, k=cfg.tdnn_kernel, drop=cfg.tdnn_drop)
        knet = KalmanNetGRU(D, H=cfg.knet_hidden)
        scp = SCP_Seq2Seq(D, w_out=cfg.w_out, H=cfg.scp_hidden, layers=cfg.scp_layers, dropout=cfg.dropout)

        # Train
        t0 = time.time()
        tdnn_knet = train_model(cfg, tdnn_knet, "TDNN_KalmanNet", outdir, train_ds, val_ds, num_workers=num_workers)
        knet = train_model(cfg, knet, "KalmanNet_GRU", outdir, train_ds, val_ds, num_workers=num_workers)
        scp = train_model(cfg, scp, "SCP_Seq2Seq", outdir, train_ds, val_ds, num_workers=num_workers)
        print(f"⏱️ Total training (this scenario): {(time.time()-t0)/60.0:.2f} min")

        # Eval
        eval_models = {
            "TDNN-KalmanNet": tdnn_knet.eval(),
            "KalmanNet": knet.eval(),
            "KalmanFilter": None,
            "SCP": scp.eval(),
            "Outdated": None,
        }
        nmse_curve = eval_nmse_horizon(cfg, eval_models, val_ds, A=A, trials=cfg.horizon_trials)

        # Save
        nmse_raw_csv = os.path.join(outdir, "nmse_vs_predslots_raw.csv")
        save_csv_nmse_raw(nmse_raw_csv, nmse_curve)

        nmse_png = os.path.join(outdir, "nmse_vs_w_axis_AplusT.png")
        plot_nmse_curve_w_axis(
            cfg, A, nmse_curve,
            save_path=nmse_png,
            title=f"NMSE vs w=A+t | q={cfg.q_in}, L={cfg.w_out} | SNR={cfg.pilot_snr_db}dB",
        )

        params = {
            "TDNN-KalmanNet": count_params(tdnn_knet),
            "KalmanNet": count_params(knet),
            "SCP": count_params(scp),
            "KalmanFilter": 0,
            "Outdated": 0,
        }
        params_png = os.path.join(outdir, "params_bar.png")
        plot_bar(params, "Parameter Count Comparison", "Params (#)", params_png)

        size_mb = {
            "TDNN-KalmanNet(fp32)": bytes_to_mb(state_dict_size_bytes(tdnn_knet, fp16=False)),
            "TDNN-KalmanNet(fp16)": bytes_to_mb(state_dict_size_bytes(tdnn_knet, fp16=True)),
            "KalmanNet(fp32)": bytes_to_mb(state_dict_size_bytes(knet, fp16=False)),
            "KalmanNet(fp16)": bytes_to_mb(state_dict_size_bytes(knet, fp16=True)),
            "SCP(fp32)": bytes_to_mb(state_dict_size_bytes(scp, fp16=False)),
            "SCP(fp16)": bytes_to_mb(state_dict_size_bytes(scp, fp16=True)),
        }
        size_png = os.path.join(outdir, "model_size_bar.png")
        plot_bar(size_mb, "UE-side Model Storage (state_dict) Approx.", "Size (MB)", size_png)

        latency_png = None
        latency = {}
        if cfg.measure_latency and cfg.device.startswith("cuda"):
            latency["TDNN-KalmanNet"] = measure_forward_latency_ms(tdnn_knet, cfg, cfg.q_in, D, iters=80)
            latency["KalmanNet"] = measure_forward_latency_ms(knet, cfg, cfg.q_in, D, iters=80)
            latency["SCP"] = measure_forward_latency_ms(scp, cfg, cfg.q_in, D, iters=80)
            latency_png = os.path.join(outdir, "latency_bar.png")
            plot_bar(latency, "Inference Latency (Forward) on GPU", "Latency (ms)", latency_png)

        summary_path = os.path.join(outdir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "delay_ms": float(delay_ms),
                    "A_slots": int(A),
                    "Ts_ms": float(cfg.coherence_ms),
                    "doppler_hz_minmax": [float(cfg.ut_doppler_hz_min), float(cfg.ut_doppler_hz_max)],
                    "params": params,
                    "model_size_mb": size_mb,
                    "latency_ms": latency,
                    "mean_nmse_db": {k: float(np.mean(v)) for k, v in nmse_curve.items()},
                },
                f, indent=2, ensure_ascii=False
            )

        print("\n✅ Saved results to:", outdir)
        outs = [nmse_png, nmse_raw_csv, params_png, size_png, summary_path, log_path]
        if latency_png:
            outs.append(latency_png)
        for p in outs:
            print("  -", p, "|", file_exists_and_size(p))

        show_image(nmse_png, "NMSE vs w=A+t")
        show_image(params_png, "Parameter Count")
        show_image(size_png, "Model Storage (MB)")
        if latency_png:
            show_image(latency_png, "Latency (ms)")

        zip_path = os.path.join(base_dir, "results", f"{exp_tag}.zip")
        zip_path = make_zip_of_folder(outdir, zip_out=zip_path)
        print("\n📦 Zipped results:", zip_path, "|", file_exists_and_size(zip_path))
        show_download_link(zip_path)

        return nmse_curve

    finally:
        sys.stdout = orig_stdout
        try:
            log_f.close()
        except Exception:
            pass


# -------------------------
# 11) Main
# -------------------------
def parse_q_list(s: str) -> List[int]:
    xs = []
    for part in s.split(","):
        part = part.strip()
        if part:
            xs.append(int(part))
    return xs

def apply_scenario(cfg: SimCfg, scenario: str) -> SimCfg:
    # baseline: close to your previous setting
    if scenario == "baseline":
        return cfg

    # tdnn_adv: fairly penalize scalar KF and reward TDNN structure
    if scenario == "tdnn_adv":
        return replace(
            cfg,
            rician_K_db=3.0,
            num_paths=12,
            as_std_deg=2.0,
            aoa_rw_std_deg=0.06,
            doppler_path_std_hz=18.0,
            doppler_rw_std_hz=2.0,
            doppler_jump_prob=0.06,
            doppler_jump_std_hz=45.0,
            phase_noise_std_rad=0.02,
            gain_ar_rho=0.995,
            gain_ar_std=0.08,
            pilot_outlier_prob=0.02,
            pilot_outlier_scale=8.0,
            # loss weighting to separate tail more
            loss_tail_weight=2.0,
            loss_first_boost=1.0,
            # slightly more training stability
            lr=min(cfg.lr, 8e-4),
        )

    return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/kaggle/working/LEO_Project", help="output folder")
    parser.add_argument("--force_data", action="store_true", help="regenerate datasets")
    parser.add_argument("--q_list", type=str, default="4,15", help="comma-separated q values, e.g., 4,15")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--alt_km", type=int, default=1500, help="fixed altitude for delay axis")
    parser.add_argument("--w_out", type=int, default=15, help="prediction horizon L")
    parser.add_argument("--dop_min", type=float, default=50.0, help="residual doppler min (Hz)")
    parser.add_argument("--dop_max", type=float, default=100.0, help="residual doppler max (Hz)")
    parser.add_argument("--Ts_ms", type=float, default=1.0, help="coherence interval Ts (ms)")
    parser.add_argument("--scenario", type=str, default="baseline", choices=["baseline", "tdnn_adv"],
                        help="baseline or tdnn_adv (penalize scalar KF fairly)")
    parser.add_argument("--no_latency", action="store_true", help="disable latency measurement")

    if in_notebook():
        args = parser.parse_args(args=[])
        print("ℹ️ Notebook mode: using default arguments.")
    else:
        args = parser.parse_args()

    ensure_dir(args.base_dir)
    ensure_dir(os.path.join(args.base_dir, "results"))
    ensure_dir(os.path.join(args.base_dir, "datasets"))

    seed_all(42)

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    q_values = parse_q_list(args.q_list)

    print(f"📌 base_dir     = {args.base_dir}")
    print(f"📌 device       = {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"📌 q_list       = {q_values}")
    print(f"📌 fixed alt_km = {args.alt_km}")
    print(f"📌 w_out        = {args.w_out}")
    print(f"📌 doppler      = {args.dop_min}~{args.dop_max} Hz (residual)")
    print(f"📌 Ts_ms        = {args.Ts_ms}")
    print(f"📌 scenario     = {args.scenario}")

    for q in q_values:
        cfg = replace(
            SimCfg(),
            q_in=int(q),
            w_out=int(args.w_out),
            delay_mode="oneway",
            fixed_alt_km=int(args.alt_km),
            ut_doppler_hz_min=float(args.dop_min),
            ut_doppler_hz_max=float(args.dop_max),
            coherence_ms=float(args.Ts_ms),
            measure_latency=(not args.no_latency),
        )
        cfg = apply_scenario(cfg, args.scenario)

        delay_ms, A = cfg.delay_steps_from_alt_km(cfg.fixed_alt_km)

        tag = (
            f"{args.scenario}_ALT{cfg.fixed_alt_km}_q{cfg.q_in}_A{A}_L{cfg.w_out}_"
            f"dop{safe_float_tag(cfg.ut_doppler_hz_min)}to{safe_float_tag(cfg.ut_doppler_hz_max)}_"
            f"Ts{safe_float_tag(cfg.coherence_ms)}_snr{int(cfg.pilot_snr_db)}_"
            f"K{safe_float_tag(cfg.rician_K_db)}_P{cfg.num_paths}_"
            f"aoaRW{safe_float_tag(cfg.aoa_rw_std_deg)}_dopRW{safe_float_tag(cfg.doppler_rw_std_hz)}_"
            f"{timestamp_str()}"
        )

        print("\n" + "=" * 80)
        print(f"🚀 Running: {tag}")
        print(f"   TDNN blocks auto = {cfg.pick_tdnn_blocks()}  (q={cfg.q_in})")
        print(f"   oneway delay_ms={delay_ms:.2f}, A={A} slots")
        train_and_eval_one(args.base_dir, cfg, exp_tag=tag, force_data=args.force_data, num_workers=args.num_workers)

    print("\n🎉 All done!")

if __name__ == "__main__":
    main()
