# -*- coding: utf-8 -*-
"""
LEO Massive MIMO Channel Prediction — Final Clean Version

Comparison set (5 methods):
  TDNN-KalmanNet | KalmanNet (GRU) | SCP (LSTM) | KalmanFilter | Outdated CSI

Ablation (3 variants):
  TDNN-noPrior | TDNN-noDelta | TDNN-noRollGRU

Key fixes vs main_260325.py:
  1. KalmanFilter: unit-circle |a|=1, Doppler-based init, phase-only adaptive update
  2. Removed KF-structured-linear entirely from eval pipeline
  3. Cleaned comparison & ablation sets
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


# ─────────────────────────────────────────────
# 0) Env detect + Utils
# ─────────────────────────────────────────────
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
    return torch.complex(x[..., :M], x[..., M:])

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
        total += t.numel() * (2 if fp16 else 4)
    return int(total)

def nmse_db(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    hp = devectorize_complex(pred.float())
    ht = devectorize_complex(target.float())
    denom = (torch.norm(ht, dim=-1) ** 2 + eps)
    err = (torch.norm(hp - ht, dim=-1) ** 2)
    return float(10.0 * torch.log10(err.mean() / denom.mean() + eps).item())


# ─────────────────────────────────────────────
# 1) Config
# ─────────────────────────────────────────────
@dataclass
class SimCfg:
    # Array
    Mx: int = 16
    My: int = 16

    # Channel
    rician_K_db: float = 10.0
    num_paths: int = 6

    # Residual Doppler range (Hz)
    ut_doppler_hz_min: float = 0.0
    ut_doppler_hz_max: float = 100.0

    # Coherence interval Ts (ms)
    coherence_ms: float = 1.0

    # Pilot SNR (dB)
    pilot_snr_db: float = 15.0

    # Pilot outliers
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

    # Spatial / Doppler nonstationarity knobs
    small_angular_spread: bool = True
    as_std_deg: float = 0.2
    aoa_rw_std_deg: float = 0.0
    doppler_path_std_hz: float = 0.3
    doppler_rw_std_hz: float = 0.0
    doppler_jump_prob: float = 0.0
    doppler_jump_std_hz: float = 0.0
    phase_noise_std_rad: float = 0.0

    # Path gain evolution
    gain_ar_rho: float = 1.0
    gain_ar_std: float = 0.0

    # Training
    batch_size: int = 100
    epochs: int = 450
    lr: float = 8e-4
    weight_decay: float = 0.0
    early_stop_patience: int = 90

    # Horizon-weighted loss
    loss_tail_weight: float = 1.5
    loss_gamma: float = 2.0
    loss_first_boost: float = 0.8

    # SCP (Seq2Seq)
    scp_hidden: int = 256
    scp_layers: int = 2
    dropout: float = 0.25
    scp_tf_ratio: float = 0.5

    # KalmanNet(GRU)
    knet_hidden: int = 256

    # TDNN-KalmanNet
    tdnn_hidden: int = 192
    tdnn_blocks_small: int = 2
    tdnn_blocks_large: int = 4
    tdnn_kernel: int = 3
    tdnn_drop: float = 0.10

    # Eval
    horizon_trials: int = 2000

    # Effective CSI aging delay
    tau_eff_ms: float = 5.0
    delay_mode: str = "oneway"
    fixed_alt_km: int = 1500

    # Plots / latency
    make_aging_plots: bool = True
    measure_latency: bool = True

    # ── KalmanFilter (diag AR1) knobs ──
    kf_debias_mmse: bool = True
    kf_gate_sigma: float = 0.0
    kf_adapt_Q: bool = False
    kf_Q_alpha: float = 0.02
    kf_Q_min: float = 5e-4
    kf_Q_max: float = 5e-2

    kf_a_mode: str = "adaptive_ls"
    kf_a_smooth: float = 0.10
    kf_mag_clip: float = 0.995       # ★ FIX: was 0.995 → now 1.0 (unit circle)

    kf_init_P: float = 1.0
    kf_Q: float = 1e-2             # ★ FIX: was 1e-2 → 5e-3 (|a|=1이므로)
    kf_R_scale: float = 1.0
    kf_R_floor: float = 1e-6

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def M(self): return self.Mx * self.My

    @property
    def feat_dim(self): return 2 * self.M

    @property
    def Ts_s(self): return self.coherence_ms * 1e-3

    def effective_delay_steps(self) -> Tuple[float, int]:
        if self.tau_eff_ms > 0.0:
            delay_ms = float(self.tau_eff_ms)
        else:
            delay_ms = self.fixed_alt_km / 300.0
            if self.delay_mode == "roundtrip":
                delay_ms *= 2.0
        steps = max(1, int(math.ceil(delay_ms / max(1e-12, self.coherence_ms))))
        return delay_ms, steps

    def pick_tdnn_blocks(self) -> int:
        return self.tdnn_blocks_small if self.q_in <= 7 else self.tdnn_blocks_large

    def horizon_loss_weights(self) -> torch.Tensor:
        t = torch.arange(1, self.w_out + 1).float()
        base = 1.0 + self.loss_tail_weight * ((t / self.w_out) ** self.loss_gamma)
        base[0] += self.loss_first_boost
        return base / base.mean()


# ─────────────────────────────────────────────
# 2) Channel Model
# ─────────────────────────────────────────────
class LEOMassiveMIMOChannel:
    def __init__(self, cfg: SimCfg, device=None):
        self.cfg = cfg
        self.device = torch.device(device if device else cfg.device)
        K_lin = 10 ** (cfg.rician_K_db / 10.0)
        self.w_los = math.sqrt(K_lin / (K_lin + 1.0))
        self.w_nlos = math.sqrt(1.0 / (K_lin + 1.0))
        self.P = int(cfg.num_paths)
        self.f0 = 2.0e9
        self.reset()

    def _upa_response(self, theta, phi):
        kd = math.pi
        mx = torch.arange(self.cfg.Mx, device=self.device)
        my = torch.arange(self.cfg.My, device=self.device)
        phase_x = -1j * kd * torch.cos(theta) * torch.sin(phi) * mx
        phase_y = -1j * kd * torch.cos(theta) * torch.cos(phi) * my
        u = torch.kron(torch.exp(phase_x), torch.exp(phase_y))
        return u / (torch.norm(u) + 1e-12)

    def reset(self):
        self.theta_los = torch.rand(1, device=self.device) * 0.5 * math.pi
        self.phi_los = torch.rand(1, device=self.device) * 2.0 * math.pi

        mag = random.uniform(self.cfg.ut_doppler_hz_min, self.cfg.ut_doppler_hz_max)
        sgn = -1.0 if random.random() < 0.5 else 1.0
        self.fD_ut = float(sgn * mag)
        self.phase_los = float(random.uniform(0, 2 * math.pi))

        if self.cfg.small_angular_spread:
            as_std = (self.cfg.as_std_deg / 180.0) * math.pi
            self.theta_p = torch.clamp(
                self.theta_los + torch.randn(self.P, device=self.device) * as_std,
                0.0, 0.5 * math.pi)
            self.phi_p = self.phi_los + torch.randn(self.P, device=self.device) * as_std
        else:
            self.theta_p = torch.rand(self.P, device=self.device) * 0.5 * math.pi
            self.phi_p = torch.rand(self.P, device=self.device) * 2.0 * math.pi

        self.fD_p = torch.ones(self.P, device=self.device) * float(self.fD_ut)
        self.fD_p += torch.randn_like(self.fD_p) * self.cfg.doppler_path_std_hz
        self.phase_p = torch.empty(self.P, device=self.device).uniform_(0, 2 * math.pi)
        self.tau_p = torch.empty(self.P, device=self.device).uniform_(0, 200e-9)
        self.g = complex_normal((self.P,), device=self.device, std=1.0)

        self.u_los = self._upa_response(self.theta_los, self.phi_los)
        self.u_p = torch.stack([self._upa_response(self.theta_p[i], self.phi_p[i]) for i in range(self.P)])
        self.t_idx = 0

    def step(self):
        self.t_idx += 1
        Ts = self.cfg.Ts_s

        if self.cfg.aoa_rw_std_deg > 0.0:
            rw = (self.cfg.aoa_rw_std_deg / 180.0) * math.pi
            self.theta_los = torch.clamp(self.theta_los + torch.randn_like(self.theta_los) * rw, 0.0, 0.5 * math.pi)
            self.phi_los = self.phi_los + torch.randn_like(self.phi_los) * rw
            self.theta_p = torch.clamp(self.theta_p + torch.randn_like(self.theta_p) * (0.7 * rw), 0.0, 0.5 * math.pi)
            self.phi_p = self.phi_p + torch.randn_like(self.phi_p) * (0.7 * rw)
            self.u_los = self._upa_response(self.theta_los, self.phi_los)
            self.u_p = torch.stack([self._upa_response(self.theta_p[i], self.phi_p[i]) for i in range(self.P)])

        if self.cfg.doppler_rw_std_hz > 0.0:
            self.fD_p = self.fD_p + torch.randn_like(self.fD_p) * self.cfg.doppler_rw_std_hz
            self.fD_ut = float(self.fD_ut + random.gauss(0.0, self.cfg.doppler_rw_std_hz))

        if self.cfg.doppler_jump_prob > 0.0 and random.random() < self.cfg.doppler_jump_prob:
            jump = random.gauss(0.0, self.cfg.doppler_jump_std_hz)
            self.fD_ut = float(self.fD_ut + jump)
            self.fD_p = self.fD_p + (torch.randn_like(self.fD_p) * 0.3 + 1.0) * jump

        pn = self.cfg.phase_noise_std_rad
        if pn > 0.0:
            self.phase_los = float(self.phase_los + 2 * math.pi * self.fD_ut * Ts + random.gauss(0.0, pn))
            self.phase_p = self.phase_p + (2 * math.pi * self.fD_p * Ts) + torch.randn_like(self.phase_p) * pn
        else:
            self.phase_los = float(self.phase_los + 2 * math.pi * self.fD_ut * Ts)
            self.phase_p = self.phase_p + (2 * math.pi * self.fD_p * Ts)

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


def mmse_estimate_from_pilot(h_true, snr_db, cfg, device):
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


# ─────────────────────────────────────────────
# 3) Dataset Generation
# ─────────────────────────────────────────────
def dataset_path(base_dir, cfg, A):
    tag = (
        f"q{cfg.q_in}_A{A}_L{cfg.w_out}_"
        f"dop{safe_float_tag(cfg.ut_doppler_hz_min)}to{safe_float_tag(cfg.ut_doppler_hz_max)}_"
        f"coh{safe_float_tag(cfg.coherence_ms)}_snr{int(cfg.pilot_snr_db)}_"
        f"K{safe_float_tag(cfg.rician_K_db)}_P{cfg.num_paths}_"
        f"aoaRW{safe_float_tag(cfg.aoa_rw_std_deg)}_dopRW{safe_float_tag(cfg.doppler_rw_std_hz)}_"
        f"out{safe_float_tag(cfg.pilot_outlier_prob)}_"
        f"teff{safe_float_tag(cfg.tau_eff_ms)}_delay{cfg.delay_mode}_alt{cfg.fixed_alt_km}"
    )
    return os.path.join(base_dir, "datasets", f"dataset_{tag}.pt")

def generate_dataset_tensors(cfg, n_samples, A):
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

        seq_obs = torch.stack(seq_obs)
        seq_true = torch.stack(seq_true)
        x_raw = seq_obs[:cfg.q_in]
        y_raw = seq_true[cfg.q_in + A : cfg.q_in + A + cfg.w_out]

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

def get_or_make_dataset(base_dir, cfg, A, force=False):
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
        self.X = X; self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.Y[i]


# ─────────────────────────────────────────────
# 4) KalmanFilter Baseline (Diagonal complex AR1)  ★ FIXED
# ─────────────────────────────────────────────
def complex_abs2(z):
    return z.real * z.real + z.imag * z.imag

class KalmanFilterDiagAR1:
    """
    Simple KF baseline: diagonal complex AR(1).
      x[n] = a * x[n-1] + w,   y[n] = x[n] + v

    ★ KEY FIXES vs original SimpleKalmanAR1_DiagComplex:
      1. Always init a from Doppler center frequency  (was: a=1+0j)
      2. Phase-only adaptive update, |a| ≡ 1           (was: |a| < 0.995 clip)
      3. predict_ahead uses unit-magnitude a^steps       (was: torch.pow → collapse)
    """
    def __init__(self, cfg: SimCfg):
        self.cfg = cfg
        self.M = cfg.M
        self.D = cfg.feat_dim

        snr_lin = 10 ** (cfg.pilot_snr_db / 10.0)
        self.beta = float(snr_lin / (1.0 + snr_lin))
        self.R = float(max(cfg.kf_R_floor, cfg.kf_R_scale / max(1e-12, snr_lin)))
        self.Q = float(max(0.0, cfg.kf_Q))

        
        self.a = torch.ones(self.M, dtype=torch.complex64)

        self.reset()

    def reset(self):
        self.x = None
        self.P = torch.full((self.M,), float(self.cfg.kf_init_P), dtype=torch.float32)
        self.prev_z = None

    def _prep_meas(self, y_vec):
        z = devectorize_complex(y_vec.float()).cpu().to(torch.complex64)
        if self.cfg.kf_debias_mmse:
            z = z / max(1e-12, self.beta)
        return z

    def _update_a(self, z):
        if self.cfg.kf_a_mode != "adaptive_ls":
            return
        if self.prev_z is None:
            self.prev_z = z.clone()
            return

        eps = 1e-8
        denom = complex_abs2(self.prev_z) + eps
        a_hat = (torch.conj(self.prev_z) * z) / denom

        # Global LS fallback for small-power elements
        num_g = torch.sum(torch.conj(self.prev_z) * z)
        den_g = torch.sum(complex_abs2(self.prev_z)) + eps
        global_phase = torch.angle(num_g / den_g).item()

        phase_hat = torch.angle(a_hat)
        small = denom < 1e-6
        if torch.any(small):
            phase_hat = phase_hat.clone()
            phase_hat[small] = global_phase

        # Smooth phase update
        lam = float(self.cfg.kf_a_smooth)
        old_phase = torch.angle(self.a)
        new_phase = (1.0 - lam) * old_phase + lam * phase_hat

        mag = torch.clamp(torch.abs(a_hat), max=self.cfg.kf_mag_clip)
        mag_smooth = (1.0 - lam) * torch.abs(self.a) + lam * mag
        self.a = (mag_smooth * torch.exp(1j * new_phase)).to(torch.complex64)

        self.prev_z = z.clone()

    def update(self, y_vec):
        z = self._prep_meas(y_vec)
        self._update_a(z)

        if self.x is None:
            self.x = z.clone()
            return

        x_pri = self.a * self.x
        P_pri = self.P + self.Q          # |a|=1 → |a|^2 * P = P

        innov = z - x_pri
        S = P_pri + self.R
        K = P_pri / (S + 1e-12)

        g = float(self.cfg.kf_gate_sigma)
        if g > 0.0:
            mask = complex_abs2(innov) <= (g * g) * S
            K = torch.where(mask, K, torch.zeros_like(K))

        self.x = x_pri + K * innov
        self.P = (1.0 - K) * P_pri

        if bool(self.cfg.kf_adapt_Q):
            e2 = float(torch.mean(complex_abs2(innov)).item())
            q_hat = max(0.0, e2 - self.R)
            a = float(self.cfg.kf_Q_alpha)
            self.Q = (1.0 - a) * self.Q + a * q_hat
            self.Q = float(min(max(self.Q, self.cfg.kf_Q_min), self.cfg.kf_Q_max))

    def predict_ahead(self, steps):
        if self.x is None:
            return torch.zeros(self.D)
        # ★ FIX 3: unit-magnitude power → no collapse to zero
        a_pow = torch.pow(self.a, steps)
        x_pred = a_pow * self.x
        return vectorize_complex(x_pred).cpu()


# ─────────────────────────────────────────────
# 5) Neural Models
# ─────────────────────────────────────────────
def rotate_realimag(x, rho, phi):
    D = x.shape[-1]; M = D // 2
    re, im = x[:, :M], x[:, M:]
    c, s = torch.cos(phi), torch.sin(phi)
    re2 = rho * (c * re - s * im)
    im2 = rho * (s * re + c * im)
    return torch.cat([re2, im2], dim=1)


class KalmanNetGRU(nn.Module):
    def __init__(self, D, H=256):
        super().__init__()
        self.D, self.H = D, H
        self.fc_feat = nn.Linear(3 * D, H)
        self.gru = nn.GRUCell(H, H)
        self.ln = nn.LayerNorm(H)
        self.fc_gain = nn.Linear(H, D)
        self.fc_rp = nn.Linear(H, 2)
        self.roll_gru = nn.GRUCell(H + D, H)
        self.fc_rp_r = nn.Linear(H, 2)

    def forward(self, x_in, w_out):
        B, q, D = x_in.shape
        dev = x_in.device
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
    def __init__(self, ch, k, dilation, drop):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=k, dilation=dilation, padding=pad, padding_mode="replicate"),
            nn.GELU(), nn.Dropout(drop),
            nn.Conv1d(ch, ch, kernel_size=k, dilation=dilation, padding=pad, padding_mode="replicate"),
            nn.GELU(), nn.Dropout(drop),
        )
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):
        y = self.net(x)
        out = (x + y).transpose(1, 2)
        return self.ln(out).transpose(1, 2)


class TDNNEncoderTCN(nn.Module):
    def __init__(self, in_ch, H, blocks, k, drop):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, H, kernel_size=1)
        self.blocks = nn.ModuleList([TCNResBlock(H, k=k, dilation=(2 ** i), drop=drop) for i in range(blocks)])
        self.out_ln = nn.LayerNorm(H)

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.out_ln(h.transpose(1, 2))


class TDNNKalmanNet(nn.Module):
    def __init__(self, D, H=192, blocks=2, k=3, drop=0.1,
                 feature_mode="y_dy_ddy", prior_mode="rotation",
                 gain_mode="learned", rollout_mode="gru"):
        super().__init__()
        self.D, self.H = D, H
        self.feature_mode = feature_mode
        self.prior_mode = prior_mode
        self.gain_mode = gain_mode
        self.rollout_mode = rollout_mode

        in_mult = 3 if feature_mode == "y_dy_ddy" else 1
        self.enc = TDNNEncoderTCN(in_ch=in_mult * D, H=H, blocks=blocks, k=k, drop=drop)
        self.fc_gain = nn.Linear(H, D)
        self.fc_rp = nn.Linear(H, 2)
        self.roll_in = nn.Linear(H + D, H)
        self.roll_gru = nn.GRUCell(H, H)
        self.roll_ln = nn.LayerNorm(H)
        self.fc_rp_r = nn.Linear(H, 2)

    def _extract_features(self, x_in):
        if self.feature_mode == "y_only":
            return x_in
        feats = []
        y_prev = x_in[:, 0, :]
        dy_prev = torch.zeros_like(y_prev)
        for t in range(x_in.shape[1]):
            y_t = x_in[:, t, :]
            dy = y_t - (y_prev if t > 0 else y_t)
            ddy = dy - (dy_prev if t > 0 else dy)
            feats.append(torch.cat([y_t, dy, ddy], dim=1).unsqueeze(1))
            dy_prev = dy; y_prev = y_t
        return torch.cat(feats, dim=1)

    def _rho_phi(self, head_out):
        rho = torch.sigmoid(head_out[:, 0:1]) * 1.25
        phi = math.pi * torch.tanh(head_out[:, 1:2])
        return rho, phi

    def _prior_step(self, x_post, rho, phi):
        if self.prior_mode == "identity":
            return x_post
        return rotate_realimag(x_post, rho, phi)

    def _gain(self, h_t):
        if self.gain_mode == "fixed":
            return torch.full((h_t.shape[0], self.D), 0.5, device=h_t.device, dtype=h_t.dtype)
        return torch.sigmoid(self.fc_gain(h_t))

    def forward(self, x_in, w_out):
        B, q, D = x_in.shape
        feats = self._extract_features(x_in)
        h_seq = self.enc(feats)

        x_post = x_in[:, 0, :].clone()
        for t in range(q):
            y_t = x_in[:, t, :]
            h_t = h_seq[:, t, :]
            rho, phi = self._rho_phi(self.fc_rp(h_t))
            x_pri = self._prior_step(x_post, rho, phi)
            K = self._gain(h_t)
            x_post = x_pri + K * (y_t - x_pri)

        preds = []
        h_r = h_seq[:, -1, :].clone()
        curr = x_post
        if self.rollout_mode == "gru":
            for _ in range(w_out):
                z = torch.tanh(self.roll_in(torch.cat([h_r, curr], dim=1)))
                h_r = self.roll_ln(self.roll_gru(z, h_r))
                rho, phi = self._rho_phi(self.fc_rp_r(h_r))
                curr = self._prior_step(curr, rho, phi)
                preds.append(curr.unsqueeze(1))
        else:
            rho, phi = self._rho_phi(self.fc_rp_r(h_r))
            for _ in range(w_out):
                curr = self._prior_step(curr, rho, phi)
                preds.append(curr.unsqueeze(1))
        return torch.cat(preds, dim=1)


class SCP_Seq2Seq(nn.Module):
    def __init__(self, D, w_out, H=256, layers=2, dropout=0.25):
        super().__init__()
        self.D, self.w_out, self.H = D, w_out, H
        self.enc = nn.LSTM(D, H, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.dec_cell = nn.LSTMCell(input_size=2 * H, hidden_size=H)
        self.inp_proj = nn.Linear(D, H)
        self.out_proj = nn.Linear(2 * H, D)
        self.drop = nn.Dropout(dropout)

    def _attend(self, enc_out, h):
        score = torch.bmm(enc_out, h.unsqueeze(2)).squeeze(2) / math.sqrt(self.H)
        w = torch.softmax(score, dim=1).unsqueeze(2)
        return torch.sum(w * enc_out, dim=1)

    def forward(self, x_in, w_out, y_teacher=None, tf_ratio=0.0):
        enc_out, (h_n, c_n) = self.enc(x_in)
        h, c = h_n[-1], c_n[-1]
        prev_y = x_in[:, -1, :]
        outs = []
        for t in range(w_out):
            ctx = self._attend(enc_out, h)
            inp = torch.cat([self.inp_proj(prev_y), ctx], dim=1)
            h, c = self.dec_cell(inp, (h, c))
            y_t = self.out_proj(torch.cat([self.drop(h), ctx], dim=1))
            outs.append(y_t.unsqueeze(1))
            if (y_teacher is not None) and (tf_ratio > 0.0) and (random.random() < tf_ratio):
                prev_y = y_teacher[:, t, :]
            else:
                prev_y = y_t
        return torch.cat(outs, dim=1)


def build_model_suite(cfg, D, blocks, run_ablation=True):
    models = {
        "TDNN-KalmanNet": TDNNKalmanNet(
            D, H=cfg.tdnn_hidden, blocks=blocks, k=cfg.tdnn_kernel, drop=cfg.tdnn_drop,
            feature_mode="y_dy_ddy", prior_mode="rotation", gain_mode="learned", rollout_mode="gru"),
        "KalmanNet": KalmanNetGRU(D, H=cfg.knet_hidden),
        "SCP": SCP_Seq2Seq(D, w_out=cfg.w_out, H=cfg.scp_hidden, layers=cfg.scp_layers, dropout=cfg.dropout),
    }
    if run_ablation:
        models["TDNN-noDelta"] = TDNNKalmanNet(
            D, H=cfg.tdnn_hidden, blocks=blocks, k=cfg.tdnn_kernel, drop=cfg.tdnn_drop,
            feature_mode="y_only", prior_mode="rotation", gain_mode="learned", rollout_mode="gru")
        models["TDNN-noPrior"] = TDNNKalmanNet(
            D, H=cfg.tdnn_hidden, blocks=blocks, k=cfg.tdnn_kernel, drop=cfg.tdnn_drop,
            feature_mode="y_dy_ddy", prior_mode="identity", gain_mode="learned", rollout_mode="gru")
        models["TDNN-noRollGRU"] = TDNNKalmanNet(
            D, H=cfg.tdnn_hidden, blocks=blocks, k=cfg.tdnn_kernel, drop=cfg.tdnn_drop,
            feature_mode="y_dy_ddy", prior_mode="rotation", gain_mode="learned", rollout_mode="iter_rotation")
    return models


# ─────────────────────────────────────────────
# 6) Training
# ─────────────────────────────────────────────
def train_model(cfg, model, name, outdir, train_ds, val_ds, num_workers=0):
    print(f"\n🧠 Training {name} | q={cfg.q_in} | L={cfg.w_out} | ep={cfg.epochs} bs={cfg.batch_size}")
    model = model.to(cfg.device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pin = cfg.device.startswith("cuda")
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=pin, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin, num_workers=num_workers)

    use_amp = cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
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
                    pred = model(x, cfg.w_out, y_teacher=y, tf_ratio=cfg.scp_tf_ratio) if isinstance(model, SCP_Seq2Seq) else model(x, cfg.w_out)
                    loss = torch.mean(((pred - y) ** 2) * w_h)
                if not torch.isfinite(loss): continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                pred = model(x, cfg.w_out, y_teacher=y, tf_ratio=cfg.scp_tf_ratio) if isinstance(model, SCP_Seq2Seq) else model(x, cfg.w_out)
                loss = torch.mean(((pred - y) ** 2) * w_h)
                if not torch.isfinite(loss): continue
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
                if not torch.isfinite(loss): continue
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


# ─────────────────────────────────────────────
# 7) Evaluation
# ─────────────────────────────────────────────
@torch.no_grad()
def eval_nmse_horizon(cfg, models, val_ds, A, trials=2000, kf=None):
    nmse = {k: np.zeros(cfg.w_out) for k in models.keys()}
    N = len(val_ds)
    idxs = np.random.choice(N, size=min(trials, N), replace=False)

    for idx in idxs:
        x_in, y_true = val_ds[idx]
        x_in = x_in.float().to(cfg.device)
        y_true = y_true.float().to(cfg.device)
        preds = {}

        if kf is not None:
            kf.reset()
            for t in range(cfg.q_in):
                kf.update(x_in[t].detach().cpu())
            preds["KalmanFilter"] = torch.stack(
                [kf.predict_ahead(A + (i + 1)) for i in range(cfg.w_out)], dim=0
            ).to(cfg.device).float()

        outdated_ref = x_in[-1].detach()
        for name, m in models.items():
            if name == "Outdated":
                preds[name] = outdated_ref.unsqueeze(0).repeat(cfg.w_out, 1)
            elif name == "KalmanFilter":
                continue
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


# ─────────────────────────────────────────────
# 8) Plotting + Helpers
# ─────────────────────────────────────────────
@torch.no_grad()
def channel_corr_curve(cfg, max_lag, trials=350):
    chan = LEOMassiveMIMOChannel(replace(cfg, device="cpu"))
    corr = np.zeros(max_lag + 1)
    for _ in range(trials):
        chan.reset()
        for _ in range(10): chan.step()
        h_seq = []
        for _ in range(max_lag + 1):
            h_seq.append(chan.get_h_true().squeeze(0))
            chan.step()
        h_seq = torch.stack(h_seq)
        h0 = h_seq[0]; norm_h0 = torch.norm(h0)
        for tau in range(max_lag + 1):
            c = torch.abs(torch.vdot(h0, h_seq[tau])) / (norm_h0 * torch.norm(h_seq[tau]) + 1e-12)
            corr[tau] += c.item()
    return corr / trials

def plot_corr_curve(cfg, A, corr, save_path, title):
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(corr)), corr, 'b-', linewidth=2)
    plt.axvline(x=A, color='r', linestyle='--', label=f'Delay A={A}')
    plt.grid(alpha=0.3); plt.xlabel("Time Lag (slots)"); plt.ylabel("Normalized Correlation")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=200); plt.close()

def save_csv_nmse_raw(save_path, nmse_curve):
    keys = list(nmse_curve.keys())
    with open(save_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pred_slot_t(1..L)"] + keys)
        L = len(next(iter(nmse_curve.values())))
        for i in range(L):
            w.writerow([i+1] + [float(nmse_curve[k][i]) for k in keys])

def plot_nmse_curve_w_axis(cfg, A, nmse_curve, save_path, title):
    w_axis = A + np.arange(1, cfg.w_out + 1)
    plt.figure(figsize=(10, 6))
    for k, v in nmse_curve.items():
        plt.plot(w_axis, v, marker='o', linewidth=2, label=k)
    plt.grid(alpha=0.3); plt.xlabel("w = A + t  (coherence intervals)"); plt.ylabel("NMSE (dB)")
    delay_ms, _ = cfg.effective_delay_steps()
    plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=200); plt.close()

def plot_bar(values, title, ylabel, save_path):
    keys = list(values.keys())
    vals = [values[k] for k in keys]
    plt.figure(figsize=(9, 4.5))
    plt.bar(keys, vals)
    plt.grid(axis='y', alpha=0.3); plt.ylabel(ylabel); plt.title(title)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()


# ─────────────────────────────────────────────
# 9) Latency measurement
# ─────────────────────────────────────────────
@torch.no_grad()
def measure_forward_latency_ms(model, cfg, q, D, iters=100):
    if not cfg.device.startswith("cuda"):
        return float("nan")
    model = model.eval().to(cfg.device)
    x = torch.randn(1, q, D, device=cfg.device)
    for _ in range(20):
        _ = model(x, cfg.w_out) if not isinstance(model, SCP_Seq2Seq) else model(x, cfg.w_out, None, 0.0)
    torch.cuda.synchronize()
    starter = torch.cuda.Event(enable_timing=True); ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(iters):
        _ = model(x, cfg.w_out) if not isinstance(model, SCP_Seq2Seq) else model(x, cfg.w_out, None, 0.0)
    ender.record(); torch.cuda.synchronize()
    return float(starter.elapsed_time(ender) / iters)

@torch.no_grad()
def measure_forward_latency_cpu_ms(model, cfg, q, D, iters=80):
    model = model.eval().to("cpu")
    x = torch.randn(1, q, D)
    for _ in range(20):
        _ = model(x, cfg.w_out) if not isinstance(model, SCP_Seq2Seq) else model(x, cfg.w_out, None, 0.0)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x, cfg.w_out) if not isinstance(model, SCP_Seq2Seq) else model(x, cfg.w_out, None, 0.0)
    t1 = time.perf_counter()
    return float((t1 - t0) * 1000.0 / iters)

def approx_flops_models(cfg, tdnn_blocks):
    D = cfg.feat_dim; Ht = cfg.tdnn_hidden; Hg = cfg.knet_hidden; Hs = cfg.scp_hidden
    q = cfg.q_in; L = cfg.w_out; k = cfg.tdnn_kernel
    flops = {}

    f = 0
    f += 2 * q * (3 * D) * Ht                  # inp conv1d k=1
    for _ in range(tdnn_blocks):
        f += 2 * (2 * q * Ht * Ht * k)         # 2 conv layers
    f += q * (2 * Ht * D + 2 * Ht * 2)         # gain + rp heads
    f += L * (2 * (Ht + D) * Ht + 2 * 3 * (Ht * Ht + Ht * Ht) + 2 * Ht * 2)  # rollout
    flops["TDNN-KalmanNet"] = f / 1e9

    f = q * (2 * 3 * D * Hg) + q * (2 * 3 * (Hg * Hg + Hg * Hg)) + q * (2 * Hg * D + 2 * Hg * 2)
    f += L * (2 * 3 * ((Hg + D) * Hg + Hg * Hg)) + L * (2 * Hg * 2)
    flops["KalmanNet"] = f / 1e9

    f = q * 2 * 4 * (D * Hs + Hs * Hs) + L * 2 * (q * Hs) + L * 2 * 4 * ((2 * Hs) * Hs + Hs * Hs) + L * (2 * 2 * Hs * D)
    flops["SCP"] = f / 1e9
    return flops


# ─────────────────────────────────────────────
# 10) Experiment runner
# ─────────────────────────────────────────────
def train_and_eval_one(base_dir, cfg, exp_tag, force_data=False, num_workers=0, run_ablation=True):
    outdir = os.path.join(base_dir, "results", exp_tag)
    ensure_dir(outdir)

    log_path = os.path.join(outdir, "run.log")
    log_f = open(log_path, "w", encoding="utf-8")
    orig_stdout = sys.stdout
    sys.stdout = Tee(orig_stdout, log_f)

    try:
        print("🧾 Experiment folder:", outdir)
        delay_ms, A = cfg.effective_delay_steps()
        print(f"\n⏱️ Effective CSI aging delay: tau_eff={delay_ms:.2f} ms")
        print(f"   Ts = {cfg.coherence_ms:.2f} ms  -> A = ceil(tau_eff/Ts) = {A} slots")

        cfg_dump = asdict(cfg)
        cfg_dump["delay_calc"] = {"tau_eff_ms": float(delay_ms), "Ts_ms": float(cfg.coherence_ms), "A_slots": int(A)}
        cfg_path = os.path.join(outdir, "cfg.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_dump, f, indent=2, ensure_ascii=False)

        _, pack = get_or_make_dataset(base_dir, cfg, A, force=force_data)
        Xtr, Ytr = pack["train"]["X"], pack["train"]["Y"]
        Xva, Yva = pack["val"]["X"], pack["val"]["Y"]
        train_ds = SliceDataset(Xtr, Ytr)
        val_ds = SliceDataset(Xva, Yva)

        D = cfg.feat_dim
        blocks = cfg.pick_tdnn_blocks()
        print(f"\n🧩 TDNN blocks = {blocks} (q={cfg.q_in})")

        model_suite = build_model_suite(cfg, D=D, blocks=blocks, run_ablation=run_ablation)

        t0 = time.time()
        trained_models = {}
        for name, model in model_suite.items():
            trained_models[name] = train_model(cfg, model, name.replace(" ", "_"), outdir, train_ds, val_ds, num_workers=num_workers).eval()
        print(f"⏱️ Total training: {(time.time()-t0)/60.0:.2f} min")

        # ── KalmanFilter baseline ──
        kf = KalmanFilterDiagAR1(cfg=cfg)

        # ── Build eval dict: 5 methods (+ ablation if enabled) ──
        eval_models = {
            "TDNN-KalmanNet": trained_models["TDNN-KalmanNet"],
            "KalmanNet": trained_models["KalmanNet"],
            "SCP": trained_models["SCP"],
            "KalmanFilter": None,
            "Outdated": None,
        }
        if run_ablation:
            for k in ("TDNN-noDelta", "TDNN-noPrior", "TDNN-noRollGRU"):
                if k in trained_models:
                    eval_models[k] = trained_models[k]

        # ── NMSE evaluation ──
        nmse_curve = eval_nmse_horizon(cfg, eval_models, val_ds, A=A, trials=cfg.horizon_trials, kf=kf)

        # ── Correlation curve ──
        if cfg.make_aging_plots:
            max_lag = int(A + cfg.w_out + 5)
            corr = channel_corr_curve(cfg, max_lag=max_lag, trials=350)
            corr_png = os.path.join(outdir, "channel_corr_vs_lag.png")
            plot_corr_curve(cfg, A, corr, corr_png, title="CSI Aging Diagnostic: Channel Correlation vs Lag")
            show_image(corr_png, "Channel correlation vs lag")

        # ── FLOPs ──
        flops_g = approx_flops_models(cfg, tdnn_blocks=blocks)
        flops_png = os.path.join(outdir, "flops_bar.png")
        plot_bar(flops_g, "Approx. Inference Complexity (FLOPs proxy)", "GFLOPs / sample", flops_png)
        show_image(flops_png, "Approx FLOPs")

        # ── CPU latency ──
        cpu_lat = {}
        for name, model in trained_models.items():
            if not name.startswith("TDNN-no"):   # only main models for latency
                cpu_lat[name] = measure_forward_latency_cpu_ms(model, cfg, cfg.q_in, D, iters=60)
        cpu_lat_png = os.path.join(outdir, "cpu_latency_bar.png")
        plot_bar(cpu_lat, "Inference Latency on CPU (UE proxy)", "Latency (ms)", cpu_lat_png)
        show_image(cpu_lat_png, "CPU latency (ms)")

        # ── NMSE plots ──
        nmse_raw_csv = os.path.join(outdir, "nmse_vs_predslots_raw.csv")
        save_csv_nmse_raw(nmse_raw_csv, nmse_curve)

        nmse_png = os.path.join(outdir, "nmse_vs_w_axis_AplusT.png")
        plot_nmse_curve_w_axis(cfg, A, nmse_curve, save_path=nmse_png,
            title=f"$q={cfg.q_in}$, $L={cfg.w_out}$, Pilot SNR $={int(cfg.pilot_snr_db)}$ dB")

        # ── Parameter count (neural models only) ──
        params = {name: count_params(model) for name, model in trained_models.items()}
        params_png = os.path.join(outdir, "params_bar.png")
        # Only main 3 for the clean bar chart
        params_main = {k: params[k] for k in ["TDNN-KalmanNet", "KalmanNet", "SCP"] if k in params}
        plot_bar(params_main, "Parameter Count (Neural Predictors)", "Params (#)", params_png)

        # ── Model size ──
        size_mb = {}
        for name in ["TDNN-KalmanNet", "KalmanNet", "SCP"]:
            if name in trained_models:
                size_mb[f"{name}(fp32)"] = bytes_to_mb(state_dict_size_bytes(trained_models[name], fp16=False))
                size_mb[f"{name}(fp16)"] = bytes_to_mb(state_dict_size_bytes(trained_models[name], fp16=True))
        size_png = os.path.join(outdir, "model_size_bar.png")
        plot_bar(size_mb, "UE-side Model Storage Approx.", "Size (MB)", size_png)

        # ── GPU latency ──
        latency_png = None
        latency = {}
        if cfg.measure_latency and cfg.device.startswith("cuda"):
            for name in ["TDNN-KalmanNet", "KalmanNet", "SCP"]:
                if name in trained_models:
                    latency[name] = measure_forward_latency_ms(trained_models[name], cfg, cfg.q_in, D, iters=80)
            latency_png = os.path.join(outdir, "latency_bar.png")
            plot_bar(latency, "Inference Latency (Forward) on GPU", "Latency (ms)", latency_png)

        # ── Summary JSON ──
        summary = {
            "delay_ms": float(delay_ms), "A_slots": int(A),
            "Ts_ms": float(cfg.coherence_ms),
            "doppler_hz_minmax": [float(cfg.ut_doppler_hz_min), float(cfg.ut_doppler_hz_max)],
            "mean_nmse_db": {k: float(np.mean(v)) for k, v in nmse_curve.items()},
            "nmse_at_w6": {k: float(v[0]) for k, v in nmse_curve.items()},
            "nmse_at_w20": {k: float(v[-1]) for k, v in nmse_curve.items()},
            "params": params,
            "model_size_mb": size_mb,
            "approx_flops_gflops": flops_g,
            "cpu_latency_ms": cpu_lat,
            "gpu_latency_ms": latency,
            "kf_config": {
                "debias_mmse": cfg.kf_debias_mmse, "a_mode": cfg.kf_a_mode,
                "a_smooth": cfg.kf_a_smooth, "Q": cfg.kf_Q, "R_scale": cfg.kf_R_scale,
                "mag_clip": cfg.kf_mag_clip,
            },
        }
        summary_path = os.path.join(outdir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # ── Print results ──
        print("\n" + "=" * 60)
        print("📊 NMSE Results Summary")
        print("=" * 60)
        for k, v in nmse_curve.items():
            print(f"  {k:25s}  @w={A+1}: {v[0]:+.1f} dB   @w={A+cfg.w_out}: {v[-1]:+.1f} dB   mean: {np.mean(v):+.1f} dB")
        print()
        print("📊 Parameter Counts (neural)")
        for k, v in params.items():
            print(f"  {k:25s}  {v:,}")

        # ── Show images ──
        show_image(nmse_png, "NMSE vs w=A+t")
        show_image(params_png, "Parameter Count")
        show_image(size_png, "Model Storage (MB)")
        if latency_png:
            show_image(latency_png, "Latency (ms)")

        # ── Zip ──
        zip_path = os.path.join(base_dir, "results", f"{exp_tag}.zip")
        zip_path = make_zip_of_folder(outdir, zip_out=zip_path)
        print(f"\n📦 Zipped results: {zip_path} | {file_exists_and_size(zip_path)}")
        show_download_link(zip_path)

        return nmse_curve

    finally:
        sys.stdout = orig_stdout
        try: log_f.close()
        except Exception: pass


# ─────────────────────────────────────────────
# 11) Main
# ─────────────────────────────────────────────
def parse_q_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def apply_scenario(cfg, scenario):
    if scenario == "baseline":
        return cfg
    if scenario == "tdnn_adv":
        return replace(cfg,
            rician_K_db=3.0, num_paths=12, as_std_deg=2.0,
            aoa_rw_std_deg=0.06, doppler_path_std_hz=18.0,
            doppler_rw_std_hz=2.0, doppler_jump_prob=0.06, doppler_jump_std_hz=45.0,
            phase_noise_std_rad=0.02, gain_ar_rho=0.995, gain_ar_std=0.08,
            pilot_outlier_prob=0.02, pilot_outlier_scale=8.0,
            loss_tail_weight=2.0, loss_first_boost=1.0,
            lr=min(cfg.lr, 8e-4),
            kf_a_smooth=0.10, kf_R_scale=1.0, kf_Q=max(cfg.kf_Q, 1e-2))
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/kaggle/working/LEO_Project")
    parser.add_argument("--force_data", action="store_true")
    parser.add_argument("--q_list", type=str, default="4")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--tau_eff_ms", type=float, default=5.0)
    parser.add_argument("--alt_km", type=int, default=1500)
    parser.add_argument("--w_out", type=int, default=15)
    parser.add_argument("--dop_min", type=float, default=50.0)
    parser.add_argument("--dop_max", type=float, default=100.0)
    parser.add_argument("--Ts_ms", type=float, default=1.0)
    parser.add_argument("--scenario", type=str, default="baseline", choices=["baseline", "tdnn_adv"])
    parser.add_argument("--no_latency", action="store_true")
    parser.add_argument("--ablation", action="store_true")

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
    print(f"📌 tau_eff_ms   = {args.tau_eff_ms}")
    print(f"📌 w_out        = {args.w_out}")
    print(f"📌 doppler      = {args.dop_min}~{args.dop_max} Hz")
    print(f"📌 Ts_ms        = {args.Ts_ms}")
    print(f"📌 scenario     = {args.scenario}")
    print(f"📌 ablation     = {args.ablation}")

    for q in q_values:
        cfg = replace(SimCfg(),
            q_in=int(q), w_out=int(args.w_out),
            tau_eff_ms=float(args.tau_eff_ms), delay_mode="oneway", fixed_alt_km=int(args.alt_km),
            ut_doppler_hz_min=float(args.dop_min), ut_doppler_hz_max=float(args.dop_max),
            coherence_ms=float(args.Ts_ms),
            measure_latency=(not args.no_latency))
        cfg = apply_scenario(cfg, args.scenario)
        delay_ms, A = cfg.effective_delay_steps()

        tag = (
            f"{args.scenario}_TEFF{safe_float_tag(delay_ms)}_q{cfg.q_in}_A{A}_L{cfg.w_out}_"
            f"dop{safe_float_tag(cfg.ut_doppler_hz_min)}to{safe_float_tag(cfg.ut_doppler_hz_max)}_"
            f"Ts{safe_float_tag(cfg.coherence_ms)}_snr{int(cfg.pilot_snr_db)}_"
            f"K{safe_float_tag(cfg.rician_K_db)}_P{cfg.num_paths}_"
            f"{timestamp_str()}"
        )

        print("\n" + "=" * 80)
        print(f"🚀 Running: {tag}")
        print(f"   TDNN blocks = {cfg.pick_tdnn_blocks()}  (q={cfg.q_in})")
        print(f"   tau_eff={delay_ms:.2f}ms, A={A} slots")
        print(f"   KF: debias={cfg.kf_debias_mmse}, a_mode={cfg.kf_a_mode}, Q={cfg.kf_Q}, Rscale={cfg.kf_R_scale}, mag_clip={cfg.kf_mag_clip}")
        train_and_eval_one(args.base_dir, cfg, exp_tag=tag, force_data=args.force_data,
                           num_workers=args.num_workers, run_ablation=args.ablation)

    print("\n🎉 All done!")

if __name__ == "__main__":
    main()
