# -*- coding: utf-8 -*-
"""
LEO Massive MIMO Channel Prediction — ConvNeXtV2-KalmanNet
Platform: Lightning AI (also works on Kaggle/Colab/local)

Comparison (6 methods):
  ConvNeXtV2-KalmanNet | TDNN-KalmanNet | KalmanNet (GRU) |
  SCP (LSTM) | KalmanFilter | Outdated CSI

Ablation (3 variants):
  TDNN-noPrior | TDNN-noDelta | TDNN-noRollGRU

Key design choices:
  - ConvNeXtV2 encoder: DW Conv + LN + GRN (CVPR 2023, no plateau)
  - TDNN encoder: dilated TCN with residual blocks (baseline)
  - KalmanFilter: unit-circle AR(1), Doppler-init, phase-only adaptive
  - LR: linear warmup (10ep) + cosine decay
"""

import os, math, time, json, csv, argparse, random, shutil, sys
from pathlib import Path
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# ============================================================
# Defaults — edit here for quick config changes
# ============================================================
DEFAULTS = {
    "base_dir": "./LEO_Project",       # Lightning AI: relative path works
    "force_data": False,
    "q_list": "15",                     # ★ input length
    "num_workers": 0,                  # safer default for VSCode/local runs
    "tau_eff_ms": 5.0,
    "alt_km": 1500,
    "w_out": 15,
    "dop_min": 50.0,
    "dop_max": 100.0,
    "Ts_ms": 1.0,
    "scenario": "baseline",
    "no_latency": False,
    "ablation": True,                  # ★ ablation enabled
}


# ─────────────────────────────────────────────
# 0) Utils
# ─────────────────────────────────────────────
def in_notebook() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False

if not in_notebook():
    plt.switch_backend("Agg")

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def timestamp_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_float_tag(x):
    return f"{x:.1f}".replace(".", "p")

def file_exists_and_size(path):
    if not os.path.exists(path):
        return "MISSING"
    return f"{os.path.getsize(path)/1024:.1f} KB"

def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v}")

def resolve_base_dir(base_dir: str) -> str:
    p = Path(base_dir)
    if p.is_absolute():
        return str(p)
    try:
        anchor = Path(__file__).resolve().parent
    except NameError:
        anchor = Path.cwd()
    return str((anchor / p).resolve())

def show_image(path, title=""):
    if not in_notebook():
        return
    try:
        from IPython.display import display, Image as IPyImage
        if title:
            print(f"\n🖼️ {title} -> {path}")
        display(IPyImage(filename=path))
    except Exception:
        pass

def show_download_link(path):
    if not in_notebook():
        print("Zip saved at:", path)
        return
    try:
        from IPython.display import display, FileLink
        print("\n⬇️ Download link:")
        display(FileLink(path))
    except Exception:
        print("Zip saved at:", path)

def make_zip_of_folder(folder, zip_out):
    return shutil.make_archive(zip_out.replace(".zip", ""), "zip", folder)

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
    sc = std / 1.41421356
    return torch.complex(torch.randn(shape, device=device) * sc,
                         torch.randn(shape, device=device) * sc)

def vectorize_complex(h):
    return torch.cat([h.real, h.imag], dim=-1)

def devectorize_complex(x):
    M = x.shape[-1] // 2
    return torch.complex(x[..., :M], x[..., M:])

def rms_normalize(x, eps=1e-8):
    scale = torch.sqrt(torch.mean(x.float() ** 2) + eps)
    return x / (scale + eps), scale

def bytes_to_mb(x):
    return float(x) / (1024.0 * 1024.0)

def count_params(m):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))

def state_dict_size_bytes(model, fp16=False):
    return int(sum(t.numel() * (2 if fp16 else 4)
                   for t in model.state_dict().values() if torch.is_tensor(t)))

def nmse_db(pred, target, eps=1e-12):
    hp = devectorize_complex(pred.float())
    ht = devectorize_complex(target.float())
    err = (torch.norm(hp - ht, dim=-1) ** 2).mean()
    den = (torch.norm(ht, dim=-1) ** 2).mean() + eps
    return float(10.0 * torch.log10(err / den + eps).item())


# ─────────────────────────────────────────────
# 1) Config
# ─────────────────────────────────────────────
@dataclass
class SimCfg:
    Mx: int = 16
    My: int = 16
    rician_K_db: float = 10.0
    num_paths: int = 6
    ut_doppler_hz_min: float = 0.0
    ut_doppler_hz_max: float = 100.0
    coherence_ms: float = 1.0
    pilot_snr_db: float = 15.0
    pilot_outlier_prob: float = 0.0
    pilot_outlier_scale: float = 8.0

    q_in: int = 15
    w_out: int = 15
    train_samples: int = 90000
    val_samples: int = 10000
    use_rms_norm: bool = True
    dataset_store_fp16: bool = True

    small_angular_spread: bool = True
    as_std_deg: float = 0.2
    aoa_rw_std_deg: float = 0.0
    doppler_path_std_hz: float = 0.3
    doppler_rw_std_hz: float = 0.0
    doppler_jump_prob: float = 0.0
    doppler_jump_std_hz: float = 0.0
    phase_noise_std_rad: float = 0.0
    gain_ar_rho: float = 1.0
    gain_ar_std: float = 0.0

    batch_size: int = 100
    epochs: int = 450
    lr: float = 8e-4
    weight_decay: float = 0.0
    early_stop_patience: int = 90
    warmup_epochs: int = 10

    loss_tail_weight: float = 1.5
    loss_gamma: float = 2.0
    loss_first_boost: float = 0.8

    scp_hidden: int = 256
    scp_layers: int = 2
    dropout: float = 0.25
    scp_tf_ratio: float = 0.5

    knet_hidden: int = 256

    tdnn_hidden: int = 192
    tdnn_blocks_small: int = 2
    tdnn_blocks_large: int = 4
    tdnn_kernel: int = 3
    tdnn_drop: float = 0.10

    cnv2_hidden: int = 192
    cnv2_blocks_small: int = 2
    cnv2_blocks_large: int = 4
    cnv2_kernel: int = 9
    cnv2_expand: int = 2
    cnv2_drop: float = 0.10

    horizon_trials: int = 2000
    tau_eff_ms: float = 5.0
    delay_mode: str = "oneway"
    fixed_alt_km: int = 1500
    make_aging_plots: bool = True
    measure_latency: bool = True

    # KalmanFilter (diag AR1) — unit circle
    kf_debias_mmse: bool = True
    kf_a_mode: str = "adaptive_ls"
    kf_a_smooth: float = 0.10
    kf_mag_clip: float = 1.0          # unit circle
    kf_init_P: float = 1.0
    kf_Q: float = 5e-3
    kf_R_scale: float = 1.0
    kf_R_floor: float = 1e-6

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def M(self):
        return self.Mx * self.My

    @property
    def feat_dim(self):
        return 2 * self.M

    @property
    def Ts_s(self):
        return self.coherence_ms * 1e-3

    def effective_delay_steps(self) -> Tuple[float, int]:
        if self.tau_eff_ms > 0:
            delay_ms = float(self.tau_eff_ms)
        else:
            delay_ms = self.fixed_alt_km / 300.0
            if self.delay_mode == "roundtrip":
                delay_ms *= 2.0
        return delay_ms, max(1, int(math.ceil(delay_ms / max(1e-12, self.coherence_ms))))

    def pick_tdnn_blocks(self):
        return self.tdnn_blocks_small if self.q_in <= 7 else self.tdnn_blocks_large

    def pick_cnv2_blocks(self):
        return self.cnv2_blocks_small if self.q_in <= 7 else self.cnv2_blocks_large

    def horizon_loss_weights(self):
        t = torch.arange(1, self.w_out + 1).float()
        base = 1.0 + self.loss_tail_weight * ((t / self.w_out) ** self.loss_gamma)
        base[0] += self.loss_first_boost
        return base / base.mean()


# ─────────────────────────────────────────────
# 2) Channel Model
# ─────────────────────────────────────────────
class LEOMassiveMIMOChannel:
    def __init__(self, cfg, device=None):
        self.cfg = cfg
        self.device = torch.device(device or cfg.device)
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
        u = torch.kron(
            torch.exp(-1j * kd * torch.cos(theta) * torch.sin(phi) * mx),
            torch.exp(-1j * kd * torch.cos(theta) * torch.cos(phi) * my),
        )
        return u / (torch.norm(u) + 1e-12)

    def reset(self):
        self.theta_los = torch.rand(1, device=self.device) * 0.5 * math.pi
        self.phi_los = torch.rand(1, device=self.device) * 2.0 * math.pi
        mag = random.uniform(self.cfg.ut_doppler_hz_min, self.cfg.ut_doppler_hz_max)
        self.fD_ut = float((-1.0 if random.random() < 0.5 else 1.0) * mag)
        self.phase_los = float(random.uniform(0, 2 * math.pi))

        if self.cfg.small_angular_spread:
            as_std = (self.cfg.as_std_deg / 180.0) * math.pi
            self.theta_p = torch.clamp(
                self.theta_los + torch.randn(self.P, device=self.device) * as_std, 0.0, 0.5 * math.pi)
            self.phi_p = self.phi_los + torch.randn(self.P, device=self.device) * as_std
        else:
            self.theta_p = torch.rand(self.P, device=self.device) * 0.5 * math.pi
            self.phi_p = torch.rand(self.P, device=self.device) * 2.0 * math.pi

        self.fD_p = torch.ones(self.P, device=self.device) * float(self.fD_ut)
        self.fD_p += torch.randn(self.P, device=self.device) * self.cfg.doppler_path_std_hz
        self.phase_p = torch.empty(self.P, device=self.device).uniform_(0, 2 * math.pi)
        self.tau_p = torch.empty(self.P, device=self.device).uniform_(0, 200e-9)
        self.g = complex_normal((self.P,), device=self.device, std=1.0)
        self.u_los = self._upa_response(self.theta_los, self.phi_los)
        self.u_p = torch.stack([self._upa_response(self.theta_p[i], self.phi_p[i]) for i in range(self.P)])

    def step(self):
        Ts = self.cfg.Ts_s
        if self.cfg.aoa_rw_std_deg > 0:
            rw = (self.cfg.aoa_rw_std_deg / 180.0) * math.pi
            self.theta_los = torch.clamp(self.theta_los + torch.randn_like(self.theta_los) * rw, 0, 0.5 * math.pi)
            self.phi_los += torch.randn_like(self.phi_los) * rw
            self.theta_p = torch.clamp(self.theta_p + torch.randn_like(self.theta_p) * (0.7 * rw), 0, 0.5 * math.pi)
            self.phi_p += torch.randn_like(self.phi_p) * (0.7 * rw)
            self.u_los = self._upa_response(self.theta_los, self.phi_los)
            self.u_p = torch.stack([self._upa_response(self.theta_p[i], self.phi_p[i]) for i in range(self.P)])
        if self.cfg.doppler_rw_std_hz > 0:
            self.fD_p += torch.randn_like(self.fD_p) * self.cfg.doppler_rw_std_hz
            self.fD_ut = float(self.fD_ut + random.gauss(0, self.cfg.doppler_rw_std_hz))
        if self.cfg.doppler_jump_prob > 0 and random.random() < self.cfg.doppler_jump_prob:
            jump = random.gauss(0, self.cfg.doppler_jump_std_hz)
            self.fD_ut = float(self.fD_ut + jump)
            self.fD_p += (torch.randn_like(self.fD_p) * 0.3 + 1.0) * jump
        pn = self.cfg.phase_noise_std_rad
        self.phase_los = float(self.phase_los + 2 * math.pi * self.fD_ut * Ts + (random.gauss(0, pn) if pn > 0 else 0))
        self.phase_p = self.phase_p + 2 * math.pi * self.fD_p * Ts + (torch.randn_like(self.phase_p) * pn if pn > 0 else 0)
        if self.cfg.gain_ar_std > 0 and self.cfg.gain_ar_rho < 1.0:
            rho = self.cfg.gain_ar_rho
            self.g = rho * self.g + math.sqrt(max(1e-6, 1 - rho * rho)) * complex_normal(self.g.shape, device=self.device, std=self.cfg.gain_ar_std)

    def get_h_true(self):
        los = self.w_los * torch.exp(1j * torch.tensor(self.phase_los, device=self.device)) * self.u_los
        coeff = self.g * torch.exp(1j * (self.phase_p - 2 * math.pi * self.f0 * self.tau_p))
        nlos = self.w_nlos * (coeff[:, None] * self.u_p).sum(dim=0) / math.sqrt(self.P)
        return los + nlos


def mmse_estimate_from_pilot(h_true, snr_db, cfg, device):
    snr_lin = 10 ** (snr_db / 10.0)
    noise_std = float(torch.sqrt(torch.mean(torch.abs(h_true) ** 2).real / snr_lin).item())
    if cfg.pilot_outlier_prob > 0 and random.random() < cfg.pilot_outlier_prob:
        noise_std *= cfg.pilot_outlier_scale
    return (snr_lin / (1 + snr_lin)) * (h_true + complex_normal(h_true.shape, device=device, std=noise_std))


# ─────────────────────────────────────────────
# 3) Dataset
# ─────────────────────────────────────────────
def dataset_path(base_dir, cfg, A):
    tag = (f"q{cfg.q_in}_A{A}_L{cfg.w_out}_"
           f"dop{safe_float_tag(cfg.ut_doppler_hz_min)}to{safe_float_tag(cfg.ut_doppler_hz_max)}_"
           f"coh{safe_float_tag(cfg.coherence_ms)}_snr{int(cfg.pilot_snr_db)}_"
           f"K{safe_float_tag(cfg.rician_K_db)}_P{cfg.num_paths}_"
           f"teff{safe_float_tag(cfg.tau_eff_ms)}_alt{cfg.fixed_alt_km}")
    return os.path.join(base_dir, "datasets", f"dataset_{tag}.pt")


def generate_dataset_tensors(cfg, n_samples, A):
    print(f"⚡ Generating {n_samples} samples | q={cfg.q_in}, A={A}, L={cfg.w_out}, "
          f"dop={cfg.ut_doppler_hz_min}~{cfg.ut_doppler_hz_max} Hz")
    cfg_gen = replace(cfg, device="cpu")
    chan = LEOMassiveMIMOChannel(cfg_gen, "cpu")
    D = cfg.feat_dim
    total = cfg.q_in + A + cfg.w_out
    out_dtype = torch.float16 if cfg.dataset_store_fp16 else torch.float32
    X = torch.empty((n_samples, cfg.q_in, D), dtype=out_dtype)
    Y = torch.empty((n_samples, cfg.w_out, D), dtype=out_dtype)
    t0 = time.time()

    for i in range(n_samples):
        chan.reset()
        for _ in range(random.randint(5, 10)):
            chan.step()
        seq_obs, seq_true = [], []
        for _ in range(total):
            h = chan.get_h_true()
            y = mmse_estimate_from_pilot(h, cfg.pilot_snr_db, cfg, torch.device("cpu"))
            seq_true.append(vectorize_complex(h))
            seq_obs.append(vectorize_complex(y))
            chan.step()
        seq_obs = torch.stack(seq_obs)
        seq_true = torch.stack(seq_true)
        x_raw = seq_obs[:cfg.q_in]
        y_raw = seq_true[cfg.q_in + A: cfg.q_in + A + cfg.w_out]

        if cfg.use_rms_norm:
            x_n, sc = rms_normalize(x_raw)
            if (not torch.isfinite(x_n).all()) or (not torch.isfinite(sc)) or sc < 1e-6:
                x_n = torch.nan_to_num(x_raw)
                sc = torch.tensor(1.0)
            y_n = y_raw / (sc + 1e-8)
            X[i] = torch.clamp(torch.nan_to_num(x_n), -10, 10).to(out_dtype)
            Y[i] = torch.clamp(torch.nan_to_num(y_n), -10, 10).to(out_dtype)
        else:
            X[i] = torch.nan_to_num(x_raw).to(out_dtype)
            Y[i] = torch.nan_to_num(y_raw).to(out_dtype)

        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{n_samples} done | {time.time()-t0:.1f}s")
    return X.contiguous(), Y.contiguous()


def get_or_make_dataset(base_dir, cfg, A, force=False):
    ensure_dir(os.path.join(base_dir, "datasets"))
    path = dataset_path(base_dir, cfg, A)
    print(f"🗂️ Dataset path: {path}")
    if os.path.exists(path):
        if force:
            print("♻️ force_data=True -> regenerating dataset cache")
        else:
            print(f"✅ Using cached dataset: {path}")
            return path, torch.load(path, map_location="cpu", weights_only=False)
    else:
        print("🆕 No cached dataset found -> generating a new dataset")
    Xtr, Ytr = generate_dataset_tensors(cfg, cfg.train_samples, A)
    Xva, Yva = generate_dataset_tensors(cfg, cfg.val_samples, A)
    pack = {"cfg": asdict(cfg), "A": int(A),
            "train": {"X": Xtr, "Y": Ytr}, "val": {"X": Xva, "Y": Yva}}
    torch.save(pack, path)
    print(f"📦 Saved dataset: {path} | {file_exists_and_size(path)}")
    return path, pack


class SliceDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.Y[i]


# ─────────────────────────────────────────────
# 4) KalmanFilter (diag AR1) — unit-circle fix
# ─────────────────────────────────────────────
def complex_abs2(z):
    return z.real * z.real + z.imag * z.imag


class KalmanFilterDiagAR1:
    """
    Diagonal complex AR(1) Kalman filter.
    ★ Fixes: Doppler-based init, |a|≡1 (unit circle), phase-only adaptive update.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.M = cfg.M
        self.D = cfg.feat_dim
        snr_lin = 10 ** (cfg.pilot_snr_db / 10.0)
        self.beta = float(snr_lin / (1 + snr_lin))
        self.R = float(max(cfg.kf_R_floor, cfg.kf_R_scale / max(1e-12, snr_lin)))
        self.Q = float(max(0, cfg.kf_Q))
        # Doppler-based init
        fd = 0.5 * (cfg.ut_doppler_hz_min + cfg.ut_doppler_hz_max)
        phi0 = 2 * math.pi * fd * cfg.Ts_s
        self.a = torch.full((self.M,), complex(math.cos(phi0), math.sin(phi0)), dtype=torch.complex64)
        self.reset()

    def reset(self):
        self.x = None
        self.P = torch.full((self.M,), self.cfg.kf_init_P, dtype=torch.float32)
        self.prev_z = None

    def _prep_meas(self, y_vec):
        z = devectorize_complex(y_vec.float()).cpu().to(torch.complex64)
        return z / max(1e-12, self.beta) if self.cfg.kf_debias_mmse else z

    def _update_a(self, z):
        if self.cfg.kf_a_mode != "adaptive_ls":
            return
        if self.prev_z is None:
            self.prev_z = z.clone()
            return
        eps = 1e-8
        denom = complex_abs2(self.prev_z) + eps
        phase_hat = torch.angle((torch.conj(self.prev_z) * z) / denom)
        num_g = torch.sum(torch.conj(self.prev_z) * z)
        den_g = torch.sum(complex_abs2(self.prev_z)) + eps
        global_phase = torch.angle(num_g / den_g).item()
        small = denom < 1e-6
        if torch.any(small):
            phase_hat = phase_hat.clone()
            phase_hat[small] = global_phase
        lam = self.cfg.kf_a_smooth
        # |a| ≡ 1 always (unit circle)
        self.a = torch.exp(1j * ((1 - lam) * torch.angle(self.a) + lam * phase_hat)).to(torch.complex64)
        self.prev_z = z.clone()

    def update(self, y_vec):
        z = self._prep_meas(y_vec)
        self._update_a(z)
        if self.x is None:
            self.x = z.clone()
            return
        x_pri = self.a * self.x
        P_pri = self.P + self.Q
        innov = z - x_pri
        S = P_pri + self.R
        K = P_pri / (S + 1e-12)
        self.x = x_pri + K * innov
        self.P = (1 - K) * P_pri

    def predict_ahead(self, steps):
        if self.x is None:
            return torch.zeros(self.D)
        # unit-magnitude: no collapse
        a_pow = torch.exp(1j * torch.angle(self.a) * steps).to(torch.complex64)
        return vectorize_complex(a_pow * self.x).cpu()


# ─────────────────────────────────────────────
# 5) Neural Models
# ─────────────────────────────────────────────
def rotate_realimag(x, rho, phi):
    M = x.shape[-1] // 2
    re, im = x[:, :M], x[:, M:]
    c, s = torch.cos(phi), torch.sin(phi)
    return torch.cat([rho * (c * re - s * im), rho * (s * re + c * im)], dim=1)


# --- Feature extraction (shared) ---
def extract_y_dy_ddy(x_in):
    """Build [y, dy, ddy] features for each time step."""
    feats = []
    y_prev = x_in[:, 0, :]
    dy_prev = torch.zeros_like(y_prev)
    for t in range(x_in.shape[1]):
        y_t = x_in[:, t, :]
        dy = y_t - (y_prev if t > 0 else y_t)
        ddy = dy - (dy_prev if t > 0 else dy)
        feats.append(torch.cat([y_t, dy, ddy], 1).unsqueeze(1))
        dy_prev = dy
        y_prev = y_t
    return torch.cat(feats, 1)


# --- KalmanNet (GRU) ---
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
            feat = torch.cat([y_t - y_prev, y_t - x_post, prev_upd], 1)
            h = self.ln(self.gru(torch.tanh(self.fc_feat(feat)), h))
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
            h_r = self.ln(self.roll_gru(torch.cat([h_r, curr], 1), h_r))
            rp = self.fc_rp_r(h_r)
            rho = torch.sigmoid(rp[:, 0:1]) * 1.25
            phi = math.pi * torch.tanh(rp[:, 1:2])
            curr = rotate_realimag(curr, rho, phi)
            preds.append(curr.unsqueeze(1))
        return torch.cat(preds, 1)


# --- TDNN Encoder ---
class TCNResBlock(nn.Module):
    def __init__(self, ch, k, dilation, drop):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, dilation=dilation, padding=pad, padding_mode="replicate"),
            nn.GELU(), nn.Dropout(drop),
            nn.Conv1d(ch, ch, k, dilation=dilation, padding=pad, padding_mode="replicate"),
            nn.GELU(), nn.Dropout(drop),
        )
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):
        return self.ln((x + self.net(x)).transpose(1, 2)).transpose(1, 2)


class TDNNEncoderTCN(nn.Module):
    def __init__(self, in_ch, H, blocks, k, drop):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, H, 1)
        self.blocks = nn.ModuleList([TCNResBlock(H, k, 2 ** i, drop) for i in range(blocks)])
        self.out_ln = nn.LayerNorm(H)

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        return self.out_ln(h.transpose(1, 2))


# --- ConvNeXt V2 Encoder (CVPR 2023) ---
class LayerNorm1d(nn.Module):
    """Channel-last LayerNorm for (B, C, T) tensors."""
    def __init__(self, ch):
        super().__init__()
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class GRN1d(nn.Module):
    """
    Global Response Normalization (ConvNeXt V2).
    Prevents feature collapse by enhancing inter-channel competition.
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=2, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """DW Conv → LN → PW expand → GELU → GRN → PW shrink → + residual"""
    def __init__(self, ch, kernel=9, expand=2, drop=0.1):
        super().__init__()
        hidden = ch * expand
        pad = kernel // 2
        self.dw = nn.Conv1d(ch, ch, kernel, padding=pad, groups=ch, padding_mode="replicate")
        nn.init.trunc_normal_(self.dw.weight, std=0.02)
        if self.dw.bias is not None:
            nn.init.zeros_(self.dw.bias)
        self.norm = LayerNorm1d(ch)
        self.pw1 = nn.Conv1d(ch, hidden, 1)
        self.act = nn.GELU()
        self.grn = GRN1d(hidden)
        self.pw2 = nn.Conv1d(hidden, ch, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        residual = x
        y = self.dw(x)
        y = self.norm(y)
        y = self.pw1(y)
        y = self.act(y)
        y = self.grn(y)
        y = self.pw2(y)
        y = self.drop(y)
        return residual + y


class ConvNeXtV2Encoder(nn.Module):
    def __init__(self, in_ch, H, blocks, kernel=9, expand=2, drop=0.1):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, H, 1)
        self.blocks = nn.ModuleList([ConvNeXtV2Block(H, kernel, expand, drop) for _ in range(blocks)])
        self.out_ln = nn.LayerNorm(H)

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        return self.out_ln(h.transpose(1, 2))


# --- Kalman-style predictor (shared backbone) ---
class _KalmanNetBase(nn.Module):
    """Shared predict-correct + rollout logic."""
    def __init__(self, D, H, feature_mode, prior_mode, gain_mode, rollout_mode):
        super().__init__()
        self.D, self.H = D, H
        self.feature_mode = feature_mode
        self.prior_mode = prior_mode
        self.gain_mode = gain_mode
        self.rollout_mode = rollout_mode

        self.fc_gain = nn.Linear(H, D)
        self.fc_rp = nn.Linear(H, 2)
        self.roll_in = nn.Linear(H + D, H)
        self.roll_gru = nn.GRUCell(H, H)
        self.roll_ln = nn.LayerNorm(H)
        self.fc_rp_r = nn.Linear(H, 2)

    def _features(self, x_in):
        return extract_y_dy_ddy(x_in) if self.feature_mode == "y_dy_ddy" else x_in

    def _rho_phi(self, h):
        return torch.sigmoid(h[:, 0:1]) * 1.25, math.pi * torch.tanh(h[:, 1:2])

    def _prior(self, x, rho, phi):
        return x if self.prior_mode == "identity" else rotate_realimag(x, rho, phi)

    def _gain(self, h):
        if self.gain_mode == "fixed":
            return torch.full((h.shape[0], self.D), 0.5, device=h.device)
        return torch.sigmoid(self.fc_gain(h))

    def _kalman_loop(self, x_in, h_seq, w_out):
        B, q, D = x_in.shape
        x_post = x_in[:, 0, :].clone()
        for t in range(q):
            rho, phi = self._rho_phi(self.fc_rp(h_seq[:, t, :]))
            x_pri = self._prior(x_post, rho, phi)
            x_post = x_pri + self._gain(h_seq[:, t, :]) * (x_in[:, t, :] - x_pri)

        preds = []
        h_r = h_seq[:, -1, :].clone()
        curr = x_post
        if self.rollout_mode == "gru":
            for _ in range(w_out):
                h_r = self.roll_ln(self.roll_gru(
                    torch.tanh(self.roll_in(torch.cat([h_r, curr], 1))), h_r))
                rho, phi = self._rho_phi(self.fc_rp_r(h_r))
                curr = self._prior(curr, rho, phi)
                preds.append(curr.unsqueeze(1))
        else:
            rho, phi = self._rho_phi(self.fc_rp_r(h_r))
            for _ in range(w_out):
                curr = self._prior(curr, rho, phi)
                preds.append(curr.unsqueeze(1))
        return torch.cat(preds, 1)


class TDNNKalmanNet(_KalmanNetBase):
    def __init__(self, D, H=192, blocks=2, k=3, drop=0.1,
                 feature_mode="y_dy_ddy", prior_mode="rotation",
                 gain_mode="learned", rollout_mode="gru"):
        super().__init__(D, H, feature_mode, prior_mode, gain_mode, rollout_mode)
        in_mult = 3 if feature_mode == "y_dy_ddy" else 1
        self.enc = TDNNEncoderTCN(in_mult * D, H, blocks, k, drop)

    def forward(self, x_in, w_out):
        return self._kalman_loop(x_in, self.enc(self._features(x_in)), w_out)


class ConvNeXtV2KalmanNet(_KalmanNetBase):
    def __init__(self, D, H=192, blocks=2, kernel=9, expand=2, drop=0.1,
                 feature_mode="y_dy_ddy", prior_mode="rotation",
                 gain_mode="learned", rollout_mode="gru"):
        super().__init__(D, H, feature_mode, prior_mode, gain_mode, rollout_mode)
        in_mult = 3 if feature_mode == "y_dy_ddy" else 1
        self.enc = ConvNeXtV2Encoder(in_mult * D, H, blocks, kernel, expand, drop)

    def forward(self, x_in, w_out):
        return self._kalman_loop(x_in, self.enc(self._features(x_in)), w_out)


# --- SCP (LSTM Seq2Seq) ---
class SCP_Seq2Seq(nn.Module):
    def __init__(self, D, w_out, H=256, layers=2, dropout=0.25):
        super().__init__()
        self.D, self.w_out, self.H = D, w_out, H
        self.enc = nn.LSTM(D, H, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0)
        self.dec_cell = nn.LSTMCell(2 * H, H)
        self.inp_proj = nn.Linear(D, H)
        self.out_proj = nn.Linear(2 * H, D)
        self.drop = nn.Dropout(dropout)

    def _attend(self, enc_out, h):
        w = torch.softmax(torch.bmm(enc_out, h.unsqueeze(2)).squeeze(2) / math.sqrt(self.H), 1).unsqueeze(2)
        return torch.sum(w * enc_out, 1)

    def forward(self, x_in, w_out, y_teacher=None, tf_ratio=0.0):
        enc_out, (h_n, c_n) = self.enc(x_in)
        h, c = h_n[-1], c_n[-1]
        prev_y = x_in[:, -1, :]
        outs = []
        for t in range(w_out):
            ctx = self._attend(enc_out, h)
            h, c = self.dec_cell(torch.cat([self.inp_proj(prev_y), ctx], 1), (h, c))
            y_t = self.out_proj(torch.cat([self.drop(h), ctx], 1))
            outs.append(y_t.unsqueeze(1))
            prev_y = y_teacher[:, t, :] if y_teacher is not None and tf_ratio > 0 and random.random() < tf_ratio else y_t
        return torch.cat(outs, 1)


def build_model_suite(cfg, D, tdnn_blocks, cnv2_blocks, run_ablation=False):
    models = {
        "ConvNeXtV2-KalmanNet": ConvNeXtV2KalmanNet(
            D, cfg.cnv2_hidden, cnv2_blocks, cfg.cnv2_kernel, cfg.cnv2_expand, cfg.cnv2_drop),
        "TDNN-KalmanNet": TDNNKalmanNet(
            D, cfg.tdnn_hidden, tdnn_blocks, cfg.tdnn_kernel, cfg.tdnn_drop),
        "KalmanNet": KalmanNetGRU(D, cfg.knet_hidden),
        "SCP": SCP_Seq2Seq(D, cfg.w_out, cfg.scp_hidden, cfg.scp_layers, cfg.dropout),
    }
    if run_ablation:
        models["TDNN-noDelta"] = TDNNKalmanNet(
            D, cfg.tdnn_hidden, tdnn_blocks, cfg.tdnn_kernel, cfg.tdnn_drop, feature_mode="y_only")
        models["TDNN-noPrior"] = TDNNKalmanNet(
            D, cfg.tdnn_hidden, tdnn_blocks, cfg.tdnn_kernel, cfg.tdnn_drop, prior_mode="identity")
        models["TDNN-noRollGRU"] = TDNNKalmanNet(
            D, cfg.tdnn_hidden, tdnn_blocks, cfg.tdnn_kernel, cfg.tdnn_drop, rollout_mode="iter_rotation")
    return models


# ─────────────────────────────────────────────
# 6) Training — warmup + cosine LR
# ─────────────────────────────────────────────
def train_model(cfg, model, name, outdir, train_ds, val_ds, num_workers=0):
    print(f"\n🧠 Training {name} | q={cfg.q_in} L={cfg.w_out} ep={cfg.epochs} bs={cfg.batch_size}")
    model = model.to(cfg.device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    warmup_ep = cfg.warmup_epochs
    def lr_lambda(ep):
        if ep < warmup_ep:
            return float(ep + 1) / float(max(1, warmup_ep))
        progress = float(ep - warmup_ep) / float(max(1, cfg.epochs - warmup_ep))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    pin = cfg.device.startswith("cuda")
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          pin_memory=pin, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        pin_memory=pin, num_workers=num_workers)
    use_amp = cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    w_h = cfg.horizon_loss_weights().to(cfg.device).view(1, cfg.w_out, 1)
    best_val, bad = 1e18, 0
    best_path = os.path.join(outdir, f"best_{name}.pt")

    for ep in range(cfg.epochs):
        model.train()
        tr, nb = 0.0, 0
        for x, y in train_dl:
            x = torch.nan_to_num(x.to(cfg.device, non_blocking=True).float())
            y = torch.nan_to_num(y.to(cfg.device, non_blocking=True).float())
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    pred = (model(x, cfg.w_out, y_teacher=y, tf_ratio=cfg.scp_tf_ratio)
                            if isinstance(model, SCP_Seq2Seq) else model(x, cfg.w_out))
                    loss = torch.mean(((pred - y) ** 2) * w_h)
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                pred = (model(x, cfg.w_out, y_teacher=y, tf_ratio=cfg.scp_tf_ratio)
                        if isinstance(model, SCP_Seq2Seq) else model(x, cfg.w_out))
                loss = torch.mean(((pred - y) ** 2) * w_h)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tr += float(loss.item())
            nb += 1
        tr /= max(1, nb)
        scheduler.step()

        model.eval()
        va, nv = 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x = torch.nan_to_num(x.to(cfg.device, non_blocking=True).float())
                y = torch.nan_to_num(y.to(cfg.device, non_blocking=True).float())
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        pred = (model(x, cfg.w_out) if not isinstance(model, SCP_Seq2Seq)
                                else model(x, cfg.w_out, None, 0))
                        loss = torch.mean(((pred - y) ** 2) * w_h)
                else:
                    pred = (model(x, cfg.w_out) if not isinstance(model, SCP_Seq2Seq)
                            else model(x, cfg.w_out, None, 0))
                    loss = torch.mean(((pred - y) ** 2) * w_h)
                if not torch.isfinite(loss):
                    continue
                va += float(loss.item())
                nv += 1
        va /= max(1, nv)

        if va < best_val:
            best_val, bad = va, 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1

        cur_lr = scheduler.get_last_lr()[0]
        if ep % 25 == 0 or ep == cfg.epochs - 1:
            print(f"[{name}] ep={ep:03d} train={tr:.6f} val={va:.6f} best={best_val:.6f} "
                  f"lr={cur_lr:.2e} (bad={bad}/{cfg.early_stop_patience})")
        if bad >= cfg.early_stop_patience:
            print(f"🟡 Early stop for {name} at ep={ep}")
            break

    model.load_state_dict(torch.load(best_path, map_location=cfg.device, weights_only=False))
    print(f"✅ best: {best_path} | {file_exists_and_size(best_path)}")
    return model


# ─────────────────────────────────────────────
# 7) Evaluation
# ─────────────────────────────────────────────
@torch.no_grad()
def eval_nmse_horizon(cfg, models, val_ds, A, trials=2000, kf=None):
    nmse = {k: np.zeros(cfg.w_out) for k in models}
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
                [kf.predict_ahead(A + i + 1) for i in range(cfg.w_out)]
            ).to(cfg.device).float()

        for name, m in models.items():
            if name == "Outdated":
                preds[name] = x_in[-1].unsqueeze(0).repeat(cfg.w_out, 1)
            elif name == "KalmanFilter":
                continue
            elif isinstance(m, SCP_Seq2Seq):
                preds[name] = m(x_in.unsqueeze(0), cfg.w_out, None, 0)[0]
            else:
                preds[name] = m(x_in.unsqueeze(0), cfg.w_out)[0]

        for t in range(cfg.w_out):
            for name in models:
                nmse[name][t] += nmse_db(preds[name][t], y_true[t])

    for k in nmse:
        nmse[k] /= len(idxs)
    return nmse


# ─────────────────────────────────────────────
# 8) Plotting
# ─────────────────────────────────────────────
@torch.no_grad()
def channel_corr_curve(cfg, max_lag, trials=350):
    chan = LEOMassiveMIMOChannel(replace(cfg, device="cpu"))
    corr = np.zeros(max_lag + 1)
    for _ in range(trials):
        chan.reset()
        for _ in range(10):
            chan.step()
        h_seq = []
        for _ in range(max_lag + 1):
            h_seq.append(chan.get_h_true().squeeze(0))
            chan.step()
        h_seq = torch.stack(h_seq)
        h0 = h_seq[0]
        n0 = torch.norm(h0)
        for tau in range(max_lag + 1):
            corr[tau] += (torch.abs(torch.vdot(h0, h_seq[tau])) / (n0 * torch.norm(h_seq[tau]) + 1e-12)).item()
    return corr / trials


def plot_corr_curve(cfg, A, corr, save_path, title):
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(corr)), corr, 'b-', lw=2)
    plt.axvline(A, color='r', ls='--', label=f'A={A}')
    plt.grid(alpha=0.3); plt.xlabel("Lag (slots)"); plt.ylabel("Correlation")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=200); plt.close()


def save_csv_nmse_raw(path, nmse_curve):
    keys = list(nmse_curve.keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + keys)
        for i in range(len(next(iter(nmse_curve.values())))):
            w.writerow([i + 1] + [float(nmse_curve[k][i]) for k in keys])


def plot_nmse_curve_w_axis(cfg, A, nmse_curve, save_path, title):
    w_axis = A + np.arange(1, cfg.w_out + 1)
    plt.figure(figsize=(10, 6))
    for k, v in nmse_curve.items():
        plt.plot(w_axis, v, marker='o', lw=2, label=k)
    plt.grid(alpha=0.3); plt.xlabel("w = A+t (coherence intervals)"); plt.ylabel("NMSE (dB)")
    delay_ms, _ = cfg.effective_delay_steps()
    plt.title(title + f"\ntau={delay_ms:.1f}ms Ts={cfg.coherence_ms:.1f}ms "
              f"dop={cfg.ut_doppler_hz_min}-{cfg.ut_doppler_hz_max}Hz A={A}")
    plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=200); plt.close()


def plot_bar(values, title, ylabel, save_path):
    plt.figure(figsize=(10, 4.8))
    plt.bar(list(values.keys()), list(values.values()))
    plt.grid(axis='y', alpha=0.3); plt.ylabel(ylabel); plt.title(title)
    plt.xticks(rotation=15, ha='right'); plt.tight_layout()
    plt.savefig(save_path, dpi=200); plt.close()


# ─────────────────────────────────────────────
# 9) Latency
# ─────────────────────────────────────────────
@torch.no_grad()
def measure_latency_gpu(model, cfg, q, D, iters=100):
    if not cfg.device.startswith("cuda"):
        return float("nan")
    model = model.eval().to(cfg.device)
    x = torch.randn(1, q, D, device=cfg.device)
    for _ in range(20):
        _ = model(x, cfg.w_out) if not isinstance(model, SCP_Seq2Seq) else model(x, cfg.w_out, None, 0)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        _ = model(x, cfg.w_out) if not isinstance(model, SCP_Seq2Seq) else model(x, cfg.w_out, None, 0)
    e.record(); torch.cuda.synchronize()
    return float(s.elapsed_time(e) / iters)


@torch.no_grad()
def measure_latency_cpu(model, cfg, q, D, iters=80):
    model = model.eval().to("cpu")
    x = torch.randn(1, q, D)
    for _ in range(20):
        _ = model(x, cfg.w_out) if not isinstance(model, SCP_Seq2Seq) else model(x, cfg.w_out, None, 0)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x, cfg.w_out) if not isinstance(model, SCP_Seq2Seq) else model(x, cfg.w_out, None, 0)
    return (time.perf_counter() - t0) * 1000 / iters


def approx_flops_models(cfg, tdnn_blocks, cnv2_blocks):
    D = cfg.feat_dim; Ht = cfg.tdnn_hidden; Hc = cfg.cnv2_hidden
    Hg = cfg.knet_hidden; Hs = cfg.scp_hidden
    q = cfg.q_in; L = cfg.w_out; k = cfg.tdnn_kernel; kc = cfg.cnv2_kernel
    flops = {}

    # TDNN
    f = 2 * q * (3 * D) * Ht
    for _ in range(tdnn_blocks):
        f += 2 * (2 * q * Ht * Ht * k)
    f += q * (2 * Ht * D + 2 * Ht * 2)
    f += L * (2 * (Ht + D) * Ht + 2 * 3 * (Ht * Ht + Ht * Ht) + 2 * Ht * 2)
    flops["TDNN-KalmanNet"] = f / 1e9

    # ConvNeXtV2
    f = 2 * q * (3 * D) * Hc
    for _ in range(cnv2_blocks):
        f += 2 * q * Hc * kc + 2 * q * Hc * (Hc * cfg.cnv2_expand) + 2 * q * (Hc * cfg.cnv2_expand) * Hc
    f += q * (2 * Hc * D + 2 * Hc * 2)
    f += L * (2 * (Hc + D) * Hc + 2 * 3 * (Hc * Hc + Hc * Hc) + 2 * Hc * 2)
    flops["ConvNeXtV2-KalmanNet"] = f / 1e9

    # KalmanNet
    f = q * (2 * 3 * D * Hg) + q * (2 * 3 * (Hg * Hg + Hg * Hg)) + q * (2 * Hg * D + 2 * Hg * 2)
    f += L * (2 * 3 * ((Hg + D) * Hg + Hg * Hg)) + L * (2 * Hg * 2)
    flops["KalmanNet"] = f / 1e9

    # SCP
    f = q * 2 * 4 * (D * Hs + Hs * Hs) + L * 2 * (q * Hs)
    f += L * 2 * 4 * ((2 * Hs) * Hs + Hs * Hs) + L * (2 * 2 * Hs * D)
    flops["SCP"] = f / 1e9

    return flops


# ─────────────────────────────────────────────
# 10) Experiment runner
# ─────────────────────────────────────────────
def train_and_eval_one(base_dir, cfg, exp_tag, force_data=False, num_workers=0, run_ablation=False):
    outdir = os.path.join(base_dir, "results", exp_tag)
    ensure_dir(outdir)
    log_path = os.path.join(outdir, "run.log")
    log_f = open(log_path, "w", encoding="utf-8")
    orig_stdout = sys.stdout
    sys.stdout = Tee(orig_stdout, log_f)

    try:
        print("🧾 Folder:", outdir)
        delay_ms, A = cfg.effective_delay_steps()
        print(f"⏱️ tau_eff={delay_ms:.2f}ms Ts={cfg.coherence_ms:.2f}ms → A={A}")

        with open(os.path.join(outdir, "cfg.json"), "w") as f:
            json.dump({**asdict(cfg), "A": A, "delay_ms": delay_ms}, f, indent=2)

        _, pack = get_or_make_dataset(base_dir, cfg, A, force_data)
        train_ds = SliceDataset(pack["train"]["X"], pack["train"]["Y"])
        val_ds = SliceDataset(pack["val"]["X"], pack["val"]["Y"])

        D = cfg.feat_dim
        tb = cfg.pick_tdnn_blocks()
        cb = cfg.pick_cnv2_blocks()
        print(f"🧩 TDNN blocks={tb}, ConvNeXtV2 blocks={cb} (q={cfg.q_in})")

        model_suite = build_model_suite(cfg, D, tb, cb, run_ablation)

        t0 = time.time()
        trained = {}
        for name, model in model_suite.items():
            trained[name] = train_model(cfg, model, name, outdir, train_ds, val_ds, num_workers).eval()
        print(f"⏱️ Total training: {(time.time()-t0)/60:.2f} min")

        # Eval
        kf = KalmanFilterDiagAR1(cfg)
        eval_models = {
            "ConvNeXtV2-KalmanNet": trained["ConvNeXtV2-KalmanNet"],
            "TDNN-KalmanNet": trained["TDNN-KalmanNet"],
            "KalmanNet": trained["KalmanNet"],
            "SCP": trained["SCP"],
            "KalmanFilter": None,
            "Outdated": None,
        }
        if run_ablation:
            for k in ("TDNN-noDelta", "TDNN-noPrior", "TDNN-noRollGRU"):
                if k in trained:
                    eval_models[k] = trained[k]

        nmse_curve = eval_nmse_horizon(cfg, eval_models, val_ds, A, cfg.horizon_trials, kf)

        # Plots
        if cfg.make_aging_plots:
            corr = channel_corr_curve(cfg, A + cfg.w_out + 5, 350)
            plot_corr_curve(cfg, A, corr, os.path.join(outdir, "corr.png"), "Channel Correlation vs Lag")
            show_image(os.path.join(outdir, "corr.png"))

        flops = approx_flops_models(cfg, tb, cb)
        plot_bar(flops, "Inference FLOPs", "GFLOPs", os.path.join(outdir, "flops.png"))
        show_image(os.path.join(outdir, "flops.png"))

        # CPU latency (main models only)
        cpu_lat = {}
        for n in ["ConvNeXtV2-KalmanNet", "TDNN-KalmanNet", "KalmanNet", "SCP"]:
            if n in trained:
                cpu_lat[n] = measure_latency_cpu(trained[n], cfg, cfg.q_in, D, 60)
        plot_bar(cpu_lat, "CPU Latency", "ms", os.path.join(outdir, "cpu_lat.png"))
        show_image(os.path.join(outdir, "cpu_lat.png"))

        save_csv_nmse_raw(os.path.join(outdir, "nmse_raw.csv"), nmse_curve)
        
        # 1) Plot ALL models (including ablation if enabled)
        plot_nmse_curve_w_axis(cfg, A, nmse_curve, os.path.join(outdir, "nmse_all.png"),
                               f"NMSE (All) | q={cfg.q_in} L={cfg.w_out} SNR={cfg.pilot_snr_db}dB")
        show_image(os.path.join(outdir, "nmse_all.png"))

        # 2) Plot ONLY MAIN models (excluding ablation)
        main_models_keys = ["ConvNeXtV2-KalmanNet", "TDNN-KalmanNet", "KalmanNet", "SCP", "KalmanFilter", "Outdated"]
        nmse_curve_main = {k: v for k, v in nmse_curve.items() if k in main_models_keys}
        
        plot_nmse_curve_w_axis(cfg, A, nmse_curve_main, os.path.join(outdir, "nmse_main.png"),
                               f"NMSE (Main Models) | q={cfg.q_in} L={cfg.w_out} SNR={cfg.pilot_snr_db}dB")
        show_image(os.path.join(outdir, "nmse_main.png"))

        params = {n: count_params(m) for n, m in trained.items()}
        params_main = {k: params[k] for k in ["ConvNeXtV2-KalmanNet", "TDNN-KalmanNet", "KalmanNet", "SCP"] if k in params}
        plot_bar(params_main, "Parameter Count", "#", os.path.join(outdir, "params.png"))
        show_image(os.path.join(outdir, "params.png"))

        size_mb = {}
        for n in ["ConvNeXtV2-KalmanNet", "TDNN-KalmanNet", "KalmanNet", "SCP"]:
            if n in trained:
                size_mb[f"{n}(fp32)"] = bytes_to_mb(state_dict_size_bytes(trained[n]))
                size_mb[f"{n}(fp16)"] = bytes_to_mb(state_dict_size_bytes(trained[n], True))
        plot_bar(size_mb, "Model Size", "MB", os.path.join(outdir, "size.png"))

        gpu_lat = {}
        if cfg.measure_latency and cfg.device.startswith("cuda"):
            for n in ["ConvNeXtV2-KalmanNet", "TDNN-KalmanNet", "KalmanNet", "SCP"]:
                if n in trained:
                    gpu_lat[n] = measure_latency_gpu(trained[n], cfg, cfg.q_in, D, 80)
            plot_bar(gpu_lat, "GPU Latency", "ms", os.path.join(outdir, "gpu_lat.png"))

        summary = {
            "delay_ms": delay_ms, "A": A,
            "mean_nmse": {k: float(np.mean(v)) for k, v in nmse_curve.items()},
            "nmse_w_first": {k: float(v[0]) for k, v in nmse_curve.items()},
            "nmse_w_last": {k: float(v[-1]) for k, v in nmse_curve.items()},
            "params": params, "flops": flops,
            "cpu_lat": cpu_lat, "gpu_lat": gpu_lat,
        }
        with open(os.path.join(outdir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70)
        print("📊 NMSE Summary")
        print("=" * 70)
        for k, v in nmse_curve.items():
            print(f"  {k:25s} @w={A+1}: {v[0]:+.2f}  @w={A+cfg.w_out}: {v[-1]:+.2f}  mean: {np.mean(v):+.2f} dB")
        print(f"\n📊 Params")
        for k, v in params.items():
            print(f"  {k:25s} {v:>10,}")

        zip_path = make_zip_of_folder(outdir, os.path.join(base_dir, "results", f"{exp_tag}.zip"))
        print(f"\n📦 {zip_path}")
        show_download_link(zip_path)
        return nmse_curve

    finally:
        sys.stdout = orig_stdout
        try:
            log_f.close()
        except Exception:
            pass


# ─────────────────────────────────────────────
# 11) Main
# ─────────────────────────────────────────────
def parse_q_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def apply_scenario(cfg, scenario):
    if scenario == "tdnn_adv":
        return replace(
            cfg, rician_K_db=3.0, num_paths=12, as_std_deg=2.0,
            aoa_rw_std_deg=0.06, doppler_path_std_hz=18.0, doppler_rw_std_hz=2.0,
            doppler_jump_prob=0.06, doppler_jump_std_hz=45.0, phase_noise_std_rad=0.02,
            gain_ar_rho=0.995, gain_ar_std=0.08, pilot_outlier_prob=0.02,
            loss_tail_weight=2.0, loss_first_boost=1.0,
            lr=min(cfg.lr, 8e-4), kf_Q=max(cfg.kf_Q, 1e-2))
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=DEFAULTS["base_dir"])
    parser.add_argument("--force_data", type=str2bool, nargs="?", const=True, default=DEFAULTS["force_data"])
    parser.add_argument("--q_list", default=DEFAULTS["q_list"])
    parser.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--tau_eff_ms", type=float, default=DEFAULTS["tau_eff_ms"])
    parser.add_argument("--alt_km", type=int, default=DEFAULTS["alt_km"])
    parser.add_argument("--w_out", type=int, default=DEFAULTS["w_out"])
    parser.add_argument("--dop_min", type=float, default=DEFAULTS["dop_min"])
    parser.add_argument("--dop_max", type=float, default=DEFAULTS["dop_max"])
    parser.add_argument("--Ts_ms", type=float, default=DEFAULTS["Ts_ms"])
    parser.add_argument("--scenario", default=DEFAULTS["scenario"], choices=["baseline", "tdnn_adv"])
    parser.add_argument("--no_latency", type=str2bool, nargs="?", const=True, default=DEFAULTS["no_latency"])
    parser.add_argument("--ablation", type=str2bool, nargs="?", const=True, default=DEFAULTS["ablation"])

    if in_notebook():
        args = parser.parse_args(args=[])
        print("ℹ️ Notebook mode: using DEFAULTS / parser defaults.")
    else:
        args = parser.parse_args()

    args.base_dir = resolve_base_dir(args.base_dir)
    ensure_dir(args.base_dir)
    ensure_dir(os.path.join(args.base_dir, "results"))
    ensure_dir(os.path.join(args.base_dir, "datasets"))
    seed_all(42)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    q_values = parse_q_list(args.q_list)
    print(f"📌 base_dir={args.base_dir}  device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"📌 q={q_values} tau={args.tau_eff_ms}ms dop={args.dop_min}-{args.dop_max}Hz")
    print(f"📌 scenario={args.scenario} ablation={args.ablation} force_data={args.force_data} no_latency={args.no_latency}")

    for q in q_values:
        cfg = replace(
            SimCfg(), q_in=q, w_out=args.w_out,
            tau_eff_ms=args.tau_eff_ms, fixed_alt_km=args.alt_km,
            ut_doppler_hz_min=args.dop_min, ut_doppler_hz_max=args.dop_max,
            coherence_ms=args.Ts_ms, measure_latency=not args.no_latency)
        cfg = apply_scenario(cfg, args.scenario)
        delay_ms, A = cfg.effective_delay_steps()

        tag = (f"{args.scenario}_q{q}_A{A}_L{cfg.w_out}_"
               f"dop{safe_float_tag(cfg.ut_doppler_hz_min)}to{safe_float_tag(cfg.ut_doppler_hz_max)}_"
               f"{timestamp_str()}")

        print(f"\n{'='*80}")
        print(f"🚀 {tag}")
        print(f"   TDNN blocks={cfg.pick_tdnn_blocks()} ConvNeXtV2 blocks={cfg.pick_cnv2_blocks()}")
        print(f"   tau={delay_ms:.1f}ms A={A} KF: Q={cfg.kf_Q} mag_clip={cfg.kf_mag_clip}")

        train_and_eval_one(args.base_dir, cfg, tag, args.force_data, args.num_workers, args.ablation)

    print("\n🎉 All done!")


if __name__ == "__main__":
    main()
