# -*- coding: utf-8 -*-
"""
Unified LEO Channel Prediction: TDNN-KalmanNet & Robustness Analysis (Fixed SimCfg)
"""

import os
import sys
import glob
import json
import csv
import math
import time
import random
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict, replace
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 0. Setup & Utils
# =========================================================

def setup_environment():
    # Check if in Colab
    if 'google.colab' in sys.modules:
        from google.colab import drive
        drive.mount('/content/drive')
        base_dir = "/content/drive/MyDrive/LEO_TDNN_Project"
    else:
        base_dir = "./results"

    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def complex_normal(shape, device, std=1.0):
    scale = std / 1.41421356
    re = torch.randn(shape, device=device) * scale
    im = torch.randn(shape, device=device) * scale
    return torch.complex(re, im)


def vectorize_complex(h: torch.Tensor) -> torch.Tensor:
    # (..., M) complex -> (..., 2M) real
    return torch.cat([h.real, h.imag], dim=-1)


def devectorize_complex(x: torch.Tensor) -> torch.Tensor:
    # (..., 2M) real -> (..., M) complex
    D = x.shape[-1]
    M = D // 2
    return torch.complex(x[..., :M], x[..., M:])


def rms_normalize(x: torch.Tensor, eps: float = 1e-8):
    # Normalize based on Input Power
    scale = torch.sqrt(torch.mean(x.float() ** 2) + eps)
    return x / (scale + eps), scale


def mmse_estimate(h_true: torch.Tensor, snr_db: float, device: torch.device):
    snr_lin = 10 ** (snr_db / 10.0)
    # Signal power is approx 1 due to channel model
    noise_std = math.sqrt(1.0 / snr_lin)
    noise = complex_normal(h_true.shape, device=device, std=noise_std)
    y = h_true + noise
    # LMMSE shrinkage (simplified)
    beta = snr_lin / (1.0 + snr_lin)
    return beta * y


def nmse_db(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    # pred, target: Real vectors (B, D)
    hp = devectorize_complex(pred.float())
    ht = devectorize_complex(target.float())
    err = torch.sum(torch.abs(hp - ht) ** 2, dim=-1)
    pwr = torch.sum(torch.abs(ht) ** 2, dim=-1)
    nmse = 10 * torch.log10(torch.mean(err / (pwr + eps)))
    return nmse.item()


# =========================================================
# 1. Configuration (Fixed)
# =========================================================
@dataclass
class SimCfg:
    # System
    Mx: int = 16
    My: int = 16
    rician_K_db: float = 10.0
    pilot_snr_db: float = 15.0  # Training SNR

    # Motion (Default for Training)
    max_ut_doppler_hz: float = 100.0
    coherence_ms: float = 1.0

    # Sequence
    q_in: int = 10  # Will be swept [5, 10, 30]
    w_out: int = 30  # Fixed as requested

    # Training (Enhanced)
    train_samples: int = 40000
    test_samples: int = 3000
    batch_size: int = 128
    epochs: int = 150
    lr: float = 1e-3

    # Model Params
    hidden_dim: int = 256
    scp_hidden: int = 256  # SCP Specific

    # TDNN Specific
    tdnn_context: int = 5

    # [Fixed] Missing attribute added
    use_rms_norm: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def M(self):
        return self.Mx * self.My

    @property
    def feat_dim(self):
        return 2 * self.M

    @property
    def dt_s(self):
        return self.coherence_ms * 1e-3


# =========================================================
# 2. Channel Model
# =========================================================
class LEOMassiveMIMOChannel:
    def __init__(self, cfg: SimCfg, override_doppler=None):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.P = 6

        # Rician Factors
        K_lin = 10 ** (cfg.rician_K_db / 10.0)
        self.w_los = math.sqrt(K_lin / (K_lin + 1.0))
        self.w_nlos = math.sqrt(1.0 / (K_lin + 1.0))

        # Doppler setting
        self.doppler_max = override_doppler if override_doppler else cfg.max_ut_doppler_hz

        self.reset()

    def _upa_response(self, theta, phi):
        mx = torch.arange(self.cfg.Mx, device=self.device)
        my = torch.arange(self.cfg.My, device=self.device)
        # Broadcasting for batch generation
        if theta.dim() > 0:
            mx = mx.view(1, -1)
            my = my.view(1, -1)
            theta = theta.unsqueeze(-1)
            phi = phi.unsqueeze(-1)

        phase_x = 1j * math.pi * torch.sin(theta) * torch.cos(phi) * mx
        phase_y = 1j * math.pi * torch.sin(theta) * torch.sin(phi) * my

        # Kronecker product approx
        if theta.dim() > 0:
            ux = torch.exp(phase_x).unsqueeze(-1)  # (B, Mx, 1)
            uy = torch.exp(phase_y).unsqueeze(-2)  # (B, 1, My)
            u = (ux * uy).view(theta.shape[0], -1)  # (B, M)
        else:
            u = torch.kron(torch.exp(phase_x), torch.exp(phase_y))

        return u / (torch.norm(u, dim=-1, keepdim=True) + 1e-12)

    def reset(self, batch_size=1):
        self.batch_size = batch_size

        # Angles
        self.theta_los = torch.rand(batch_size, device=self.device) * 0.5 * math.pi
        self.phi_los = torch.rand(batch_size, device=self.device) * 2.0 * math.pi
        self.u_los = self._upa_response(self.theta_los, self.phi_los)

        # NLoS Paths
        self.fD_ut = torch.empty(batch_size, device=self.device).uniform_(
            -self.doppler_max, self.doppler_max
        )

        self.u_nlos_list = []
        self.fD_nlos_list = []
        self.g_list = []
        self.phi0_nlos_list = []

        for _ in range(self.P):
            th = torch.rand(batch_size, device=self.device) * 0.5 * math.pi
            ph = torch.rand(batch_size, device=self.device) * 2.0 * math.pi
            self.u_nlos_list.append(self._upa_response(th, ph))

            fd = torch.empty(batch_size, device=self.device).uniform_(
                -self.doppler_max, self.doppler_max
            )
            self.fD_nlos_list.append(fd)

            self.g_list.append(complex_normal((batch_size,), self.device))
            self.phi0_nlos_list.append(torch.rand(batch_size, device=self.device) * 2 * math.pi)

        self.phi0_los = torch.rand(batch_size, device=self.device) * 2 * math.pi
        self.t_idx = 0

    def step(self):
        self.t_idx += 1

    def get_h_true(self) -> torch.Tensor:
        t = self.t_idx * self.cfg.dt_s

        # LoS
        phase_los = 2 * math.pi * self.fD_ut * t + self.phi0_los
        h_los = self.w_los * torch.exp(1j * phase_los).unsqueeze(-1) * self.u_los

        # NLoS
        h_nlos_sum = torch.zeros_like(h_los)
        for i in range(self.P):
            phase_n = 2 * math.pi * self.fD_nlos_list[i] * t + self.phi0_nlos_list[i]
            val = (
                self.g_list[i].unsqueeze(-1)
                * torch.exp(1j * phase_n).unsqueeze(-1)
                * self.u_nlos_list[i]
            )
            h_nlos_sum += val

        return h_los + self.w_nlos * (h_nlos_sum / math.sqrt(self.P))


# =========================================================
# 3. Models
# =========================================================

# --- 3.1 TDNN-KalmanNet (Proposed) ---
class TDNNKalmanNet(nn.Module):
    def __init__(self, D, H=128, context=5):
        super().__init__()
        self.D = D
        self.H = H
        self.context = context

        # TDNN: Conv1d acts as a sliding window processor
        # Input features: [Innovation, State, Diff] -> 3*D channels
        self.tdnn = nn.Sequential(
            nn.Conv1d(in_channels=3 * D, out_channels=H, kernel_size=context, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, D),
            nn.Sigmoid(),  # Gain K is in [0, 1]
        )

        self.fc_rot = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, 2))
        self.gru_pred = nn.GRUCell(D, H)
        self.fc_out = nn.Linear(H, D)

    def forward(self, x_in, w_out):
        B, q, D = x_in.shape
        dev = x_in.device

        x_post = torch.zeros(B, D, device=dev)
        y_prev = x_in[:, 0, :]

        # Buffer to store history for TDNN
        feat_buf = torch.zeros(B, 3 * D, self.context, device=dev)

        # --- Filtering ---
        for t in range(q):
            y_t = x_in[:, t, :]

            # Physics Predict
            rp = self.fc_rot(x_post)
            rho = torch.sigmoid(rp[:, 0:1]) * 1.5
            phi = torch.tanh(rp[:, 1:2]) * math.pi

            x_complex = devectorize_complex(x_post)
            rot = torch.polar(rho, phi)
            x_pri = vectorize_complex(x_complex * rot)

            # Feature Construction
            innov = y_t - x_pri
            diff = y_t - y_prev
            curr_feat = torch.cat([innov, x_post, diff], dim=1)  # (B, 3D)

            # Update Buffer
            feat_buf = torch.roll(feat_buf, -1, dims=2)
            feat_buf[:, :, -1] = curr_feat

            # Compute Gain
            K = self.tdnn(feat_buf)

            # Update
            x_post = x_pri + K * innov
            y_prev = y_t

        # --- Prediction ---
        preds = []
        h_pred = torch.zeros(B, self.H, device=dev)
        curr = x_post

        for _ in range(w_out):
            h_pred = self.gru_pred(curr, h_pred)
            delta = self.fc_out(h_pred)
            curr = curr + delta
            preds.append(curr.unsqueeze(1))

        return torch.cat(preds, dim=1)


# --- 3.2 KalmanNet (GRU-based) ---
class KalmanNet_GRU(nn.Module):
    def __init__(self, D, H=192):
        super().__init__()
        self.D = D
        self.H = H
        self.fc_feat = nn.Linear(3 * D, H)
        self.gru = nn.GRUCell(H, H)
        self.fc_gain = nn.Linear(H, D)
        self.fc_rot = nn.Linear(H, 2)

        self.gru_pred = nn.GRUCell(D, H)
        self.fc_out = nn.Linear(H, D)

    def forward(self, x_in, w_out):
        B, q, D = x_in.shape
        dev = x_in.device
        x_post = torch.zeros(B, D, device=dev)
        h = torch.zeros(B, self.H, device=dev)
        y_prev = x_in[:, 0, :]

        for t in range(q):
            y_t = x_in[:, t, :]
            innov_proxy = y_t - x_post
            feat = torch.cat([innov_proxy, x_post, y_t - y_prev], dim=1)

            z = torch.tanh(self.fc_feat(feat))
            h = self.gru(z, h)

            rp = self.fc_rot(h)
            rho = torch.sigmoid(rp[:, 0:1]) * 1.5
            phi = torch.tanh(rp[:, 1:2]) * math.pi

            x_c = devectorize_complex(x_post)
            rot = torch.polar(rho, phi)
            x_pri = vectorize_complex(x_c * rot)

            K = torch.sigmoid(self.fc_gain(h))
            x_post = x_pri + K * (y_t - x_pri)
            y_prev = y_t

        preds = []
        h_p = torch.zeros(B, self.H, device=dev)
        curr = x_post
        for _ in range(w_out):
            h_p = self.gru_pred(curr, h_p)
            curr = curr + self.fc_out(h_p)
            preds.append(curr.unsqueeze(1))

        return torch.cat(preds, dim=1)


# --- 3.3 SCP (LSTM) ---
class SCP_Zhang(nn.Module):
    def __init__(self, D, H=256, dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(D, H, num_layers=2, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(H, D)

    def forward(self, x_in, w_out):
        _, (h, c) = self.lstm(x_in)
        curr = x_in[:, -1, :].unsqueeze(1)
        preds = []
        hx = (h, c)
        for _ in range(w_out):
            out, hx = self.lstm(curr, hx)
            y = self.fc(out)
            preds.append(y)
            curr = y
        return torch.cat(preds, dim=1)


# --- 3.4 Traditional Kalman Filter ---
class AnalyticalKF:
    def __init__(self, D, F_val):
        self.D = D
        self.F_val = F_val

    def predict(self, x_in, w_out):
        last_x = x_in[:, -1, :]
        preds = []
        curr = last_x
        for _ in range(w_out):
            curr = curr * self.F_val
            preds.append(curr.unsqueeze(1))
        return torch.cat(preds, dim=1)


# =========================================================
# 4. Data Gen & Training
# =========================================================

def generate_data(cfg: SimCfg, n_samples: int, snr_override=None, doppler_override=None):
    chan = LEOMassiveMIMOChannel(cfg, override_doppler=doppler_override)
    chunk_size = 2000
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    X_list, Y_list = [], []
    snr = snr_override if snr_override is not None else cfg.pilot_snr_db
    total_steps = cfg.q_in + cfg.w_out

    for _ in range(n_chunks):
        curr_bs = min(chunk_size, n_samples - len(X_list) * chunk_size)
        if curr_bs <= 0:
            break

        chan.reset(curr_bs)
        chan.step()
        chan.step()

        seq_h, seq_y = [], []
        for _ in range(total_steps):
            h = chan.get_h_true()
            y = mmse_estimate(h, snr, cfg.device)
            seq_h.append(vectorize_complex(h))
            seq_y.append(vectorize_complex(y))
            chan.step()

        full_h = torch.stack(seq_h, dim=1)
        full_y = torch.stack(seq_y, dim=1)

        x_chunk = full_y[:, : cfg.q_in, :]
        y_chunk = full_h[:, cfg.q_in :, :]

        if cfg.use_rms_norm:
            x_chunk, sc = rms_normalize(x_chunk)
            y_chunk = y_chunk / (sc + 1e-8)

        X_list.append(x_chunk.cpu())
        Y_list.append(y_chunk.cpu())

    return torch.cat(X_list), torch.cat(Y_list)


def train_model(cfg, model, name, train_X, train_Y, val_X, val_Y):
    print(f"   Training {name}...", end="")
    model = model.to(cfg.device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(train_X, train_Y)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    st = time.time()
    for ep in range(cfg.epochs):
        model.train()
        for bx, by in dl:
            bx, by = bx.to(cfg.device), by.to(cfg.device)
            opt.zero_grad()
            pred = model(bx, cfg.w_out)
            loss = loss_fn(pred, by)
            loss.backward()
            opt.step()

    et = time.time()
    print(f" Done ({et-st:.1f}s)")
    return model


# =========================================================
# 5. Evaluation Modules
# =========================================================
class Evaluator:
    def __init__(self, cfg, models):
        self.cfg = cfg
        self.models = models
        # Fixed Bessel function call for torch
        arg = 2 * math.pi * cfg.max_ut_doppler_hz * cfg.dt_s
        self.F_ar1 = float(torch.special.bessel_j0(torch.tensor(arg)).item())
        self.kf_base = AnalyticalKF(cfg.feat_dim, self.F_ar1)

    def run_horizon_eval(self, X, Y):
        res = {k: np.zeros(self.cfg.w_out) for k in self.models.keys()}
        res['KalmanFilter'] = np.zeros(self.cfg.w_out)
        res['Outdated'] = np.zeros(self.cfg.w_out)

        ds = TensorDataset(X, Y)
        dl = DataLoader(ds, batch_size=256)

        cnt = 0
        for bx, by in dl:
            bx, by = bx.to(self.cfg.device), by.to(self.cfg.device)
            B = bx.shape[0]
            cnt += B

            p_out = bx[:, -1, :].unsqueeze(1).repeat(1, self.cfg.w_out, 1)
            p_kf = self.kf_base.predict(bx, self.cfg.w_out)

            for t in range(self.cfg.w_out):
                res['Outdated'][t] += nmse_db(p_out[:, t], by[:, t]) * B
                res['KalmanFilter'][t] += nmse_db(p_kf[:, t], by[:, t]) * B

            for name, m in self.models.items():
                m.eval()
                with torch.no_grad():
                    pm = m(bx, self.cfg.w_out)
                    for t in range(self.cfg.w_out):
                        res[name][t] += nmse_db(pm[:, t], by[:, t]) * B

        for k in res:
            res[k] /= cnt
        return res

    def run_inference_time(self):
        dummy = torch.randn(1, self.cfg.q_in, self.cfg.feat_dim).to(self.cfg.device)
        times = {}
        for m in self.models.values():
            m(dummy, self.cfg.w_out)

        for name, m in self.models.items():
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(100):
                m(dummy, self.cfg.w_out)
            torch.cuda.synchronize()
            times[name] = (time.time() - t0) / 100 * 1000  # ms
        return times

    def run_snr_sweep(self, snr_list):
        res = {k: [] for k in self.models.keys()}
        res['KalmanFilter'] = []
        res['Outdated'] = []
        for snr in snr_list:
            tX, tY = generate_data(self.cfg, 500, snr_override=snr)
            horizon_res = self.run_horizon_eval(tX, tY)
            for k in res:
                res[k].append(np.mean(horizon_res[k]))
        return res

    def run_doppler_sweep(self, doppler_list):
        res = {k: [] for k in self.models.keys()}
        res['KalmanFilter'] = []
        res['Outdated'] = []
        for dop in doppler_list:
            tX, tY = generate_data(self.cfg, 500, doppler_override=dop)
            horizon_res = self.run_horizon_eval(tX, tY)
            for k in res:
                res[k].append(np.mean(horizon_res[k]))
        return res

    def run_altitude_sweep(self, alt_km_list):
        res = {k: [] for k in self.models.keys()}
        res['KalmanFilter'] = []
        res['Outdated'] = []

        chan = LEOMassiveMIMOChannel(self.cfg)
        chan.reset(500)
        seq_len = 200
        h_seq, y_seq = [], []
        for _ in range(seq_len):
            h = chan.get_h_true()
            y = mmse_estimate(h, 15, self.cfg.device)
            h_seq.append(vectorize_complex(h))
            y_seq.append(vectorize_complex(y))
            chan.step()
        H = torch.stack(h_seq, dim=1)
        Y = torch.stack(y_seq, dim=1)

        for alt in alt_km_list:
            delay_step = int(alt / 300 * 2)
            start_idx = 0
            end_idx = start_idx + self.cfg.q_in
            target_start = end_idx + delay_step

            X_val = Y[:, start_idx:end_idx, :]
            Y_val = H[:, target_start : target_start + self.cfg.w_out, :]

            if self.cfg.use_rms_norm:
                X_val, sc = rms_normalize(X_val)
                Y_val = Y_val / (sc + 1e-8)

            # Safety for index range
            if Y_val.shape[1] == self.cfg.w_out:
                horizon_res = self.run_horizon_eval(X_val, Y_val)
                for k in res:
                    res[k].append(np.mean(horizon_res[k]))
            else:
                for k in res:
                    res[k].append(0.0)  # Padding if error
        return res


# =========================================================
# 6. Plotting
# =========================================================

def plot_multi_curves(x, data_dict, xlabel, ylabel, title, fpath):
    plt.figure(figsize=(8, 6))
    markers = ['o', 's', '^', 'D', 'x', '*']
    for i, (name, y_vals) in enumerate(data_dict.items()):
        plt.plot(x, y_vals, marker=markers[i % len(markers)], label=name, linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(fpath)
    plt.close()


def plot_bar(data_dict, ylabel, title, fpath):
    plt.figure(figsize=(8, 5))
    names = list(data_dict.keys())
    vals = list(data_dict.values())
    plt.bar(names, vals, color='skyblue', edgecolor='black')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(fpath)
    plt.close()


# =========================================================
# 7. Main Loop
# =========================================================
if __name__ == "__main__":
    base_dir = setup_environment()
    seed_all(42)

    input_lengths = [5, 10, 30]
    fixed_w_out = 30

    print("🚀 Starting Experiments...")
    print(f"   Inputs: {input_lengths}, Output: {fixed_w_out}")
    print("   Models: TDNN-KNet, KNet-GRU, KF, SCP, Outdated")

    for q in input_lengths:
        print("\n==========================================")
        print(f"▶ Experiment: q_in = {q}")
        print("==========================================")

        exp_dir = os.path.join(base_dir, f"Exp_q{q}_w{fixed_w_out}_{timestamp_str()}")
        ensure_dir(exp_dir)

        cfg = SimCfg(q_in=q, w_out=fixed_w_out)

        # 1. Data Gen
        print(f"1. Generating Data ({cfg.train_samples})...")
        Xtr, Ytr = generate_data(cfg, cfg.train_samples)
        Xte, Yte = generate_data(cfg, cfg.test_samples)

        # 2. Models
        D = cfg.feat_dim
        models = {
            "TDNN-KalmanNet": TDNNKalmanNet(D, H=128, context=min(5, q)).to(cfg.device),
            "KalmanNet(GRU)": KalmanNet_GRU(D, cfg.hidden_dim).to(cfg.device),
            "SCP": SCP_Zhang(D, cfg.scp_hidden).to(cfg.device),
        }

        # 3. Train
        for name, model in models.items():
            models[name] = train_model(cfg, model, name, Xtr, Ytr, Xte, Yte)

        # 4. Evaluation
        evaluator = Evaluator(cfg, models)

        # --- Graph 1: NMSE vs Horizon ---
        print("   -> Eval: Horizon...")
        horizon_res = evaluator.run_horizon_eval(Xte, Yte)
        plot_multi_curves(
            np.arange(1, fixed_w_out + 1),
            horizon_res,
            "Prediction Step",
            "NMSE (dB)",
            f"NMSE vs Horizon (q={q})",
            os.path.join(exp_dir, "1_nmse_horizon.png"),
        )

        # --- Graph 2: Parameters ---
        print("   -> Eval: Params...")
        params = {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()}
        plot_bar(params, "Count", "Number of Parameters", os.path.join(exp_dir, "2_params.png"))

        # --- Graph 3: Inference Time ---
        print("   -> Eval: Time...")
        times = evaluator.run_inference_time()
        plot_bar(times, "Time (ms/batch)", "Inference Time", os.path.join(exp_dir, "3_time.png"))

        # --- Graph 4: Robustness A (Doppler) ---
        print("   -> Eval: Doppler Robustness...")
        dop_list = [100, 200, 400, 800]
        dop_res = evaluator.run_doppler_sweep(dop_list)
        plot_multi_curves(
            dop_list,
            dop_res,
            "Max Doppler (Hz)",
            "Avg NMSE (dB)",
            f"Robustness: Doppler (q={q})",
            os.path.join(exp_dir, "4_robust_doppler.png"),
        )

        # --- Graph 5: Robustness B (SNR) ---
        print("   -> Eval: SNR Robustness...")
        snr_list = [0, 5, 10, 15]
        snr_res = evaluator.run_snr_sweep(snr_list)
        plot_multi_curves(
            snr_list,
            snr_res,
            "SNR (dB)",
            "Avg NMSE (dB)",
            f"Robustness: SNR (q={q})",
            os.path.join(exp_dir, "5_robust_snr.png"),
        )

        # --- Graph 6: Robustness C (Altitude/Delay) ---
        print("   -> Eval: Altitude Robustness...")
        alt_list = [500, 1000, 1500]
        alt_res = evaluator.run_altitude_sweep(alt_list)
        plot_multi_curves(
            alt_list,
            alt_res,
            "Altitude (km)",
            "Avg NMSE (dB)",
            f"Robustness: Altitude (q={q})",
            os.path.join(exp_dir, "6_robust_altitude.png"),
        )

        print(f"✅ Experiment q={q} Finished. Saved to {exp_dir}")

    print("\n🎉 All Done.")
