# -*- coding: utf-8 -*-
"""
LEO Massive MIMO Channel Prediction: Paper Reproduction Suite (Zhang et al. 2021) - FIXED & ROBUST

[Key Fixes]
1. Explicit Broadcasting in generate_batch to prevent dimension mismatch (100 vs 10000).
2. Forced Scalar Types for Rician weights (w_los, w_nlos).
3. Robust Argparse for Notebooks.
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

# -----------------------------
# Utilities
# -----------------------------
C_LIGHT = 299792458.0
R_EARTH_KM = 6371.0

def in_notebook() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None: return False
        return "IPKernelApp" in ip.config
    except: return False

if not in_notebook(): plt.switch_backend("Agg")

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)
def timestamp_str(): return datetime.now().strftime("%Y%m%d_%H%M%S")

class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams:
            try: s.write(data); s.flush()
            except: pass
    def flush(self): pass

def complex_normal(shape, device, std=1.0):
    # CN(0, std^2) -> Real/Imag parts are N(0, std^2/2)
    scale = std / 1.41421356
    re = torch.randn(shape, device=device) * scale
    im = torch.randn(shape, device=device) * scale
    return torch.complex(re, im)

def to_ri(x_c: torch.Tensor) -> torch.Tensor:
    return torch.cat([x_c.real, x_c.imag], dim=-1)

def nmse_db(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    err = (pred - target).pow(2).sum(dim=-1)
    norm = target.pow(2).sum(dim=-1).clamp_min(eps)
    nmse_val = (err / norm).mean()
    return 10.0 * torch.log10(nmse_val).item()

def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

# -----------------------------
# Config
# -----------------------------
@dataclass
class SimCfg:
    # System
    Mx: int = 16
    My: int = 16
    rician_K_db: float = 10.0
    
    # Doppler: Fixed High Doppler for Worst Case (Paper Fig 5 scenario)
    ut_doppler_hz_min: float = 95.0
    ut_doppler_hz_max: float = 100.0
    
    coherence_ms: float = 1.0
    pilot_snr_db: float = 15.0 

    # Sequence
    q_in: int = 4  
    w_out: int = 15 # Horizon

    # Dataset
    train_samples: int = 90000
    val_samples: int = 10000

    # Training
    batch_size: int = 100
    epochs: int = 500
    lr: float = 1e-3
    patience: int = 40 # Early stopping

    # Channel Params
    alt_km: int = 1500  # High Altitude -> Large Delay
    min_elev_deg: float = 30.0
    round_trip: bool = True
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def M(self): return self.Mx * self.My
    @property
    def feat_dim(self): return 2 * self.M

    # SCP Hyperparams
    scp_hidden: int = 256
    scp_dense: int = 512
    scp_layers: int = 2
    scp_dropout: float = 0.25

# -----------------------------
# Channel Model (Robust Version)
# -----------------------------
def slant_range_km(alt_km, elev_deg):
    E = math.radians(elev_deg)
    R = R_EARTH_KM
    h = alt_km
    return math.sqrt((R + h)**2 - (R * math.cos(E))**2) - R * math.sin(E)

def compute_delay_slots(cfg: SimCfg):
    d_km = slant_range_km(cfg.alt_km, cfg.min_elev_deg)
    delay_s = (d_km * 1000.0) / C_LIGHT
    if cfg.round_trip: delay_s *= 2.0
    A = int(math.ceil(delay_s * 1000 / cfg.coherence_ms))
    return A, delay_s * 1000

class LEOMassiveMIMOChannel:
    def __init__(self, cfg: SimCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        K_lin = 10 ** (cfg.rician_K_db / 10.0)
        # Ensure these are scalars
        self.w_los = float(math.sqrt(K_lin / (K_lin + 1.0)))
        self.w_nlos = float(math.sqrt(1.0 / (K_lin + 1.0)))
        
        self.mx = torch.arange(cfg.Mx, device=self.device).float()
        self.my = torch.arange(cfg.My, device=self.device).float()
        
    def _upa(self, theta, phi):
        # theta, phi: (B, 1)
        # phase: (B, M)
        phase_x = 1j * math.pi * torch.sin(theta) * torch.cos(phi) * self.mx.unsqueeze(0) # (B, Mx)
        phase_y = 1j * math.pi * torch.sin(theta) * torch.sin(phi) * self.my.unsqueeze(0) # (B, My)
        
        # Kronecker product via reshape
        # ux: (B, Mx, 1), uy: (B, 1, My) -> (B, Mx, My) -> (B, M)
        u = (torch.exp(phase_x).unsqueeze(2) * torch.exp(phase_y).unsqueeze(1)).reshape(-1, self.cfg.M)
        return u / math.sqrt(self.cfg.M)

    def generate_batch(self, B, total_len):
        dt = self.cfg.coherence_ms * 1e-3
        t = torch.arange(total_len, device=self.device).float().view(1, -1) # (1, T)
        
        # Angles
        theta = torch.rand(B, 1, device=self.device) * 0.5 * math.pi
        phi = torch.rand(B, 1, device=self.device) * 2.0 * math.pi
        
        # LoS Doppler
        fd_ut = (torch.rand(B, 1, device=self.device) * (self.cfg.ut_doppler_hz_max - self.cfg.ut_doppler_hz_min) + 
                 self.cfg.ut_doppler_hz_min)
        
        # Phase: (B, 1) * (1, T) -> (B, T)
        phase_los = 2 * math.pi * fd_ut * t * dt 
        
        # Steering Vector: (B, M)
        u_los = self._upa(theta, phi) 
        
        # h_los construction
        # exp(j*phase): (B, T) -> (B, T, 1)
        # u_los: (B, M) -> (B, 1, M)
        # Result: (B, T, M)
        h_los = self.w_los * torch.exp(1j * phase_los).unsqueeze(-1) * u_los.unsqueeze(1)
        
        # NLoS
        h_nlos = self.w_nlos * complex_normal((B, total_len, self.cfg.M), self.device)
        
        h_true = h_los + h_nlos
        
        # Pilot Noise
        noise_std = math.sqrt(1.0 / (10 ** (self.cfg.pilot_snr_db / 10.0)))
        n = complex_normal(h_true.shape, self.device, noise_std)
        y_est = h_true + n
        
        return to_ri(y_est), to_ri(h_true)

# -----------------------------
# Models
# -----------------------------

# SCP (Encoder-Decoder LSTM)
class SCP_Robust(nn.Module):
    def __init__(self, D, H=256, dense_H=512, layers=2, dropout=0.25):
        super().__init__()
        self.encoder = nn.LSTM(D, H, num_layers=layers, batch_first=True, dropout=dropout)
        self.decoder_cell = nn.LSTMCell(D, H)
        self.fc = nn.Sequential(
            nn.Linear(H, dense_H), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dense_H, D)
        )
        self.H = H
        
    def forward(self, x, w_out):
        B = x.shape[0]
        _, (h, c) = self.encoder(x)
        h_dec, c_dec = h[-1], c[-1] 
        
        curr_in = x[:, -1, :]
        preds = []
        for _ in range(w_out):
            h_dec, c_dec = self.decoder_cell(curr_in, (h_dec, c_dec))
            out = self.fc(h_dec)
            preds.append(out)
            curr_in = out
        return torch.stack(preds, dim=1)

# TDNN-KalmanNet
class TDNNKalmanNet(nn.Module):
    def __init__(self, D, H=256, w_out=15):
        super().__init__()
        self.w_out = w_out
        self.enc = nn.Sequential(
            nn.Conv1d(2*D, 128, 3, dilation=1, padding=1), nn.ReLU(),
            nn.Conv1d(128, 128, 3, dilation=2, padding=2), nn.ReLU(),
            nn.Conv1d(128, H, 3, dilation=4, padding=4), nn.ReLU(),
        )
        self.fc_gru = nn.GRUCell(H+D, H)
        self.fc_out = nn.Linear(H, D)
        self.fc_rp = nn.Linear(H, 2)
        
    def forward(self, x_in):
        B, q, D = x_in.shape
        diff = x_in - torch.roll(x_in, 1, dims=1); diff[:, 0, :] = 0
        feat = torch.cat([x_in, diff], dim=2)
        h = self.enc(feat.transpose(1, 2)).mean(dim=2)
        
        preds = []
        curr = x_in[:, -1, :]
        h_state = h
        
        for _ in range(self.w_out):
            rp = self.fc_rp(h_state)
            rho = 1.0 + 0.3 * torch.tanh(rp[:, 0:1])
            phi = 0.6 * torch.tanh(rp[:, 1:2])
            
            D_half = D // 2
            re, im = curr[:, :D_half], curr[:, D_half:]
            c, s = torch.cos(phi), torch.sin(phi)
            curr_rot = torch.cat([rho*(c*re - s*im), rho*(s*re + c*im)], dim=1)
            
            inp = torch.cat([curr_rot, h], dim=1)
            h_state = self.fc_gru(inp, h_state)
            res = self.fc_out(h_state)
            curr = curr_rot + res
            preds.append(curr)
            
        return torch.stack(preds, dim=1)

# -----------------------------
# Train Loop
# -----------------------------
def train_experiment(cfg: SimCfg, out_dir: str):
    sim = LEOMassiveMIMOChannel(cfg)
    A, delay_ms = compute_delay_slots(cfg)
    print(f"[*] Scenario: Alt={cfg.alt_km}km, A={A} slots (Delay={delay_ms:.2f}ms)")
    
    D = cfg.feat_dim
    models = {
        "SCP": SCP_Robust(D, H=cfg.scp_hidden, dense_H=cfg.scp_dense, layers=cfg.scp_layers, dropout=cfg.scp_dropout).to(cfg.device),
        "TDNN-KalmanNet": TDNNKalmanNet(D, w_out=cfg.w_out).to(cfg.device)
    }
    
    optimizers = {k: optim.Adam(v.parameters(), lr=cfg.lr) for k, v in models.items()}
    loss_fn = nn.MSELoss()
    
    best_loss = {k: float('inf') for k in models}
    patience_counter = {k: 0 for k in models}
    
    steps_per_epoch = cfg.train_samples // cfg.batch_size
    
    for ep in range(cfg.epochs):
        for model in models.values(): model.train()
        
        for _ in range(steps_per_epoch):
            y_est, h_true = sim.generate_batch(cfg.batch_size, cfg.q_in + A + cfg.w_out)
            x_in = y_est[:, :cfg.q_in].float()
            y_tg = h_true[:, cfg.q_in+A:].float()
            
            for name, model in models.items():
                if patience_counter[name] >= cfg.patience: continue
                
                optimizers[name].zero_grad()
                if name == "TDNN-KalmanNet": pred = model(x_in)
                else: pred = model(x_in, cfg.w_out)
                
                loss = loss_fn(pred, y_tg)
                loss.backward()
                optimizers[name].step()

        # Validation
        if ep % 5 == 0:
            val_loss = {k: 0.0 for k in models}
            with torch.no_grad():
                y_est, h_true = sim.generate_batch(cfg.val_samples, cfg.q_in + A + cfg.w_out)
                x_in, y_tg = y_est[:, :cfg.q_in].float(), h_true[:, cfg.q_in+A:].float()
                
                for name, model in models.items():
                    model.eval()
                    if name == "TDNN-KalmanNet": pred = model(x_in)
                    else: pred = model(x_in, cfg.w_out)
                    val_loss[name] = loss_fn(pred, y_tg).item()
                    
                    if val_loss[name] < best_loss[name]:
                        best_loss[name] = val_loss[name]
                        patience_counter[name] = 0
                        torch.save(model.state_dict(), os.path.join(out_dir, f"best_{name}.pt"))
                    else:
                        patience_counter[name] += 1
            
            print(f"Ep {ep}: Val SCP={val_loss['SCP']:.5f} (Bad={patience_counter['SCP']}), "
                  f"TDNN={val_loss['TDNN-KalmanNet']:.5f} (Bad={patience_counter['TDNN-KalmanNet']})")
            
            if all(cnt >= cfg.patience for cnt in patience_counter.values()):
                print("All models early stopped.")
                break

    return models

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(cfg, models, out_dir):
    sim = LEOMassiveMIMOChannel(cfg)
    A, _ = compute_delay_slots(cfg)
    
    nmse_res = {name: np.zeros(cfg.w_out) for name in ["SCP", "TDNN-KalmanNet", "Outdated"]}
    
    steps = 100
    for _ in range(steps):
        y_est, h_true = sim.generate_batch(100, cfg.q_in + A + cfg.w_out)
        x_in = y_est[:, :cfg.q_in].float().to(cfg.device)
        y_tg = h_true[:, cfg.q_in+A:].float().to(cfg.device)
        
        last_obs = x_in[:, -1, :].unsqueeze(1).repeat(1, cfg.w_out, 1)
        
        models["SCP"].eval()
        scp_pred = models["SCP"](x_in, cfg.w_out)
        
        models["TDNN-KalmanNet"].eval()
        tdnn_pred = models["TDNN-KalmanNet"](x_in)
        
        for t in range(cfg.w_out):
            nmse_res["Outdated"][t] += nmse_db(last_obs[:, t], y_tg[:, t])
            nmse_res["SCP"][t] += nmse_db(scp_pred[:, t], y_tg[:, t])
            nmse_res["TDNN-KalmanNet"][t] += nmse_db(tdnn_pred[:, t], y_tg[:, t])
            
    for k in nmse_res: nmse_res[k] /= steps
    
    # Plot
    w_axis = np.arange(1, cfg.w_out + 1) + A 
    plt.figure(figsize=(8, 6))
    plt.plot(w_axis, nmse_res["Outdated"], 'r-o', label='Outdated')
    plt.plot(w_axis, nmse_res["SCP"], 'k-^', label='SCP')
    plt.plot(w_axis, nmse_res["TDNN-KalmanNet"], 'b-s', label='TDNN-KalmanNet')
    
    plt.xlabel('Prediction Step (t)')
    plt.ylabel('NMSE (dB)')
    plt.title(f'NMSE @ {cfg.alt_km}km')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "Result.png"))
    print("✅ Result saved.")
    
    # Save CSV
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + list(nmse_res.keys()))
        for i in range(cfg.w_out):
            w.writerow([w_axis[i]] + [nmse_res[k][i] for k in nmse_res])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="kaggle/working/LEO_Results_1500km")
    
    # Fixed for Notebooks
    if in_notebook(): args = parser.parse_args(args=[])
    else: args = parser.parse_args()
    
    ensure_dir(args.out_dir)
    sys.stdout = Tee(sys.stdout, open(f"{args.out_dir}/log.txt", "w"))
    
    cfg = SimCfg(alt_km=1500, w_out=15) 
    
    print(f"🚀 Starting Experiment @ 1500km")
    models = train_experiment(cfg, args.out_dir)
    
    # Load Best
    for name in models:
        path = f"{args.out_dir}/best_{name}.pt"
        if os.path.exists(path):
            models[name].load_state_dict(torch.load(path))
            
    evaluate(cfg, models, args.out_dir)
    shutil.make_archive("LEO_Results", 'zip', args.out_dir)
    print("📦 Results zipped.")

if __name__ == "__main__":
    main()
