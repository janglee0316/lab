# -*- coding: utf-8 -*-
"""
LEO Massive MIMO Channel Prediction: All-in-One Suite (Lightning AI / Jupyter-friendly)
- Dataset Gen + Train + Eval
- Compare: TDNN-KalmanNet, KalmanNet(GRU), Kalman Filter(KalmanRot), SCP, Outdated(Paper-aligned)
- Outdated (Paper-aligned): use perfect true CSI at t=q_in+1 and keep it fixed for all future slots

Lightning AI adaptations:
- Removed Kaggle-specific paths (/kaggle/working)
- Added workspace root auto-detection for Lightning Studio
- base_dir default: <workspace_root>/LEO_Project
- zip saved under: <base_dir>/results/<exp_tag>.zip
"""

import os, math, time, json, csv, argparse, random, shutil
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from typing import Dict, Tuple, Optional

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

# 스크립트/서버에서는 Agg. 노트북이면 inline/기본 백엔드 유지.
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
    # Lightning Studio에서는 File Browser로도 다운로드 가능 + FileLink도 대부분 동작
    if not in_notebook():
        print("Zip saved at:", path)
        return
    try:
        from IPython.display import display, FileLink
        print("\n⬇️ Download link:")
        display(FileLink(path))
        print("↑ 링크 클릭해서 다운로드 (또는 Lightning 파일 브라우저에서 다운로드)")
    except Exception as e:
        print("show_download_link failed:", e)
        print("Zip saved at:", path)


def make_zip_of_folder(folder: str, zip_out: str):
    base_name = zip_out.replace(".zip", "")
    zip_path = shutil.make_archive(base_name, "zip", folder)
    return zip_path


def file_exists_and_size(path: str) -> str:
    if not os.path.exists(path):
        return "MISSING"
    sz = os.path.getsize(path)
    return f"{sz/1024:.1f} KB"


def list_dir_tree(root: str, max_files: int = 200):
    print(f"\n📂 Listing: {root}")
    if not os.path.exists(root):
        print("  (root not found)")
        return
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            files.append(os.path.join(dirpath, fn))
    files = sorted(files)
    for p in files[:max_files]:
        print("  -", p, "|", file_exists_and_size(p))
    if len(files) > max_files:
        print(f"  ... ({len(files)-max_files} more)")


def detect_workspace_root() -> str:
    """
    Lightning AI Studio에서 흔히 존재하는 경로를 우선순위로 탐지.
    - /teamspace/studios/this_studio  (Lightning Studio 대표 경로)
    - /teamspace
    - /workspace
    - 현재 작업 디렉토리
    """
    env_candidates = [
        os.environ.get("LIGHTNING_WORKSPACE_DIR"),
        os.environ.get("LIGHTNING_ARTIFACTS_DIR"),
        os.environ.get("LIGHTNING_CLOUD_WORKSPACE"),
    ]
    candidates = [c for c in env_candidates if c] + [
        "/teamspace/studios/this_studio",
        "/teamspace",
        "/workspace",
        os.getcwd(),
    ]
    for c in candidates:
        try:
            if c and os.path.exists(c):
                # 최소 확인: 하위에 test 폴더 생성 가능 여부
                test_dir = os.path.join(c, ".write_test_tmp")
                os.makedirs(test_dir, exist_ok=True)
                os.rmdir(test_dir)
                return c
        except Exception:
            continue
    return os.getcwd()


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


def mmse_estimate_from_pilot(h_true: torch.Tensor, snr_db: float, device: torch.device):
    snr_lin = 10 ** (snr_db / 10.0)
    sig_pow = torch.mean(torch.abs(h_true) ** 2).real
    noise_var = sig_pow / snr_lin
    noise = complex_normal(h_true.shape, device=device, std=torch.sqrt(noise_var).item())
    y = h_true + noise
    beta = snr_lin / (1.0 + snr_lin)
    return beta * y


def nmse_db(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    hp = devectorize_complex(pred.float())
    ht = devectorize_complex(target.float())
    denom = (torch.norm(ht) ** 2 + eps)
    err = (torch.norm(hp - ht) ** 2)
    return float(10.0 * torch.log10(err / denom + eps).item())


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


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
    max_ut_doppler_hz: float = 100.0
    coherence_ms: float = 1.0
    pilot_snr_db: float = 15.0

    # Sequence
    q_in: int = 4
    w_out: int = 30

    # Dataset size
    train_samples: int = 70000
    val_samples: int = 8000

    # Normalize + store
    use_rms_norm: bool = True
    dataset_store_fp16: bool = True

    # Channel realism
    small_angular_spread: bool = True
    as_std_deg: float = 2.0
    doppler_path_std_hz: float = 1.0
    doppler_rw_std_hz: float = 0.0

    # Training
    batch_size: int = 80
    epochs: int = 250
    lr: float = 1e-3
    weight_decay: float = 0.0
    early_stop_patience: int = 25

    # SCP
    scp_hidden: int = 256
    scp_layers: int = 2
    scp_dense: int = 512
    dropout: float = 0.25
    teacher_forcing: float = 0.5

    # KalmanNet(GRU)
    knet_hidden: int = 256

    # TDNN-KalmanNet (tuned for q_in=4)
    # k=3, dilations (1,2) => RF = 1 + 2*(1+2) = 7
    tdnn_hidden: int = 256
    tdnn_layers: int = 2
    tdnn_kernel: int = 3
    tdnn_drop: float = 0.10

    # Eval
    horizon_trials: int = 2000

    # Altitudes for delay compare
    altitudes_km: Tuple[int, ...] = (500, 1000, 1500)
    delay_mode: str = "oneway"  # "oneway" or "roundtrip"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def M(self): return self.Mx * self.My
    @property
    def feat_dim(self): return 2 * self.M
    @property
    def dt_s(self): return self.coherence_ms * 1e-3

    def delay_steps_from_alt_km(self, alt_km: int) -> int:
        delay_ms = alt_km / 300.0
        if self.delay_mode == "roundtrip":
            delay_ms *= 2.0
        return max(1, int(round(delay_ms / self.coherence_ms)))


# -------------------------
# 2) Channel Model
# -------------------------
class LEOMassiveMIMOChannel:
    def __init__(self, cfg: SimCfg, device=None):
        self.cfg = cfg
        self.device = torch.device(device if device else cfg.device)

        K_lin = 10 ** (cfg.rician_K_db / 10.0)
        self.w_los = math.sqrt(K_lin / (K_lin + 1.0))
        self.w_nlos = math.sqrt(1.0 / (K_lin + 1.0))

        self.P = 6
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
        self.theta_los = torch.rand(1, device=self.device) * 0.5 * math.pi
        self.phi_los = torch.rand(1, device=self.device) * 2.0 * math.pi
        self.u_los = self._upa_response(self.theta_los, self.phi_los)

        self.fD_ut = random.uniform(-self.cfg.max_ut_doppler_hz, self.cfg.max_ut_doppler_hz)
        self.phi0_los = random.uniform(0, 2 * math.pi)

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

        self.u_p = torch.stack([self._upa_response(self.theta_p[i], self.phi_p[i]) for i in range(self.P)])
        self.g = complex_normal((self.P,), device=self.device)

        self.fD_p = torch.tensor(self.fD_ut, device=self.device).repeat(self.P)
        self.fD_p += torch.randn_like(self.fD_p) * self.cfg.doppler_path_std_hz

        self.phi0_p = torch.empty(self.P, device=self.device).uniform_(0, 2 * math.pi)
        self.f0 = 2.0e9
        self.tau_p = torch.empty(self.P, device=self.device).uniform_(0, 200e-9)
        self.t_idx = 0

    def step(self):
        self.t_idx += 1
        if self.cfg.doppler_rw_std_hz > 0.0:
            self.fD_p = torch.clamp(
                self.fD_p + torch.randn_like(self.fD_p) * self.cfg.doppler_rw_std_hz,
                -self.cfg.max_ut_doppler_hz,
                self.cfg.max_ut_doppler_hz
            )

    def get_h_true(self):
        t = self.t_idx * self.cfg.dt_s
        los = self.w_los * torch.exp(
            1j * torch.tensor(2 * math.pi * self.fD_ut * t + self.phi0_los, device=self.device)
        ) * self.u_los

        nlos_phase = 2 * math.pi * self.fD_p * t + self.phi0_p - 2 * math.pi * self.f0 * self.tau_p
        coeff = self.g * torch.exp(1j * nlos_phase)
        nlos = self.w_nlos * (coeff[:, None] * self.u_p).sum(dim=0) / math.sqrt(self.P)
        return los + nlos


# -------------------------
# 3) Dataset Generation
# -------------------------
def dataset_path(base_dir: str, cfg: SimCfg):
    tag = f"q{cfg.q_in}_L{cfg.w_out}_dop{int(cfg.max_ut_doppler_hz)}_coh{safe_float_tag(cfg.coherence_ms)}_snr{int(cfg.pilot_snr_db)}"
    return os.path.join(base_dir, "datasets", f"dataset_{tag}.pt")


def generate_dataset_tensors(cfg: SimCfg, n_samples: int):
    print(f"⚡ Generating {n_samples} samples | q={cfg.q_in}, w={cfg.w_out}, dop={cfg.max_ut_doppler_hz}, coh={cfg.coherence_ms}, snr={cfg.pilot_snr_db}")
    dev_gen = torch.device("cpu")
    cfg_gen = replace(cfg, device="cpu")
    chan = LEOMassiveMIMOChannel(cfg_gen, device="cpu")

    D = cfg.feat_dim
    total = cfg.q_in + cfg.w_out
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
            y = mmse_estimate_from_pilot(h, cfg.pilot_snr_db, dev_gen)
            seq_true.append(vectorize_complex(h).to("cpu"))
            seq_obs.append(vectorize_complex(y).to("cpu"))
            chan.step()

        seq_obs = torch.stack(seq_obs)
        seq_true = torch.stack(seq_true)

        x_raw = seq_obs[:cfg.q_in]
        y_raw = seq_true[cfg.q_in:]   # future TRUE

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


def get_or_make_dataset(base_dir: str, cfg: SimCfg, force: bool = False):
    ensure_dir(os.path.join(base_dir, "datasets"))
    path = dataset_path(base_dir, cfg)
    if (not force) and os.path.exists(path):
        pack = torch.load(path, map_location="cpu")
        print(f"✅ Using cached dataset: {path}")
        return path, pack

    Xtr, Ytr = generate_dataset_tensors(cfg, cfg.train_samples)
    Xva, Yva = generate_dataset_tensors(cfg, cfg.val_samples)
    pack = {"cfg": asdict(cfg), "train": {"X": Xtr, "Y": Ytr}, "val": {"X": Xva, "Y": Yva}}
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
# 5) Baselines (KalmanRot)
# -------------------------
def estimate_rho_phi(y_prev: torch.Tensor, y_curr: torch.Tensor, eps: float = 1e-12):
    hp = devectorize_complex(y_prev.float())
    hc = devectorize_complex(y_curr.float())
    a = torch.sum(torch.conj(hp) * hc) / (torch.sum(torch.conj(hp) * hp) + eps)
    rho = torch.clamp(torch.abs(a), 0.0, 1.5)
    phi = torch.angle(a)
    return rho, phi


def rotate_realimag(x: torch.Tensor, rho: torch.Tensor, phi: torch.Tensor):
    D = x.shape[-1]
    M = D // 2
    re, im = x[:, :M], x[:, M:]
    c, s = torch.cos(phi), torch.sin(phi)
    re2 = rho * (c * re - s * im)
    im2 = rho * (s * re + c * im)
    return torch.cat([re2, im2], dim=1)


class KalmanRotPredictor:
    def __init__(self, D: int, R_var: float = 0.05):
        self.D = D
        self.R = R_var
        self.reset()

    def reset(self):
        self.x = None
        self.P = 1.0
        self.rho = None
        self.phi = None
        self.prev_y = None

    def update(self, y: torch.Tensor):
        if self.prev_y is None:
            self.prev_y = y.clone()
        else:
            rho, phi = estimate_rho_phi(self.prev_y, y)
            self.rho, self.phi = rho, phi
            self.prev_y = y.clone()

        if self.x is None:
            self.x = y.clone()
            return

        rho = float(self.rho) if self.rho is not None else 1.0
        phi = float(self.phi) if self.phi is not None else 0.0

        x_pri = rotate_realimag(
            self.x.unsqueeze(0),
            torch.tensor([[rho]], device=y.device),
            torch.tensor([[phi]], device=y.device),
        )[0]

        F2 = rho * rho
        Q = max(1e-6, 1.0 - min(F2, 0.9999))
        P_pri = F2 * self.P + Q

        K = P_pri / (P_pri + self.R)
        self.x = x_pri + K * (y - x_pri)
        self.P = (1.0 - K) * P_pri

    def predict_ahead(self, steps: int) -> torch.Tensor:
        if self.x is None:
            return torch.zeros(self.D)
        rho = float(self.rho) if self.rho is not None else 1.0
        phi = float(self.phi) if self.phi is not None else 0.0

        x = self.x.unsqueeze(0)
        rho_t = torch.tensor([[rho]], device=x.device)
        phi_t = torch.tensor([[phi]], device=x.device)
        for _ in range(steps):
            x = rotate_realimag(x, rho_t, phi_t)
        return x[0]


# -------------------------
# 6) Models
# -------------------------
class SCP_Zhang_TF(nn.Module):
    def __init__(self, D: int, H: int = 256, layers: int = 2, dense: int = 512, dropout: float = 0.25):
        super().__init__()
        self.D, self.H, self.layers = D, H, layers
        self.encoder = nn.LSTM(D, H, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        cells = [nn.LSTMCell(D, H)] + [nn.LSTMCell(H, H) for _ in range(layers - 1)]
        self.dec_cells = nn.ModuleList(cells)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(H, dense)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(dense, D)

    def forward(self, x_in: torch.Tensor, w_out: int, y_teacher: Optional[torch.Tensor] = None, tf_ratio: float = 0.0):
        _, (h_n, c_n) = self.encoder(x_in)
        h = [h_n[i] for i in range(self.layers)]
        c = [c_n[i] for i in range(self.layers)]
        dec_in = x_in[:, -1, :]
        outs = []
        for t in range(w_out):
            inp = dec_in
            for li, cell in enumerate(self.dec_cells):
                h[li], c[li] = cell(inp, (h[li], c[li]))
                inp = self.drop(h[li])
            z = self.fc2(self.drop(self.act(self.fc1(inp))))
            outs.append(z.unsqueeze(1))
            if (y_teacher is not None) and (tf_ratio > 0.0) and (random.random() < tf_ratio):
                dec_in = y_teacher[:, t, :]
            else:
                dec_in = z
        return torch.cat(outs, dim=1)


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
        x_post = torch.zeros(B, D, device=dev)
        h = torch.zeros(B, self.H, device=dev)
        y_prev = x_in[:, 0, :]
        prev_upd = torch.zeros(B, D, device=dev)

        for t in range(q):
            y_t = x_in[:, t, :]
            feat = torch.cat([y_t - y_prev, y_t - x_post, prev_upd], dim=1)
            z = torch.tanh(self.fc_feat(feat))
            h = self.ln(self.gru(z, h))

            rp = self.fc_rp(h)
            rho = torch.sigmoid(rp[:, 0:1]) * 1.2
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
            rho = torch.sigmoid(rp[:, 0:1]) * 1.2
            phi = math.pi * torch.tanh(rp[:, 1:2])
            curr = rotate_realimag(curr, rho, phi)
            preds.append(curr.unsqueeze(1))
        return torch.cat(preds, dim=1)


class TDNNEncoder(nn.Module):
    def __init__(self, in_ch: int, H: int, layers: int = 2, k: int = 3, drop: float = 0.1):
        super().__init__()
        blocks = []
        ch = in_ch
        for i in range(layers):
            dilation = 2 ** i  # 1,2
            pad = (k - 1) * dilation // 2
            blocks += [
                nn.Conv1d(ch, H, kernel_size=k, dilation=dilation, padding=pad, padding_mode="replicate"),
                nn.GELU(),
                nn.Dropout(drop),
            ]
            ch = H
        self.net = nn.Sequential(*blocks)
        self.ln = nn.LayerNorm(H)

    def forward(self, x):
        x = x.transpose(1, 2)      # (B,T,C)->(B,C,T)
        h = self.net(x)
        h = h.transpose(1, 2)      # (B,C,T)->(B,T,C)
        return self.ln(h)


class TDNNKalmanNet(nn.Module):
    def __init__(self, D: int, H: int = 256, tdnn_layers: int = 2, k: int = 3, drop: float = 0.1):
        super().__init__()
        self.D, self.H = D, H
        self.enc = TDNNEncoder(in_ch=2 * D, H=H, layers=tdnn_layers, k=k, drop=drop)
        self.fc_gain = nn.Linear(H, D)
        self.fc_rp = nn.Linear(H, 2)
        self.fc_roll = nn.Sequential(
            nn.Linear(H + D, H),
            nn.GELU(),
            nn.LayerNorm(H),
            nn.Dropout(drop)
        )
        self.fc_rp_r = nn.Linear(H, 2)

    def forward(self, x_in: torch.Tensor, w_out: int):
        B, q, D = x_in.shape
        dev = x_in.device
        y_prev = torch.zeros(B, D, device=dev)

        feats = []
        for t in range(q):
            y_t = x_in[:, t, :]
            dy = y_t - (y_prev if t > 0 else y_t)
            feats.append(torch.cat([y_t, dy], dim=1).unsqueeze(1))
            y_prev = y_t
        feats = torch.cat(feats, dim=1)   # (B,q,2D)

        h_seq = self.enc(feats)           # (B,q,H)

        x_post = torch.zeros(B, D, device=dev)
        for t in range(q):
            y_t = x_in[:, t, :]
            h_t = h_seq[:, t, :]
            rp = self.fc_rp(h_t)
            rho = torch.sigmoid(rp[:, 0:1]) * 1.2
            phi = math.pi * torch.tanh(rp[:, 1:2])
            x_pri = rotate_realimag(x_post, rho, phi)
            K = torch.sigmoid(self.fc_gain(h_t))
            x_post = x_pri + K * (y_t - x_pri)

        preds = []
        h_r, curr = h_seq[:, -1, :], x_post
        for _ in range(w_out):
            h_r = self.fc_roll(torch.cat([h_r, curr], dim=1))
            rp = self.fc_rp_r(h_r)
            rho = torch.sigmoid(rp[:, 0:1]) * 1.2
            phi = math.pi * torch.tanh(rp[:, 1:2])
            curr = rotate_realimag(curr, rho, phi)
            preds.append(curr.unsqueeze(1))
        return torch.cat(preds, dim=1)


# -------------------------
# 7) Training
# -------------------------
def train_model(cfg: SimCfg, model: nn.Module, name: str, outdir: str, train_ds, val_ds, num_workers: int = 0):
    print(f"\n🧠 Training {name} | q={cfg.q_in} L={cfg.w_out} | ep={cfg.epochs} bs={cfg.batch_size}")
    model = model.to(cfg.device)

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    pin = cfg.device.startswith("cuda")
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=pin, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin, num_workers=num_workers)

    use_amp = cfg.device.startswith("cuda")
    if name in ["TDNN_KalmanNet", "TDNN-KalmanNet"]:
        use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val, bad = 1e18, 0
    best_path = os.path.join(outdir, f"best_{name}.pt")

    for ep in range(cfg.epochs):
        model.train()
        tr, nb = 0.0, 0
        for x, y in train_dl:
            x = torch.nan_to_num(x.to(cfg.device, non_blocking=True).float(), nan=0.0, posinf=0.0, neginf=0.0)
            y = torch.nan_to_num(y.to(cfg.device, non_blocking=True).float(), nan=0.0, posinf=0.0, neginf=0.0)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                if isinstance(model, SCP_Zhang_TF):
                    pred = model(x, cfg.w_out, y_teacher=y, tf_ratio=cfg.teacher_forcing)
                else:
                    pred = model(x, cfg.w_out)
                loss = loss_fn(pred, y)

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            tr += float(loss.item()); nb += 1
        tr = tr / max(1, nb)

        model.eval()
        va, nv = 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x = torch.nan_to_num(x.to(cfg.device, non_blocking=True).float(), nan=0.0, posinf=0.0, neginf=0.0)
                y = torch.nan_to_num(y.to(cfg.device, non_blocking=True).float(), nan=0.0, posinf=0.0, neginf=0.0)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    if isinstance(model, SCP_Zhang_TF):
                        pred = model(x, cfg.w_out, y_teacher=None, tf_ratio=0.0)
                    else:
                        pred = model(x, cfg.w_out)
                    loss = loss_fn(pred, y)

                if not torch.isfinite(loss):
                    continue
                va += float(loss.item()); nv += 1
        va = va / max(1, nv)

        if va < best_val:
            best_val, bad = va, 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1

        if ep % 10 == 0 or ep == cfg.epochs - 1:
            print(f"[{name}] ep={ep:03d} train={tr:.6f} val={va:.6f} best={best_val:.6f}")

        if bad >= cfg.early_stop_patience:
            print(f"🟡 Early stop triggered for {name} at ep={ep}")
            break

    model.load_state_dict(torch.load(best_path, map_location=cfg.device))
    print(f"✅ best checkpoint: {best_path} | size={file_exists_and_size(best_path)}")
    return model


# -------------------------
# 8) Evaluation
# -------------------------
@torch.no_grad()
def eval_nmse_horizon(cfg: SimCfg, models: Dict[str, Optional[nn.Module]], val_ds: SliceDataset, trials: int = 2000):
    nmse = {k: np.zeros(cfg.w_out) for k in models.keys()}
    kf = KalmanRotPredictor(D=cfg.feat_dim, R_var=0.05)

    N = len(val_ds)
    idxs = np.random.choice(N, size=min(trials, N), replace=False)

    for idx in idxs:
        x_in, y_true = val_ds[idx]
        x_in = x_in.float().to(cfg.device)
        y_true = y_true.float().to(cfg.device)

        # KalmanFilter baseline from OBS
        kf.reset()
        for t in range(cfg.q_in):
            kf.update(x_in[t].detach().cpu())
        kf_preds = torch.stack([kf.predict_ahead(i+1) for i in range(cfg.w_out)], dim=0).to(cfg.device).float()

        # Outdated (Paper-aligned): perfect TRUE CSI at t=q_in+1, never updates
        last_true = y_true[0]

        preds = {}
        for name, m in models.items():
            if name == "Outdated":
                preds[name] = last_true.unsqueeze(0).repeat(cfg.w_out, 1)
            elif name == "KalmanFilter":
                preds[name] = kf_preds
            else:
                preds[name] = m(x_in.unsqueeze(0), cfg.w_out)[0]

        for t in range(cfg.w_out):
            tgt = y_true[t]
            for name in models.keys():
                nmse[name][t] += nmse_db(preds[name][t], tgt)

    for k in nmse:
        nmse[k] /= len(idxs)
    return nmse


def benchmark_inference(cfg: SimCfg, model: Optional[nn.Module], algo: str, x_example: torch.Tensor, iters: int = 200):
    # inference-time benchmark doesn't depend on Outdated using y_true (values irrelevant)
    if algo in ["Outdated", "KalmanFilter"]:
        t0 = time.perf_counter()
        kf = KalmanRotPredictor(D=cfg.feat_dim, R_var=0.05)
        for _ in range(iters):
            if algo == "Outdated":
                _ = x_example[-1].unsqueeze(0).repeat(cfg.w_out, 1)
            else:
                kf.reset()
                for t in range(cfg.q_in):
                    kf.update(x_example[t].detach().cpu())
                _ = torch.stack([kf.predict_ahead(i+1) for i in range(cfg.w_out)], dim=0)
        dt = (time.perf_counter() - t0) * 1000.0
        return dt / iters

    model.eval()
    x = x_example.unsqueeze(0).to(cfg.device).float()

    for _ in range(30):
        _ = model(x, cfg.w_out)
    if cfg.device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x, cfg.w_out)
    if cfg.device.startswith("cuda"):
        torch.cuda.synchronize()

    dt = (time.perf_counter() - t0) * 1000.0
    return dt / iters


# -------------------------
# 9) Plotting
# -------------------------
def plot_nmse_curve(x, curves: Dict[str, np.ndarray], title: str, save_path: str):
    plt.figure(figsize=(10, 6))
    for k, v in curves.items():
        plt.plot(x, v, marker='o', linewidth=2, label=k)
    plt.grid(alpha=0.3)
    plt.xlabel("Predicted Slot (t)")
    plt.ylabel("NMSE (dB)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_bar(values: Dict[str, float], title: str, ylabel: str, save_path: str):
    keys = list(values.keys())
    vals = [values[k] for k in keys]
    plt.figure(figsize=(10, 5))
    plt.bar(keys, vals)
    plt.grid(axis='y', alpha=0.3)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_delay_vs_alt(cfg: SimCfg, nmse_curve: Dict[str, np.ndarray], save_path: str, title: str):
    alts = list(cfg.altitudes_km)
    out = {k: [] for k in nmse_curve.keys()}
    for alt in alts:
        d = min(cfg.delay_steps_from_alt_km(alt), cfg.w_out)
        for k in out.keys():
            out[k].append(nmse_curve[k][d-1])
    plt.figure(figsize=(10, 6))
    for k in out.keys():
        plt.plot(alts, out[k], marker='o', linewidth=2, label=f"{k} (NMSE@delay)")
    plt.grid(alpha=0.3)
    plt.xlabel("Altitude (km)")
    plt.ylabel("NMSE at Delay Step (dB)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_csv_nmse(save_path: str, nmse_curve: Dict[str, np.ndarray]):
    keys = list(nmse_curve.keys())
    with open(save_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + keys)
        L = len(next(iter(nmse_curve.values())))
        for i in range(L):
            w.writerow([i+1] + [float(nmse_curve[k][i]) for k in keys])


# -------------------------
# 10) Experiment runner
# -------------------------
def train_and_eval_one(base_dir: str, cfg: SimCfg, exp_tag: str, force_data: bool = False, num_workers: int = 0):
    outdir = os.path.join(base_dir, "results", exp_tag)
    ensure_dir(outdir)

    cfg_path = os.path.join(outdir, "cfg.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    print("🧾 cfg saved:", cfg_path, "|", file_exists_and_size(cfg_path))

    _, pack = get_or_make_dataset(base_dir, cfg, force=force_data)
    Xtr, Ytr = pack["train"]["X"], pack["train"]["Y"]
    Xva, Yva = pack["val"]["X"], pack["val"]["Y"]

    train_ds = SliceDataset(Xtr, Ytr)
    val_ds = SliceDataset(Xva, Yva)

    D = cfg.feat_dim
    tdnn_knet = TDNNKalmanNet(D, H=cfg.tdnn_hidden, tdnn_layers=cfg.tdnn_layers, k=cfg.tdnn_kernel, drop=cfg.tdnn_drop)
    knet = KalmanNetGRU(D, H=cfg.knet_hidden)
    scp = SCP_Zhang_TF(D, H=cfg.scp_hidden, layers=cfg.scp_layers, dense=cfg.scp_dense, dropout=cfg.dropout)

    t0 = time.time()
    tdnn_knet = train_model(cfg, tdnn_knet, "TDNN_KalmanNet", outdir, train_ds, val_ds, num_workers=num_workers)
    knet = train_model(cfg, knet, "KalmanNet_GRU", outdir, train_ds, val_ds, num_workers=num_workers)
    scp = train_model(cfg, scp, "SCP", outdir, train_ds, val_ds, num_workers=num_workers)
    print(f"⏱️ Total training (this scenario): {(time.time()-t0)/60.0:.2f} min")

    eval_models = {
        "TDNN-KalmanNet": tdnn_knet.eval(),
        "KalmanNet": knet.eval(),
        "KalmanFilter": None,
        "SCP": scp.eval(),
        "Outdated": None,
    }
    nmse_curve = eval_nmse_horizon(cfg, eval_models, val_ds, trials=cfg.horizon_trials)

    x = np.arange(1, cfg.w_out + 1)
    nmse_png = os.path.join(outdir, "nmse_vs_predslots.png")
    nmse_csv = os.path.join(outdir, "nmse_vs_predslots.csv")
    plot_nmse_curve(
        x, nmse_curve,
        title=f"NMSE vs Pred Slots | q={cfg.q_in}, L={cfg.w_out} | dop={cfg.max_ut_doppler_hz}, coh={cfg.coherence_ms}, snr={cfg.pilot_snr_db}",
        save_path=nmse_png
    )
    save_csv_nmse(nmse_csv, nmse_curve)

    delay_png = os.path.join(outdir, "nmse_at_delay_vs_altitude.png")
    plot_delay_vs_alt(
        cfg, nmse_curve,
        save_path=delay_png,
        title=f"NMSE@Delay vs Altitude | q={cfg.q_in}, L={cfg.w_out} | coh={cfg.coherence_ms}"
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

    x_example = Xva[0].float()
    times_ms = {
        "TDNN-KalmanNet": benchmark_inference(cfg, tdnn_knet, "TDNN-KalmanNet", x_example),
        "KalmanNet": benchmark_inference(cfg, knet, "KalmanNet", x_example),
        "SCP": benchmark_inference(cfg, scp, "SCP", x_example),
        "KalmanFilter": benchmark_inference(cfg, None, "KalmanFilter", x_example),
        "Outdated": benchmark_inference(cfg, None, "Outdated", x_example),
    }
    time_png = os.path.join(outdir, "inference_time_bar.png")
    plot_bar(times_ms, "Inference Time (ms/sample)", "ms/sample", time_png)

    summary_path = os.path.join(outdir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"params": params, "inference_ms": times_ms}, f, indent=2, ensure_ascii=False)

    print("\n✅ Saved results to:", outdir)
    for p in [nmse_png, nmse_csv, delay_png, params_png, time_png, summary_path]:
        print("  -", p, "|", file_exists_and_size(p))

    show_image(nmse_png, "NMSE vs Pred Slots")
    show_image(delay_png, "NMSE@Delay vs Altitude")
    show_image(params_png, "Parameter Count")
    show_image(time_png, "Inference Time")

    zip_path = os.path.join(base_dir, "results", f"{exp_tag}.zip")
    zip_path = make_zip_of_folder(outdir, zip_out=zip_path)
    print("\n📦 Zipped results:", zip_path, "|", file_exists_and_size(zip_path))
    show_download_link(zip_path)

    list_dir_tree(outdir, max_files=80)
    return nmse_curve


# -------------------------
# 11) Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="", help="output folder (default: <workspace_root>/LEO_Project)")
    parser.add_argument("--force_data", action="store_true", help="regenerate datasets")
    parser.add_argument("--q_in", type=int, default=4, help="input length q")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")

    if in_notebook():
        args = parser.parse_args(args=[])
        print("ℹ️ Notebook mode: using default arguments.")
    else:
        args = parser.parse_args()

    # Lightning AI workspace root auto-detect
    if args.base_dir.strip() == "":
        ws_root = detect_workspace_root()
        args.base_dir = os.path.join(ws_root, "LEO_Project")

    ensure_dir(args.base_dir)
    ensure_dir(os.path.join(args.base_dir, "results"))
    ensure_dir(os.path.join(args.base_dir, "datasets"))

    seed_all(42)

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    base_cfg = replace(SimCfg(), q_in=int(args.q_in))

    print(f"📌 workspace_root = {os.path.dirname(args.base_dir)}")
    print(f"📌 base_dir       = {args.base_dir}")
    print(f"📌 device         = {base_cfg.device}")
    print(f"📌 q_in           = {base_cfg.q_in}")
    print(f"📌 TDNN           = layers={base_cfg.tdnn_layers}, k={base_cfg.tdnn_kernel} (RF≈7 tuned)")

    tag = f"BASELINE_q{base_cfg.q_in}_L{base_cfg.w_out}_dop{int(base_cfg.max_ut_doppler_hz)}_coh{safe_float_tag(base_cfg.coherence_ms)}_snr{int(base_cfg.pilot_snr_db)}_{timestamp_str()}"
    train_and_eval_one(args.base_dir, base_cfg, exp_tag=tag, force_data=args.force_data, num_workers=args.num_workers)

    print("\n🎉 All done!")


if __name__ == "__main__":
    main()
