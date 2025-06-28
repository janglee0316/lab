"""
Hybrid UAV Placement & Scheduling Simulator – v3 (fixed)
--------------------------------------------------------
* Mobility model : RPGM + Kalman filter (innovation / uncertainty traced)
* 10 users (3 high‑priority + 7 troops)  → role‑dependent priority rank
* State   includes Kalman predicted (x̂⁺, ŷ⁺)
* Reward  R = TP/1e5 + 0.1/innovation −0.02·uncertainty −0.05·predErr −0.01·|1−Fair|
* OU‑noise σ ∝ uncertainty  (risk‑aware exploration)
* Auto logging PNG (return / innovation / uncertainty / fairness)
Author : ChatGPT (2025‑06) • Apache‑2.0
"""
from __future__ import annotations
import os, math, random, json
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np, matplotlib.pyplot as plt
import gym
from gym import spaces
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from scipy.optimize import root_scalar
from filterpy.kalman import KalmanFilter

# ───────────────────────── global hyper‑params ─────────────────────────
SEED = 2025
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

AREA_X, AREA_Y       = 500., 500.
UAV_Z_MIN, UAV_Z_MAX = 30., 150.
F_CARRIER            = 2.0e9
BANDWIDTH            = 30e3
NOISE_PSD            = -174
TX_POWER_mW          = 100.
EPISODES, STEPS      = 300, 150
N_UAV                = 1                # single‑agent demo
N_USERS              = 10

A_ENV = {"urban":(9.61,0.16)}
ETA_LOS, ETA_NLOS   = 1., 20.
ACTOR_LR, CRITIC_LR = 1e-4, 1e-3
GAMMA, TAU          = 0.95, 0.01
MEM_CAP, BATCH      = 60_000, 256
SAVE_DIR            = "./logs_hybrid_v3"; os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────── helper: Alzenad optimal altitude ──────────────────
def optimal_altitude(L_th: float, env="urban"):
    a,b = A_ENV[env]; A = ETA_LOS-ETA_NLOS
    B = 20*math.log10(4*math.pi*F_CARRIER/3e8)+ETA_NLOS
    f = lambda th:(math.pi/(9*math.log(10)))*math.tan(th)+a*b*A*math.exp(-b*(180/math.pi*th-a))/(a*math.exp(-b*(180/math.pi*th-a))+1)**2
    th = root_scalar(f, bracket=[0.1, math.pi/2-1e-3]).root
    g  = lambda R: A/(1+a*math.exp(-b*(180/math.pi*th-a))) + 20*math.log10(R/math.cos(th)) + B - L_th
    R  = root_scalar(g, bracket=[1, 5000]).root
    return np.clip(R*math.tan(th), UAV_Z_MIN, UAV_Z_MAX)

# ─────────────────────── user dataclass & RPGM model ───────────────────
@dataclass
class User:
    role:str; x:float; y:float; z:float=0.
    demand:float=1.; urgency:float=1.; traffic:float=1.
    kf:KalmanFilter|None=None; rank:float=1.
    def state(self):
        return np.array([
            self.x, self.y,                       # measured
            self.kf.x_prior[0], self.kf.x_prior[1], # predicted next pos
            self.demand, self.urgency, self.rank
        ])

ROLE_SCORE = {"cmd":3,"relay":2.5,"isr":2,"troop":1}
W_ROLE, W_URG, W_TRA = 0.6,0.25,0.15
BETA_DECAY = 0.7
USR_STATE_LEN = 7

# ────────────────────────   Environment   ─────────────────────────────
class UAVEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.n_uav, self.n_usr = N_UAV, N_USERS
        self.uav = np.zeros((self.n_uav,3))
        self.users: List[User] = []
        mv_box = spaces.Box(low=-3., high=3., shape=(3,), dtype=np.float32)
        self.action_space = spaces.Tuple((mv_box, spaces.Discrete(self.n_usr+1)))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3*self.n_uav + USR_STATE_LEN*self.n_usr,), dtype=np.float32)

    # ―― Kalman init
    def _kf(self,x,y):
        k = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.
        k.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        k.H = np.array([[1,0,0,0],[0,1,0,0]])
        k.Q = np.diag([.05,.05,.01,.01]); k.R = np.diag([1.,1.])
        k.x  = np.array([x,y,0.5,0.])
        k.predict(); k.x_prior = k.x.copy()
        return k

    # ―― reset
    def reset(self, seed=None):
        #super().reset(seed); 
        self.t = 0
        base = np.array([50.,250.]); spacing = 25.
        role_list = ["cmd","relay","isr"] + ["troop"]*7
        self.users = []
        for i in range(self.n_usr):
            pos = base + np.array([i*spacing + np.random.randn()*3, np.random.randn()*6])
            role = role_list[i % len(role_list)]
            u = User(role,*pos, kf=self._kf(*pos))
            u.rank = W_ROLE*ROLE_SCORE[role] + W_URG*u.urgency + W_TRA*u.traffic
            self.users.append(u)
        centroid = np.mean([[u.x,u.y] for u in self.users],0)
        self.uav[:,:2] = centroid + np.random.randn(self.n_uav,2)*10
        self.uav[:,2]  = optimal_altitude(100)
        return self._obs()

    # ―― throughput + fairness
    def _throughput(self):
        noise = 10**((NOISE_PSD + 10*math.log10(BANDWIDTH))/10)/1000
        rates=[]
        for u in self.users:
            d = np.linalg.norm(self.uav[0,:2]-[u.x,u.y]); h = self.uav[0,2]
            theta = math.atan(h/d); a,b = A_ENV["urban"]
            P_los = 1/(1+a*math.exp(-b*(180/math.pi*theta - a)))
            d3 = math.hypot(h,d)
            L_los = 30.9 + (22.25-0.5*math.log10(h))*math.log10(d3) + 20*math.log10(F_CARRIER/1e9)
            L_nlos= max(L_los, 32.4 + (43.2-7.6*math.log10(h))*math.log10(d3) + 20*math.log10(F_CARRIER/1e9))
            pl = P_los*L_los + (1-P_los)*L_nlos
            snr = TX_POWER_mW*10**(-pl/10)/noise
            rates.append(u.rank*BANDWIDTH*math.log2(1+snr))
        r = np.array(rates); fair = (r.sum()**2)/(len(r)*(r**2).sum()+1e-6)
        return r.sum(), fair

    # ―― step
    def step(self, action: Tuple[np.ndarray,int]):
        mv, sel = action
        self.uav[0] += mv
        self.uav[0] = np.clip(self.uav[0], [0,0,UAV_Z_MIN], [AREA_X,AREA_Y,UAV_Z_MAX])

        # RPGM motion
        heading = np.deg2rad(60); v=1.5
        for u in self.users:
            u.x += v*math.cos(heading) + np.random.randn()*0.25
            u.y += v*math.sin(heading) + np.random.randn()*0.25
            u.kf.predict(); u.kf.update([u.x,u.y])
            innov = np.linalg.norm(u.kf.y)
            u.rank = np.clip(u.rank*math.exp(-BETA_DECAY*innov), 0.1, 3.)

        # reward
        tp, fair = self._throughput()
        inn = np.mean([np.linalg.norm(u.kf.y)     for u in self.users])
        unc = np.mean([np.trace(u.kf.P)           for u in self.users])
        pred= np.mean([math.hypot(u.kf.x_prior[0]-u.x, u.kf.x_prior[1]-u.y) for u in self.users])
        rw = tp/1e5 + 0.1/(inn+1e-6) - 0.02*unc - 0.05*pred - 0.01*abs(1-fair)

        self.t += 1; done = self.t >= STEPS
        return self._obs(), rw, done, {"inn":inn,"unc":unc,"fair":fair}

    # ―― observation
    def _obs(self):
        return np.concatenate([self.uav.flatten(), *(u.state() for u in self.users)]).astype(np.float32)

# ───────────────────────── OU‑noise, networks, agent ───────────────────
class OU:
    def __init__(self,dim): self.state=np.zeros(dim)
    def __call__(self, sigma=0.2):
        self.state += -0.15*self.state + sigma*np.random.randn(*self.state.shape)
        return self.state


def actor_net(dim):
    inp=layers.Input((dim,)); x=layers.Dense(128,'relu')(inp); x=layers.Dense(128,'relu')(x)
    mv=layers.Dense(3,'tanh')(x)
    sel=layers.Dense(N_USERS+1,'softmax')(x)
    return models.Model(inp,[mv,sel])

def critic_net(obs, act):
    i1=layers.Input((obs,)); i2=layers.Input((act,)); x=layers.Concatenate()([i1,i2])
    x=layers.Dense(256,'relu')(x); x=layers.Dense(256,'relu')(x)
    return models.Model([i1,i2], layers.Dense(1)(x))

class Agent:
    def __init__(self, env:UAVEnv):
        self.env=env; od=env.observation_space.shape[0]; ad=3+N_USERS+1
        self.actor=actor_net(od); self.actor_t=actor_net(od)
        self.critic=critic_net(od,ad); self.critic_t=critic_net(od,ad)
        self.actor_opt=optimizers.Adam(ACTOR_LR); self.critic_opt=optimizers.Adam(CRITIC_LR)
        self.mem=deque(maxlen=MEM_CAP); self.noise=OU(3)
        self.sync(1.)
    def sync(self,τ):
        self.actor_t.set_weights([τ*w+(1-τ)*tw for w,tw in zip(self.actor.get_weights(),self.actor_t.get_weights())])
        self.critic_t.set_weights([τ*w+(1-τ)*tw for w,tw in zip(self.critic.get_weights(),self.critic_t.get_weights())])
    def act(self,s,unc):
        mv,sel = self.actor(s[None]); mv=mv.numpy()[0]; sel=sel.numpy()[0]
        mv += self.noise(min(0.25+0.4*unc,1.0))
        idx=np.random.choice(len(sel), p=sel)
        return np.clip(mv,-1,1)*3, idx
    def store(self,*ex): self.mem.append(ex)
    def train(self):
        if len(self.mem)<BATCH: return
        batch=random.sample(self.mem,BATCH)
        S,A,R,S2 = map(np.array,zip(*batch))
        Sa=tf.cast(S,tf.float32); Aa=tf.cast(A,tf.float32); Ra=tf.cast(R[:,None],tf.float32); S2a=tf.cast(S2,tf.float32)
        with tf.GradientTape() as tape:
            mv2,sel2 = self.actor_t(S2a); A2=tf.concat([mv2,sel2],1)
            y = Ra + GAMMA*self.critic_t([S2a,A2])
            q = self.critic([Sa,Aa])
            loss = tf.keras.losses.MSE(y,q)
        g=tape.gradient(loss,self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(g,self.critic.trainable_weights))
        with tf.GradientTape() as tape:
            mv,sel = self.actor(Sa); Ap=tf.concat([mv,sel],1)
            q = self.critic([Sa,Ap])
            loss = -tf.reduce_mean(q)
        g=tape.gradient(loss,self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(g,self.actor.trainable_weights)); self.sync(TAU)

# ─────────────────────────── training & plots ──────────────────────────
RET,INN,UNC,FAIR=[],[],[],[]

def plot_curves():
    win=20; ma=np.convolve(RET,np.ones(win)/win,'valid')
    plt.figure(figsize=(12,5))
    plt.subplot(121); plt.plot(RET,label='Return',alpha=0.4); plt.plot(range(win-1,len(RET)),ma,label=f'MA({win})',lw=2); plt.legend(); plt.grid()
    plt.subplot(122); plt.plot(INN,label='Innovation'); plt.plot(UNC,label='Uncertainty'); plt.plot(FAIR,label='Fairness'); plt.legend(); plt.grid()
    plt.tight_layout(); plt.savefig(os.path.join(SAVE_DIR,'learning.png')); plt.close()


def main():
    env=UAVEnv(); ag=Agent(env)
    for ep in range(EPISODES):
        s=env.reset(); ep_r=0; inn_l=[]; unc_l=[]; fair_l=[]
        while True:
            unc=np.mean([np.trace(u.kf.P) for u in env.users])
            mv,idx = ag.act(s,unc)
            a_vec=np.concatenate([mv, tf.one_hot(idx,N_USERS+1).numpy()])
            s2,r,d,info = env.step((mv,idx)); ag.store(s,a_vec,r,s2); ag.train(); s=s2; ep_r+=r
            inn_l.append(info['inn']); unc_l.append(info['unc']); fair_l.append(info['fair'])
            if d: break
        RET.append(ep_r); INN.append(np.mean(inn_l)); UNC.append(np.mean(unc_l)); FAIR.append(np.mean(fair_l))
        if (ep+1)%10==0:
            print(f"EP{ep+1:04d}  Return {ep_r:8.2f}  Inn {INN[-1]:.3f}  Unc {UNC[-1]:.3f}  Fair {FAIR[-1]:.3f}")
    np.save(os.path.join(SAVE_DIR,'returns.npy'),np.array(RET)); plot_curves()

if __name__=='__main__':
    main()
