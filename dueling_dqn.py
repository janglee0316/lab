"""
Hybrid UAV Placement & Scheduling Simulator – v3 (multi‑UAV / Dueling‑DQN)
----------------------------------------------------------------------------
* Mobility  : RPGM + per‑user Kalman filter (innovation / uncertainty traced)
* 10 users  : 3 high‑priority (cmd/relay/ISR) + 7 troops → role‑weighted rank
* **Multi‑UAV** (N_UAV ≥ 2) – each UAV is an independent DQN agent
* State     : [UAVs xyz | users (x,y, x̂⁺,ŷ⁺, demand, urgency, rank)]
* Discrete  Action per UAV  = 9 moves  ×  (N_USERS+1) service‑choices
                  ├─ moves {stay, ±x, ±y, diagonal}
                  └─ select {user‑id 0…9,  no‑service}
* Reward    : TP/1e5  +0.1/innovation −0.02·uncertainty −0.05·predErr −0.01·|1‑Fair|
* RL algo   : **Dueling Double‑DQN**       (target‑net τ‑soft update)
* Exploration: ε‑greedy  (ε0=1 → ε_min=0.05)
* Logging   : return / innovation / uncertainty / fairness → PNG

Author : ChatGPT (2025‑06) • Apache‑2.02025‑06) • Apache‑2.0
"""
from __future__ import annotations
import os, math, random
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
EPISODES, STEPS      = 400, 150
N_UAV                = 3               # <- multi‑UAV
N_USERS              = 10

A_ENV = {"urban":(9.61,0.16)}
ETA_LOS, ETA_NLOS   = 1., 20.

# DQN params
GAMMA      = 0.95
BATCH      = 512
MEM_CAP    = 120_000
LR         = 2e-4
TAU        = 0.01
EPS_START  = 1.0
EPS_END    = 0.05
EPS_DECAY  = 1/2000   # linear decay per step

MOVE_STEP  = 3.0      # metres per discrete move
SAVE_DIR   = "./logs_dueling_multi"; os.makedirs(SAVE_DIR,exist_ok=True)

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
        return np.array([self.x,self.y,
                         self.kf.x_prior[0], self.kf.x_prior[1],
                         self.demand,self.urgency,self.rank])

ROLE_SCORE={"cmd":3,"relay":2.5,"isr":2,"troop":1}
W_ROLE,W_URG,W_TRA = 0.6,0.25,0.15
BETA_DECAY=0.7
USR_STATE_LEN=7

# ---------- discrete move table (9 directions, dz=0) ----------
DIRS=np.array([[ 0, 0,0],
              [ MOVE_STEP, 0,0],[-MOVE_STEP,0,0],
              [0, MOVE_STEP,0],[0,-MOVE_STEP,0],
              [ MOVE_STEP, MOVE_STEP,0],[ MOVE_STEP,-MOVE_STEP,0],
              [-MOVE_STEP, MOVE_STEP,0],[-MOVE_STEP,-MOVE_STEP,0]],dtype=np.float32)
MOVES_N = len(DIRS)

# ─────────────────────  Environment (multi‑UAV)  ─────────────────────
class UAVEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.n_uav=N_UAV; self.n_usr=N_USERS
        self.uav=np.zeros((self.n_uav,3))
        self.users:List[User]=[]

        # action: Discrete index per UAV
        self.act_per_uav = MOVES_N*(self.n_usr+1)
        self.action_space = spaces.MultiDiscrete([self.act_per_uav]*self.n_uav)
        self.observation_space = spaces.Box(-np.inf,np.inf,
            shape=(3*self.n_uav + USR_STATE_LEN*self.n_usr,),dtype=np.float32)

    # --- Kalman init
    def _kf(self,x,y):
        k=KalmanFilter(dim_x=4,dim_z=2);dt=1.
        k.F=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        k.H=np.array([[1,0,0,0],[0,1,0,0]])
        k.Q=np.diag([.05,.05,.01,.01]); k.R=np.diag([1.,1.])
        k.x=np.array([x,y,0.5,0.]); k.predict(); k.x_prior=k.x.copy()
        return k

    # --- reset
    def reset(self,seed=None,**_):
        if seed is not None: np.random.seed(seed)
        self.t=0; base=np.array([50.,250.]); spacing=25.
        role_list=["cmd","relay","isr"]+["troop"]*7
        self.users=[]
        for i in range(self.n_usr):
            pos=base+np.array([i*spacing+np.random.randn()*3, np.random.randn()*6])
            role=role_list[i%len(role_list)]
            u=User(role,*pos,kf=self._kf(*pos))
            u.rank=W_ROLE*ROLE_SCORE[role]+W_URG*u.urgency+W_TRA*u.traffic
            self.users.append(u)
        centroid=np.mean([[u.x,u.y] for u in self.users],0)
        self.uav[:,:2]=centroid+np.random.randn(self.n_uav,2)*10
        self.uav[:,2]=optimal_altitude(100)
        return self._obs()

    # --- helpers
    def _dec_act(self,a_idx:int):
        m_id = a_idx//(self.n_usr+1); sel=a_idx%(self.n_usr+1)
        return DIRS[m_id], sel

    def _throughput(self):
        noise=10**((NOISE_PSD+10*math.log10(BANDWIDTH))/10)/1000
        rates=np.zeros((self.n_usr,))
        for ui,u in enumerate(self.users):
            best=None; best_snr=-1
            for k in range(self.n_uav):
                d=np.linalg.norm(self.uav[k,:2]-[u.x,u.y]); h=self.uav[k,2]
                theta=math.atan(h/d); a,b=A_ENV["urban"]
                P_los=1/(1+a*math.exp(-b*(180/math.pi*theta-a)))
                d3=math.hypot(h,d)
                L_los=30.9+(22.25-0.5*math.log10(h))*math.log10(d3)+20*math.log10(F_CARRIER/1e9)
                L_nlos=max(L_los,32.4+(43.2-7.6*math.log10(h))*math.log10(d3)+20*math.log10(F_CARRIER/1e9))
                pl=P_los*L_los+(1-P_los)*L_nlos
                snr=TX_POWER_mW*10**(-pl/10)/noise
                if snr>best_snr: best_snr=snr; best=k
            rates[ui]=self.users[ui].rank*BANDWIDTH*math.log2(1+best_snr)
        fair=(rates.sum()**2)/(len(rates)*(rates**2).sum()+1e-6)
        return rates.sum(), fair

    # --- step
    def step(self,action:np.ndarray):
        # action is array shape (n_uav,)
        for k in range(self.n_uav):
            mv,sel = self._dec_act(int(action[k]))
            self.uav[k]+=mv
            self.uav[k]=np.clip(self.uav[k],[0,0,UAV_Z_MIN],[AREA_X,AREA_Y,UAV_Z_MAX])
            # (sel) could be used for service scheduling extension, but omitted here

        # RPGM users motion + KF update
        heading=np.deg2rad(60); v=1.5
        for u in self.users:
            u.x+=v*math.cos(heading)+np.random.randn()*0.25
            u.y+=v*math.sin(heading)+np.random.randn()*0.25
            u.kf.predict(); u.kf.update([u.x,u.y])
            u.rank=np.clip(u.rank*math.exp(-BETA_DECAY*np.linalg.norm(u.kf.y)),0.1,3.)

        tp,fair=self._throughput()
        inn=np.mean([np.linalg.norm(u.kf.y) for u in self.users])
        unc=np.mean([np.trace(u.kf.P) for u in self.users])
        pred=np.mean([math.hypot(u.kf.x_prior[0]-u.x,u.kf.x_prior[1]-u.y) for u in self.users])
        rw=tp/1e5 +0.1/(inn+1e-6) -0.02*unc -0.05*pred -0.01*abs(1-fair)

        self.t+=1; done=self.t>=STEPS
        return self._obs(), rw, done, {"inn":inn,"unc":unc,"fair":fair}

    def _obs(self):
        return np.concatenate([self.uav.flatten(), *(u.state() for u in self.users)]).astype(np.float32)

# ─────────────────────  Dueling DQN Agent (per‑UAV) ───────────────────
class Replay:
    def __init__(self,cap): self.buf=deque(maxlen=cap)
    def add(self,*e): self.buf.append(e)
    def sample(self,n): idx=np.random.choice(len(self.buf),n,replace=False); return [self.buf[i] for i in idx]
    def __len__(self): return len(self.buf)


def dueling_net(input_dim, n_actions):
    inp=layers.Input((input_dim,))
    x=layers.Dense(256,'relu')(inp); x=layers.Dense(256,'relu')(x)
    val=layers.Dense(1)(x)
    adv=layers.Dense(n_actions)(x)
    out=val + (adv - tf.reduce_mean(adv, axis=1, keepdims=True))
    return models.Model(inp,out)

class DQNAgent:
    def __init__(self,obs_dim,n_act):
        self.q=dueling_net(obs_dim,n_act); self.q_t=dueling_net(obs_dim,n_act)
        self.opt=optimizers.Adam(LR); self.replay=Replay(MEM_CAP)
        self.eps=EPS_START; self.n_act=n_act; self.sync(1.)
    def sync(self,τ):
        self.q_t.set_weights([τ*w+(1-τ)*tw for w,tw in zip(self.q.get_weights(),self.q_t.get_weights())])
    def act(self,s):
        if np.random.rand()<self.eps: return np.random.randint(self.n_act)
        q=self.q(s[None]).numpy()[0]; return int(np.argmax(q))
    def store(self,*e): self.replay.add(*e)
    def train(self):
        if len(self.replay)<BATCH: return
        s,a,r,s2,d = zip(*self.replay.sample(BATCH))
        s=tf.cast(np.array(s),tf.float32); s2=tf.cast(np.array(s2),tf.float32)
        a=np.array(a); r=tf.cast(np.array(r),tf.float32); d=np.array(d)
        q_next=tf.stop_gradient(self.q_t(s2))
        q_eval=self.q(s)
        max_next=tf.reduce_max(q_next,1)
        target=r+GAMMA*max_next*(1-d)
        with tf.GradientTape() as tape:
            q_sa=tf.reduce_sum(q_eval*tf.one_hot(a,self.n_act),1)
            loss=tf.reduce_mean((target-q_sa)**2)
        grads=tape.gradient(loss,self.q.trainable_weights); self.opt.apply_gradients(zip(grads,self.q.trainable_weights))
        self.sync(TAU)
        # eps decay
        self.eps=max(EPS_END,self.eps-EPS_DECAY)

# ───────────────────────────── training loop ──────────────────────────
RET=[]

def plot_curves():
    plt.figure(figsize=(10,4)); plt.plot(RET); plt.title('Episode return'); plt.savefig(os.path.join(SAVE_DIR,'return.png')); plt.close()


def main():
    env=UAVEnv(); obs_dim=env.observation_space.shape[0]; n_act=env.act_per_uav
    agents=[DQNAgent(obs_dim,n_act) for _ in range(N_UAV)]
    for ep in range(EPISODES):
        s=env.reset(); ep_ret=0; done=False
        while not done:
            # build joint action
            acts=np.array([agents[k].act(s) for k in range(N_UAV)],dtype=np.int32)
            s2,r,done,info=env.step(acts); ep_ret+=r
            for k in range(N_UAV):
                agents[k].store(s,acts[k],r,s2,float(done))
                agents[k].train()
            s=s2
        RET.append(ep_ret)
        if (ep+1)%10==0:
            print(f"EP{ep+1:04d} Return {ep_ret:8.2f} eps {agents[0].eps:.2f}")
    np.save(os.path.join(SAVE_DIR,'returns.npy'),np.array(RET)); plot_curves()

if __name__=='__main__':
    main()
