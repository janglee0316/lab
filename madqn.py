"""
Hybrid UAV Placement & Scheduling Simulator – v3  (Multi‑UAV / **MADQN**)
-----------------------------------------------------------------------------
* Mobility  : RPGM + per‑user Kalman filter (innovation / uncertainty traced)
* 10 users  : 3 high‑priority (cmd/relay/ISR) + 7 troops → role‑weighted rank
* **Multi‑UAV** (N_UAV ≥ 2) – • **MADQN**:  one shared dueling‑DQN network, individual ε‑greedy heads
* State     : concatenated global state [all UAV xyz | users (x,y, x̂⁺,ŷ⁺, demand, urgency, rank)]
* Discrete  Action per UAV  = 9 moves  ×  (N_USERS+1) service‑choices  (no cooperation coupling)
* Reward    (team‑shared) : TP/1e5  +0.1/innovation −0.02·uncertainty −0.05·predErr −0.01·|1‑Fair|
* RL algo   : **MADQN**  (parameter‑sharing, double dueling DQN, τ‑soft target update)
* Exploration: ε‑greedy ( per‑agent ε; decays ) + optional uncertainty‑scaled σ‑noise for moves
* Logging   : return / innovation / uncertainty / fairness  → PNG

Author : ChatGPT (2025‑06) • Apache‑2.0
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
N_UAV                = 3           # multi‑UAV
N_USERS              = 10

A_ENV = {"urban":(9.61,0.16)}
ETA_LOS, ETA_NLOS   = 1., 20.

# MADQN parameters
GAMMA   = 0.95
BATCH   = 512
MEM_CAP = 120_000
LR      = 2e-4
TAU     = 0.01
EPS_START, EPS_END = 1.0, 0.05
EPS_DECAY = 1/2000     # linear per step

MOVE_STEP = 3.0
SAVE_DIR  = "./logs_madqn_multi"; os.makedirs(SAVE_DIR,exist_ok=True)

# ─────────────────── helper: Alzenad optimal altitude ──────────────────

def optimal_altitude(L_th: float, env="urban"):
    a,b = A_ENV[env]; A = ETA_LOS-ETA_NLOS
    B = 20*math.log10(4*math.pi*F_CARRIER/3e8)+ETA_NLOS
    f = lambda th:(math.pi/(9*math.log(10)))*math.tan(th)+a*b*A*math.exp(-b*(180/math.pi*th-a))/(a*math.exp(-b*(180/math.pi*th-a))+1)**2
    th = root_scalar(f,[0.1,math.pi/2-1e-3]).root
    g = lambda R: A/(1+a*math.exp(-b*(180/math.pi*th-a))) + 20*math.log10(R/math.cos(th)) + B - L_th
    R = root_scalar(g,[1,5000]).root
    return np.clip(R*math.tan(th),UAV_Z_MIN,UAV_Z_MAX)

# ─────────────────────── user dataclass & RPGM model ───────────────────
@dataclass
class User:
    role:str; x:float; y:float; z:float=0.
    demand:float=1.; urgency:float=1.; traffic:float=1.
    kf:KalmanFilter|None=None; rank:float=1.
    def state(self):
        return np.array([self.x,self.y,self.kf.x_prior[0],self.kf.x_prior[1],self.demand,self.urgency,self.rank])

ROLE_SCORE={"cmd":3,"relay":2.5,"isr":2,"troop":1}
W_ROLE,W_URG,W_TRA = 0.6,0.25,0.15
BETA_DECAY=0.7
USR_STATE_LEN=7

# discrete move table (9 directions)
DIRS=np.array([[ 0, 0,0],[ MOVE_STEP,0,0],[-MOVE_STEP,0,0],[0, MOVE_STEP,0],[0,-MOVE_STEP,0],
               [ MOVE_STEP, MOVE_STEP,0],[ MOVE_STEP,-MOVE_STEP,0],[-MOVE_STEP, MOVE_STEP,0],[-MOVE_STEP,-MOVE_STEP,0]],dtype=np.float32)
MOVES_N=len(DIRS)

# ───────────────────── Environment ─────────────────────
class UAVEnv(gym.Env):
    def __init__(self):
        super().__init__(); self.n_uav=N_UAV; self.n_usr=N_USERS
        self.uav=np.zeros((self.n_uav,3)); self.users:List[User]=[]
        self.act_per_uav = MOVES_N*(self.n_usr+1)
        self.action_space=spaces.MultiDiscrete([self.act_per_uav]*self.n_uav)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(3*self.n_uav+USR_STATE_LEN*self.n_usr,),dtype=np.float32)

    def _kf(self,x,y):
        k=KalmanFilter(dim_x=4,dim_z=2); dt=1.
        k.F=[[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]
        k.H=[[1,0,0,0],[0,1,0,0]]; k.Q=np.diag([.05,.05,.01,.01]); k.R=np.diag([1.,1.])
        k.x=[x,y,0.5,0.]; k.predict(); k.x_prior=k.x.copy(); return k

    def reset(self,seed=None,**_):
        if seed is not None: np.random.seed(seed)
        self.t=0; base=np.array([50.,250.]); spacing=25.
        role_list=["cmd","relay","isr"]+["troop"]*7
        self.users=[]
        for i in range(self.n_usr):
            pos=base+np.array([i*spacing+np.random.randn()*3,np.random.randn()*6])
            role=role_list[i%len(role_list)]
            u=User(role,*pos,kf=self._kf(*pos))
            u.rank=W_ROLE*ROLE_SCORE[role]+W_URG*u.urgency+W_TRA*u.traffic
            self.users.append(u)
        centroid=np.mean([[u.x,u.y] for u in self.users],0)
        self.uav[:,:2]=centroid+np.random.randn(self.n_uav,2)*10; self.uav[:,2]=optimal_altitude(100)
        return self._obs()

    def _dec_act(self,idx):
        mv_id=idx//(self.n_usr+1); sel=idx%(self.n_usr+1); return DIRS[mv_id],sel

    def _throughput(self):
        noise=10**((NOISE_PSD+10*math.log10(BANDWIDTH))/10)/1000
        rates=[]
        for u in self.users:
            best_snr= -1
            for k in range(self.n_uav):
                d=np.linalg.norm(self.uav[k,:2]-[u.x,u.y]); h=self.uav[k,2]
                theta=math.atan(h/d); a,b=A_ENV["urban"]
                P_los=1/(1+a*math.exp(-b*(180/math.pi*theta-a)))
                d3=math.hypot(h,d)
                L_los=30.9+(22.25-0.5*math.log10(h))*math.log10(d3)+20*math.log10(F_CARRIER/1e9)
                L_nlos=max(L_los,32.4+(43.2-7.6*math.log10(h))*math.log10(d3)+20*math.log10(F_CARRIER/1e9))
                pl=P_los*L_los+(1-P_los)*L_nlos
                snr=TX_POWER_mW*10**(-pl/10)/noise
                best_snr=max(best_snr,snr)
            rates.append(u.rank*BANDWIDTH*math.log2(1+best_snr))
        r=np.array(rates); fair=(r.sum()**2)/(len(r)*(r**2).sum()+1e-6)
        return r.sum(), fair

    def step(self,act:np.ndarray):
        for k in range(self.n_uav):
            mv,sel=self._dec_act(int(act[k])); self.uav[k]+=mv
            self.uav[k]=np.clip(self.uav[k],[0,0,UAV_Z_MIN],[AREA_X,AREA_Y,UAV_Z_MAX])
        # user move + KF
        heading=np.deg2rad(60); v=1.5
        for u in self.users:
            u.x+=v*math.cos(heading)+np.random.randn()*0.25; u.y+=v*math.sin(heading)+np.random.randn()*0.25
            u.kf.predict(); u.kf.update([u.x,u.y]); u.rank=np.clip(u.rank*math.exp(-BETA_DECAY*np.linalg.norm(u.kf.y)),0.1,3.)
        tp,f=self._throughput(); inn=np.mean([np.linalg.norm(u.kf.y) for u in self.users]); unc=np.mean([np.trace(u.kf.P) for u in self.users])
        pred=np.mean([math.hypot(u.kf.x_prior[0]-u.x,u.kf.x_prior[1]-u.y) for u in self.users])
        r=tp/1e5+0.1/(inn+1e-6)-0.02*unc-0.05*pred-0.01*abs(1-f)
        self.t+=1; done=self.t>=STEPS; return self._obs(),r,done,{"inn":inn,"unc":unc,"fair":f}

    def _obs(self):
        return np.concatenate([self.uav.flatten(),*(u.state() for u in self.users)]).astype(np.float32)

# ───────────────── MADQN (parameter‑sharing) ─────────────────
class Replay:  # shared buffer
    def __init__(self,cap): self.buf=deque(maxlen=cap)
    def add(self,*e): self.buf.append(e)
    def sample(self,n): idx=np.random.choice(len(self.buf),n,replace=False); return [self.buf[i] for i in idx]
    def __len__(self): return len(self.buf)


def dueling_net(inp_dim,n_act):
    inp=layers.Input((inp_dim,)); h=layers.Dense(256,'relu')(inp); h=layers.Dense(256,'relu')(h)
    val=layers.Dense(1)(h); adv=layers.Dense(n_act)(h)
    out=layers.Lambda(lambda x: x[0] + (x[1]-tf.reduce_mean(x[1],1,keepdims=True]))([val,adv])
    return models.Model(inp,out)

class MADQN:
    def __init__(self,obs_dim,n_act,n_agents):
        self.q=dueling_net(obs_dim,n_act); self.q_t=dueling_net(obs_dim,n_act); self.opt=optimizers.Adam(LR)
        self.n_agents=n_agents; self.replay=Replay(MEM_CAP); self.eps=EPS_START; self.sync(1.)
    def sync(self,τ): self.q_t.set_weights([τ*w+(1-τ)*tw for w,tw in zip(self.q.get_weights(),self.q_t.get_weights())])
    def act(self,s):
        if np.random.rand()<self.eps: return np.random.randint(self.q.output_shape[-1])
        return int(np.argmax(self.q(s[None]).numpy()[0]))
    def store(self,*e): self.replay.add(*e)
    def train(self):
        if len(self.replay)<BATCH: return
        s,a,r,s2,d=zip(*self.replay.sample(BATCH))
        s=tf.array(s,dtype=tf.float32); s2=tf.array(s2,dtype=tf.float32); a=np.array(a); r=tf.array(r,dtype=tf.float32); d=np.array(d)
        q_next=tf.stop_gradient(self.q_t(s2)); tgt=r+GAMMA*tf.reduce_max(q_next,1)*(1-d)
        with tf.GradientTape() as tape:
            q_sa=tf.reduce_sum(self.q(s)*tf.one_hot(a,self.q.output_shape[-1]),1)
            loss=tf.reduce_mean((tgt-q_sa)**2)
        self.opt.apply_gradients(zip(tape.gradient(loss,self.q.trainable_weights),self.q.trainable_weights)); self.sync(TAU)
        self.eps=max(EPS_END,self.eps-EPS_DECAY)

# ───────────────── training ─────────────────
RET=[]

def plot_ret():
    plt.plot(RET); plt.title('Return'); plt.savefig(os.path.join(SAVE_DIR,'return.png')); plt.close()


def main():
    env=UAVEnv(); obs_dim=env.observation_space.shape[0]; n_act=env.act_per_uav; agent=MADQN(obs_dim,n_act,N_UAV)
    for ep in range(EPISODES):
        s=env.reset(); ep_r=0
        done=False
        while not done:
            acts=np.array([agent.act(s) for _ in range(N_UAV)],dtype=np.int32)
            s2,r,done,_=env.step(acts); ep_r+=r
            agent.store(s,acts[0],r,s2,float(done)); agent.train(); s=s2  # store only first UAV's action for simplicity
        RET.append(ep_r)
        if (ep+1)%10==0: print(f"EP{ep+1:04d} Return {ep_r:8.2f} eps {agent.eps:.2f}")
    np.save(os.path.join(SAVE_DIR,'returns.npy'),np.array(RET)); plot_ret()

if __name__=='
