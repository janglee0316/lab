"""
MEC-Edge Multi-UAV Trajectory Simulator  (freq-rank, minimal)
-------------------------------------------------------------
*   Mobility : RPGM + Kalman predictor
*   Rank     : sliding-window 통신 빈도 (role 제거)
*   Agents   : N_UAV UAVs, parameter-sharing MADDPG
*   Reward   : TP/1e5  − α·predErr                (α = 0.05)
*   Author   : ChatGPT 2025-06 • Apache-2.0
"""
from __future__ import annotations
import os, math, random, collections
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np, matplotlib.pyplot as plt
import gym, tensorflow as tf
from gym import spaces
from tensorflow.keras import layers, models, optimizers
from scipy.optimize import root_scalar
from filterpy.kalman import KalmanFilter

# ──────────────────────── 1. Global params ────────────────────────────
SEED = 2025
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

AREA_X, AREA_Y       = 500., 500.
Z_MIN,  Z_MAX        = 30., 150.
F_CARRIER            = 2e9          # Hz
BAND                 = 30e3         # Hz
NOISE_PSD           = -174          # dBm/Hz
P_TX_mW              = 100.
N_UAV, N_USER        = 3, 10
EPIS, STEPS          = 300, 150

A_ENV = {"urban":(9.61,0.16)}
ETA_L, ETA_N        = 1., 20.

ACT_LR, CRI_LR      = 1e-4, 1e-3
GAMMA, TAU          = 0.95, 0.01
MEM,   BATCH        = 100_000, 512

WIN_FREQ            = 10           # sliding window length
ALPHA               = 0.05         # reward weight for predErr
OU_SIGMA            = 0.3          # exploration noise scale
SAVE_DIR            = "./mec_maddpg"; os.makedirs(SAVE_DIR,exist_ok=True)

# ──────────────────── 2. Helper – optimal altitude ─────────────────────
def optimal_h(L_th=100):
    a,b = A_ENV["urban"]; A = ETA_L-ETA_N
    B = 20*math.log10(4*math.pi*F_CARRIER/3e8)+ETA_N
    f = lambda t:(math.pi/(9*math.log(10)))*math.tan(t)+a*b*A*math.exp(-b*(180/math.pi*t-a))/(a*math.exp(-b*(180/math.pi*t-a))+1)**2
    th = root_scalar(f,bracket=[.1,math.pi/2-.01]).root
    g  = lambda R:A/(1+a*math.exp(-b*(180/math.pi*th-a)))+20*math.log10(R/math.cos(th))+B-L_th
    R  = root_scalar(g,bracket=[1,5000]).root
    return np.clip(R*math.tan(th), Z_MIN, Z_MAX)

# ───────────────────── 3. User dataclass ───────────────────────────────
@dataclass
class User:
    x:float; y:float; kf:KalmanFilter; freq:collections.deque; rank:float
    def state(self):                       # 5-dim user state
        return np.array([self.x,self.y,
                         self.kf.x_prior[0],self.kf.x_prior[1],
                         self.rank])

USR_S = 5                                   # per-user state length

# ───────────────────── 4. Environment ──────────────────────────────────
class UAVEnv(gym.Env):
    """RPGM + Kalman mobility, rank=frequency."""
    def __init__(self):
        super().__init__()
        self.uav = np.zeros((N_UAV,3))
        mv_box = spaces.Box(-3.,3.,(3,),np.float32)
        self.action_space = spaces.Tuple(tuple([mv_box]*N_UAV))
        obs_dim = 3*N_UAV + USR_S*N_USER
        self.observation_space = spaces.Box(-np.inf,np.inf,(obs_dim,),np.float32)
        self.users:List[User] = []

    # -------- Kalman helper
    def _KF(self,x,y):
        k=KalmanFilter(dim_x=4,dim_z=2); dt=1.
        k.F=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        k.H=np.array([[1,0,0,0],[0,1,0,0]])
        k.Q=np.diag([.05,.05,.01,.01]); k.R=np.diag([1.,1.])
        k.x=[x,y,0.5,0.]; k.predict()
        return k

    # -------- reset
    def reset(self,seed=None):
        self.t=0
        base=np.array([50.,250.]); spacing=25.
        self.users=[]
        for i in range(N_USER):
            pos=base+np.array([i*spacing+np.random.randn()*3,
                                np.random.randn()*6])
            self.users.append(User(*pos,
                                   kf=self._KF(*pos),
                                   freq=collections.deque(maxlen=WIN_FREQ),
                                   rank=0.1))
        cen=np.mean([[u.x,u.y] for u in self.users],0)
        self.uav[:,:2]=cen+np.random.randn(N_UAV,2)*10
        self.uav[:,2]=optimal_h()
        return self._obs()

    # -------- rank update
    def _rank(self,u:User):
        return max(0.1, sum(u.freq)/WIN_FREQ)

    # -------- rate + fairness
    def _rate(self):
        noise=10**((NOISE_PSD+10*math.log10(BAND))/10)/1000
        rates=[]
        for u in self.users:
            bestsnr=0
            for k in range(N_UAV):
                d=np.linalg.norm(self.uav[k,:2]-[u.x,u.y]); h=self.uav[k,2]
                theta=math.atan(h/d); a,b=A_ENV["urban"]
                P_los=1/(1+a*math.exp(-b*(180/math.pi*theta-a)))
                d3=math.hypot(h,d)
                L_los=30.9+(22.25-0.5*math.log10(h))*math.log10(d3)+20*math.log10(F_CARRIER/1e9)
                L_nlos=max(L_los,32.4+(43.2-7.6*math.log10(h))*math.log10(d3)+20*math.log10(F_CARRIER/1e9))
                pl=P_los*L_los+(1-P_los)*L_nlos
                snr=P_TX_mW*10**(-pl/10)/noise
                bestsnr=max(bestsnr,snr)
            rates.append(u.rank*BAND*math.log2(1+bestsnr))
        r=np.array(rates); fair=(r.sum()**2)/(len(r)*(r**2).sum()+1e-6)
        return r.sum(),fair

    # -------- step
    def step(self,actions:Tuple[np.ndarray,...]):
        # 1) UAV move
        for k,mv in enumerate(actions):
            self.uav[k]+=mv
            self.uav[k]=np.clip(self.uav[k],
                                [0,0,Z_MIN],[AREA_X,AREA_Y,Z_MAX])

        # 2) 통신 이벤트 (p=0.3) → freq deque
        req=np.random.rand(N_USER)<0.3
        for i,f in enumerate(req): self.users[i].freq.append(int(f))

        # 3) RPGM move + Kalman + rank
        head=np.deg2rad(60); v=1.5
        for u in self.users:
            u.x+=v*math.cos(head)+np.random.randn()*0.25
            u.y+=v*math.sin(head)+np.random.randn()*0.25
            u.kf.predict(); u.kf.update([u.x,u.y])
            u.rank=self._rank(u)

        tp,fair   = self._rate()
        pred_err  = np.mean([math.hypot(u.kf.x_prior[0]-u.x,
                                        u.kf.x_prior[1]-u.y) for u in self.users])
        reward    = tp/1e5 - ALPHA*pred_err

        self.t+=1; done=self.t>=STEPS
        return self._obs(), [reward]*N_UAV, done, {}

    # -------- observation
    def _obs(self):
        return np.concatenate([self.uav.flatten(),*(u.state() for u in self.users)]).astype(np.float32)

# ───────────────────── 5. MADDPG (shared) ──────────────────────────────
def mlp(in_dim,out_dim,act_last=None):
    return models.Sequential([
        layers.Input((in_dim,)),
        layers.Dense(256,'relu'), layers.Dense(256,'relu'),
        layers.Dense(out_dim,act_last)
    ])

class MADDPG:
    def __init__(self,obs_dim,act_dim):
        self.actor   = mlp(obs_dim, act_dim, 'tanh')
        self.t_actor = mlp(obs_dim, act_dim, 'tanh')
        self.critic  = mlp(obs_dim+act_dim*N_UAV, 1)
        self.t_critic= mlp(obs_dim+act_dim*N_UAV, 1)
        self.a_opt   = optimizers.Adam(ACT_LR)
        self.c_opt   = optimizers.Adam(CRI_LR)
        self.buffer  = collections.deque(maxlen=MEM)
        self.sync(1.)

    # ------- sync targets
    def sync(self,tau):
        self.t_actor.set_weights([tau*w+(1-tau)*tw for w,tw in
                                  zip(self.actor.get_weights(),
                                      self.t_actor.get_weights())])
        self.t_critic.set_weights([tau*w+(1-tau)*tw for w,tw in
                                   zip(self.critic.get_weights(),
                                       self.t_critic.get_weights())])

    # ------- act (shared actor)
    def act(self,obs):
        mv = self.actor(obs[None]).numpy()[0][:3]          # same for all UAVs
        mv += np.random.randn(3)*OU_SIGMA
        mv = np.clip(mv,-1,1)*3
        return tuple([mv.copy() for _ in range(N_UAV)])

    # ------- store
    def remember(self,*ex): self.buffer.append(ex)

    # ------- train
    def train(self):
        if len(self.buffer)<BATCH: return
        batch=random.sample(self.buffer,BATCH)
        S,A,R,S2=map(np.array,zip(*batch))
        S=tf.convert_to_tensor(S); S2=tf.convert_to_tensor(S2)
        A=tf.convert_to_tensor(A); R=tf.convert_to_tensor(R[:,None],tf.float32)
        A_flat = tf.repeat(A, repeats=N_UAV, axis=1)   # shared action replicated

        # critic
        with tf.GradientTape() as tape:
            y = R + GAMMA*self.t_critic(tf.concat([S2,A_flat],1))
            q = self.critic(tf.concat([S,A_flat],1))
            c_loss = tf.keras.losses.MSE(y,q)
        grads=tape.gradient(c_loss,self.critic.trainable_weights)
        self.c_opt.apply_gradients(zip(grads,self.critic.trainable_weights))

        # actor
        with tf.GradientTape() as tape:
            A_pred = self.actor(S)                         # (B,3)
            A_pred_flat = tf.repeat(A_pred, repeats=N_UAV, axis=1)
            q = self.critic(tf.concat([S,A_pred_flat],1))
            a_loss = -tf.reduce_mean(q)
        grads=tape.gradient(a_loss,self.actor.trainable_weights)
        self.a_opt.apply_gradients(zip(grads,self.actor.trainable_weights))
        self.sync(TAU)

# ───────────────────── 6. Training loop ────────────────────────────────
def main():
    env = UAVEnv()
    obs_dim = env.observation_space.shape[0]; act_dim = 3
    agent = MADDPG(obs_dim, act_dim)

    returns=[]
    for ep in range(EPIS):
        obs = env.reset(); ep_r = 0
        while True:
            actions = agent.act(obs)
            obs2, rews, done, _ = env.step(actions)
            agent.remember(obs, actions[0], np.mean(rews), obs2)  # shared
            agent.train()
            obs = obs2; ep_r += np.mean(rews)
            if done: break
        returns.append(ep_r)
        if (ep+1)%10==0:
            print(f"EP{ep+1:03d}  Return {ep_r:8.2f}")

    plt.plot(returns); plt.title("Return"); plt.savefig(os.path.join(SAVE_DIR,"return.png")); plt.close()

if __name__=="__main__":
    main()
