"""
MADDPG – Multi-UAV Trajectory Simulator (freq-rank)
---------------------------------------------------
* Mobility  : RPGM + Kalman prediction
* Rank      : sliding-window 통신 빈도 기반 (role 제거)
* Agents    : 3 UAVs, parameter-sharing MADDPG
* Reward    : TP/1e5 + 0.1/inno −0.02·unc −0.05·pred −0.01·|1-Fair|
Author      : ChatGPT 2025-06 • Apache-2.0
"""
from __future__ import annotations
import os, math, random, collections, itertools
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np, matplotlib.pyplot as plt
import gym, tensorflow as tf
from gym import spaces
from tensorflow.keras import layers, models, optimizers
from scipy.optimize import root_scalar
from filterpy.kalman import KalmanFilter

# ───────── config ─────────
SEED=2025
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

AREA_X,AREA_Y   = 500.,500.
UAV_Z_MIN,UAV_Z_MAX = 30.,150.
F_CAR=2e9; BAND=30e3; NOISE_PSD=-174; P_Tx_mW=100
N_UAV=3;   N_USER=10;  EPIS=300; STEPS=150
A_ENV={"urban":(9.61,0.16)}; ETA_L,ETA_N=1.,20.
ACT_LR,CRI_LR=1e-4,1e-3; GAMMA,TAU=0.95,0.01
MEM,BATCH=100_000,512
WIN_FREQ=10; W_FREQ=1.0   # rank weight
SAVE_DIR="./maddpg_freq"; os.makedirs(SAVE_DIR,exist_ok=True)

# ───── helper: Alzenad altitude ─────
def opt_h(Lth):
    a,b=A_ENV["urban"]; A=ETA_L-ETA_N; B=20*math.log10(4*math.pi*F_CAR/3e8)+ETA_N
    f=lambda t:(math.pi/(9*math.log(10)))*math.tan(t)+a*b*A*math.exp(-b*(180/math.pi*t-a))/(a*math.exp(-b*(180/math.pi*t-a))+1)**2
    th=root_scalar(f,bracket=[.1,math.pi/2-.01]).root
    g=lambda R:A/(1+a*math.exp(-b*(180/math.pi*th-a)))+20*math.log10(R/math.cos(th))+B-Lth
    R=root_scalar(g,bracket=[1,5000]).root
    return np.clip(R*math.tan(th),UAV_Z_MIN,UAV_Z_MAX)

# ───── user dataclass ─────
@dataclass
class User:
    x:float; y:float; kf:KalmanFilter; freq:collections.deque; rank:float
    def state(self):
        return np.array([self.x,self.y,
                         self.kf.x_prior[0],self.kf.x_prior[1],
                         self.rank])

USR_S = 5   # state length per user

# ───── environment ─────
class UAVEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.uav=np.zeros((N_UAV,3))
        mv=spaces.Box(-3.,3.,(3,),np.float32)
        sel=spaces.Discrete(N_USER+1)
        self.action_space=spaces.Tuple(tuple([spaces.Tuple((mv,sel))]*N_UAV))
        dim=3*N_UAV + USR_S*N_USER
        self.observation_space=spaces.Box(-np.inf,np.inf,(dim,),np.float32)
        self.users:List[User]=[]

    # Kalman
    def _KF(self,x,y):
        k=KalmanFilter(dim_x=4,dim_z=2); dt=1.
        k.F=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        k.H=np.array([[1,0,0,0],[0,1,0,0]])
        k.Q=np.diag([.05,.05,.01,.01]); k.R=np.diag([1.,1.]); k.x=[x,y,0.5,0]; k.predict()
        return k

    def reset(self,seed=None):
        self.t=0
        base=np.array([50.,250.]); spacing=25.
        self.users=[]
        for i in range(N_USER):
            pos=base+np.array([i*spacing+np.random.randn()*3,np.random.randn()*6])
            self.users.append(User(*pos,kf=self._KF(*pos),
                                   freq=collections.deque(maxlen=WIN_FREQ),
                                   rank=0.5))
        cen=np.mean([[u.x,u.y] for u in self.users],0)
        self.uav[:,:2]=cen+np.random.randn(N_UAV,2)*10
        self.uav[:,2]=opt_h(100)
        return self._obs()

    # rank update
    def _rank(self,u:User):
        return W_FREQ*sum(u.freq)/WIN_FREQ if u.freq else 0.1

    # rate & fairness
    def _rate(self):
        noise=10**((NOISE_PSD+10*math.log10(BAND))/10)/1000
        rates=[]
        for u in self.users:
            best=0; bestsnr=0
            for k in range(N_UAV):
                d=np.linalg.norm(self.uav[k,:2]-[u.x,u.y]); h=self.uav[k,2]
                theta=math.atan(h/d); a,b=A_ENV["urban"]
                Plos=1/(1+a*math.exp(-b*(180/math.pi*theta-a)))
                d3=math.hypot(h,d)
                Llos=30.9+(22.25-0.5*math.log10(h))*math.log10(d3)+20*math.log10(F_CAR/1e9)
                Lnlos=max(Llos,32.4+(43.2-7.6*math.log10(h))*math.log10(d3)+20*math.log10(F_CAR/1e9))
                pl=Plos*Llos+(1-Plos)*Lnlos
                snr=P_Tx_mW*10**(-pl/10)/noise
                if snr>bestsnr: bestsnr=snr
            rates.append(u.rank*BAND*math.log2(1+bestsnr))
        r=np.array(rates); fair=(r.sum()**2)/(len(r)*(r**2).sum()+1e-6)
        return r.sum(),fair

    def step(self,actions):
        # action unpack
        for k,(mv,sel) in enumerate(actions):
            self.uav[k]+=mv
            self.uav[k]=np.clip(self.uav[k],
                                [0,0,UAV_Z_MIN],
                                [AREA_X,AREA_Y,UAV_Z_MAX])
        # 통신 발생 (Bernoulli 0.3)
        req=np.random.rand(N_USER)<0.3
        for i,flag in enumerate(req): self.users[i].freq.append(int(flag))

        # 이동 & kalman
        head=np.deg2rad(60); v=1.5
        for u in self.users:
            u.x+=v*math.cos(head)+np.random.randn()*0.25
            u.y+=v*math.sin(head)+np.random.randn()*0.25
            u.kf.predict(); u.kf.update([u.x,u.y])
            u.rank=max(0.1,self._rank(u))
        tp,fair=self._rate()
        inn=np.mean([np.linalg.norm(u.kf.y) for u in self.users])
        unc=np.mean([np.trace(u.kf.P) for u in self.users])
        pred=np.mean([math.hypot(u.kf.x_prior[0]-u.x,u.kf.x_prior[1]-u.y) for u in self.users])
        rw=tp/1e5+0.1/(inn+1e-6)-0.02*unc-0.05*pred-0.01*abs(1-fair)
        self.t+=1; done=self.t>=STEPS
        return self._obs(), [rw]*N_UAV, done, {"inn":inn,"unc":unc,"fair":fair}

    def _obs(self):
        return np.concatenate([self.uav.flatten(),*(u.state() for u in self.users)]).astype(np.float32)

# ───── MADDPG (parameter-sharing actors, central critic) ─────
def mlp(in_dim,out_dim,act_last=None):
    m=models.Sequential([layers.Input((in_dim,)),
                         layers.Dense(256,'relu'),
                         layers.Dense(256,'relu'),
                         layers.Dense(out_dim,act_last)])
    return m

class MADDPG:
    def __init__(self,obs_dim,act_dim):
        self.actor=mlp(obs_dim,act_dim,'tanh')
        self.actors=[self.actor for _ in range(N_UAV)]   # 공유
        self.target_actor=mlp(obs_dim,act_dim,'tanh')
        self.critic=mlp(obs_dim+act_dim*N_UAV,1,None)
        self.t_critic=mlp(obs_dim+act_dim*N_UAV,1,None)
        self.a_opt=optimizers.Adam(ACT_LR); self.c_opt=optimizers.Adam(CRI_LR)
        self._sync(1.)
        self.mem=collections.deque(maxlen=MEM)

    def _sync(self,tau):
        self.target_actor.set_weights([tau*w+(1-tau)*tw
                             for w,tw in zip(self.actor.get_weights(),
                                             self.target_actor.get_weights())])
        self.t_critic.set_weights([tau*w+(1-tau)*tw
                             for w,tw in zip(self.critic.get_weights(),
                                             self.t_critic.get_weights())])

    def remember(self,*ex): self.mem.append(ex)

    def act(self,obs_all,unc):
        acts=[]
        sigma=min(0.25+0.4*unc,1.)
        for _ in range(N_UAV):
            mv=self.actor(obs_all[None]).numpy()[0][:3]
            mv+=np.random.randn(3)*sigma
            acts.append((np.clip(mv,-1,1)*3,0))   # sel=0 (모형 단순화)
        return acts

    def train(self):
        if len(self.mem)<BATCH: return
        batch=random.sample(self.mem,BATCH)
        S,A,R,S2=map(np.array,zip(*batch))
        S=tf.convert_to_tensor(S); S2=tf.convert_to_tensor(S2)
        A=tf.convert_to_tensor(A); R=tf.convert_to_tensor(R[:,None],tf.float32)
        A_flat=tf.reshape(A,[BATCH,-1])
        with tf.GradientTape() as t:
            y=R+GAMMA*self.t_critic(tf.concat([S2,tf.stop_gradient(A_flat)],1))
            q=self.critic(tf.concat([S,A_flat],1))
            loss=tf.keras.losses.MSE(y,q)
        grads=t.gradient(loss,self.critic.trainable_weights)
        self.c_opt.apply_gradients(zip(grads,self.critic.trainable_weights))
        with tf.GradientTape() as t:
            A_pred=self.actor(S)
            A_pred_flat=tf.tile(A_pred,[1,N_UAV])  # 공유
            q=self.critic(tf.concat([S,A_pred_flat],1))
            aloss=-tf.reduce_mean(q)
        grads=t.gradient(aloss,self.actor.trainable_weights)
        self.a_opt.apply_gradients(zip(grads,self.actor.trainable_weights))
        self._sync(TAU)

# ───── train & log ─────
def run():
    env=UAVEnv()
    obs_dim=env.observation_space.shape[0]; act_dim=3
    ag=MADDPG(obs_dim,act_dim)
    ret_hist,inn_hist,unc_hist,fair_hist=[],[],[],[]
    for ep in range(EPIS):
        obs=env.reset(); ep_r=inn=unc=fair=0; t=0
        while True:
            unc_now=np.mean([np.trace(u.kf.P) for u in env.users])
            acts=ag.act(obs,unc_now)
            obs2,rews,done,info=env.step(acts)
            a_arr=np.array([a[0] for a in acts]).flatten()
            ag.remember(obs,a_arr,np.mean(rews),obs2)
            ag.train()
            obs=obs2; ep_r+=np.mean(rews); inn+=info['inn']; unc+=info['unc']; fair+=info['fair']; t+=1
            if done: break
        ret_hist.append(ep_r); inn_hist.append(inn/t); unc_hist.append(unc/t); fair_hist.append(fair/t)
        if (ep+1)%10==0:
            print(f"EP{ep+1:03d}  R {ep_r:8.2f}  In {inn/t:.3f}  Un {unc/t:.3f}  Fa {fair/t:.3f}")
    # plot
    plt.plot(ret_hist); plt.title("Return"); plt.savefig(os.path.join(SAVE_DIR,"ret.png")); plt.close()

if __name__=="__main__":
    run()
