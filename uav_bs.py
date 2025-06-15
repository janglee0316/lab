# Hybrid UAV Placement & Scheduling Simulator (Reward Shaping, Exploration 강화)
from __future__ import annotations
import numpy as np, math, random, copy, os
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List

import gym
from gym import spaces
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from scipy.optimize import root_scalar
from filterpy.kalman import KalmanFilter

SEED = 2025
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

AREA_X, AREA_Y       = 500.0, 500.0        # m
UAV_Z_MIN, UAV_Z_MAX = 30.0, 150.0         # m
F_CARRIER            = 2.0e9               # Hz
BANDWIDTH            = 30e3                # Hz
NOISE_PSD            = -174                # dBm/Hz
TX_POWER_mW          = 100.0               # mW baseline gear (modifiable)
EPISODES, STEPS      = 500, 200
N_UAV, USERS_PER_UAV = 3, 2
N_USERS              = N_UAV*USERS_PER_UAV

A_LOS, B_LOS = 1, 20
A_ENV = {"urban":(9.61,0.16)}             # (a,b) parameters
ETA_LOS, ETA_NLOS = 1.0, 20.0              # additional losses (dB)

ACTOR_LR, CRITIC_LR = 1e-4, 1e-3
GAMMA, TAU          = 0.95, 0.01
MEM_CAP, BATCH      = 50000, 256

SAVE_DIR = "./logs_hybrid"
os.makedirs(SAVE_DIR, exist_ok=True)

def optimal_altitude(path_loss_th: float, env: str = "urban") -> float:
    a, b = A_ENV[env]
    A = ETA_LOS - ETA_NLOS
    B = 20*math.log10(4*math.pi*F_CARRIER/3e8) + ETA_NLOS
    def f(theta: float):
        return (math.pi/(9*math.log(10)))*math.tan(theta) + a*b*A*math.exp(-b*(180/math.pi*theta - a))/(a*math.exp(-b*(180/math.pi*theta - a))+1)**2
    sol = root_scalar(f, bracket=[0.1, math.pi/2-1e-3], method='brentq')
    theta_opt = sol.root
    def g(R):
        return A/(1 + a*math.exp(-b*(180/math.pi*theta_opt - a))) + 20*math.log10(R/math.cos(theta_opt)) + B - path_loss_th
    R_opt = root_scalar(g, bracket=[1.0, 5000], method='bisect').root
    h_opt = R_opt*math.tan(theta_opt)
    return np.clip(h_opt, UAV_Z_MIN, UAV_Z_MAX)

@dataclass
class User:
    role: str
    x: float; y: float; z: float = 0.0
    demand: float = 1.0
    urgency: float = 1.0
    kf: KalmanFilter | None = None
    def state_vec(self):
        return np.array([self.x, self.y, self.demand, self.urgency])

class UAVEnv(gym.Env):
    metadata = {"render.modes": []}
    def __init__(self):
        super().__init__()
        self.n_uav, self.n_users = N_UAV, N_USERS
        self.users : List[User] = []
        self.uav_pos = np.zeros((self.n_uav,3))
        self.step_cnt, self.ep_cnt = 0,0

        self.action_space = []
        for _ in range(self.n_uav):
            move = spaces.Box(low=np.array([-3,-3,-3]), high=np.array([3,3,3]), dtype=np.float32)
            select = spaces.Discrete(self.n_users+1)
            self.action_space.append(spaces.Tuple((move,select)))
        obs_dim = 3*self.n_uav + 4*self.n_users
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.step_cnt+=1
        base_x, base_y = 50.0, 250.0
        spacing = 30.0
        self.users = []
        for i in range(self.n_users):
            xi = base_x + i*spacing + np.random.randn()*2
            yi = base_y + np.random.randn()*5
            role = ["cmd","relay","isr","troop","troop","logi"][i%6]
            u = User(role,xi,yi)
            u.kf = self._init_kf(xi,yi)
            self.users.append(u)
        us = np.array([[u.x,u.y] for u in self.users])
        centroid = us.mean(axis=0)
        self.uav_pos[:,:2] = centroid + np.random.randn(self.n_uav,2)*10
        opt_h = optimal_altitude(100)
        self.uav_pos[:,2] = opt_h
        return self._get_obs()

    def _init_kf(self,x,y):
        k = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0
        k.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        k.H = np.array([[1,0,0,0],[0,1,0,0]])
        k.Q = np.diag([.05,.05,.01,.01])
        k.R = np.diag([1.,1.])
        k.x = np.array([x,y,0.5,0.0])
        return k

    def step(self, actions: List[Tuple[np.ndarray,int]]):
        assert len(actions)==self.n_uav
        served_flags = np.zeros(self.n_users)
        for i,(move,sel) in enumerate(actions):
            self.uav_pos[i]+=move
            self.uav_pos[i,0]=np.clip(self.uav_pos[i,0],0,AREA_X)
            self.uav_pos[i,1]=np.clip(self.uav_pos[i,1],0,AREA_Y)
            self.uav_pos[i,2]=np.clip(self.uav_pos[i,2],UAV_Z_MIN,UAV_Z_MAX)
            if sel<self.n_users:
                served_flags[sel]=1
        heading = np.deg2rad(60)
        v=1.5
        for u in self.users:
            u.x += v*math.cos(heading) + np.random.randn()*0.3
            u.y += v*math.sin(heading) + np.random.randn()*0.3
            u.kf.predict(); z=np.array([u.x,u.y]); u.kf.update(z)
        rates = self._compute_rates()
        sum_tp = rates.sum(); worst = rates.min()
        fairness = (np.sum(rates)**2)/(len(rates)*np.sum(rates**2)+1e-6)
        penalty = np.maximum(0,0.1-fairness)*10
        base_reward = sum_tp/1e5 - penalty

        # --- Kalman-based reward shaping ---
        innovations = []
        uncertainties = []
        for u in self.users:
            # innovation: 오차, uncertainty: 추정 공분산(trace)
            innovations.append(np.linalg.norm(u.kf.y) if hasattr(u.kf, 'y') else 0)
            uncertainties.append(np.trace(u.kf.P) if hasattr(u.kf, 'P') else 0)
        avg_innovation = np.mean(innovations)
        avg_uncertainty = np.mean(uncertainties)
        # 예측 오차 작을수록 보상, 불확실성 클수록 penalty
        reward = base_reward + 0.1/(avg_innovation+1e-6) - 0.02*avg_uncertainty

        self.ep_cnt += 1
        done = self.ep_cnt>=STEPS
        # info dict에 innovation/uncertainty 기록해 결과도 추후 시각화 가능
        info = {'innovation': avg_innovation, 'uncertainty': avg_uncertainty, 'fairness': fairness}
        return self._get_obs(), [reward]*self.n_uav, done, info

    def _compute_rates(self):
        rates = np.zeros(self.n_users)
        noise = 10**((NOISE_PSD+10*np.log10(BANDWIDTH))/10)/1000
        for idx,u in enumerate(self.users):
            dists = np.linalg.norm(self.uav_pos[:,:2]-np.array([u.x,u.y]),axis=1)
            best = np.argmin(dists)
            pathloss_dB = self._pathloss(dists[best], self.uav_pos[best,2])
            snr = TX_POWER_mW*10**(-pathloss_dB/10)/(noise)
            rates[idx] = BANDWIDTH*np.log2(1+snr)
        return rates

    def _pathloss(self, r, h):
        a,b = A_ENV["urban"]
        theta = math.atan(h/r)
        P_los = 1/(1+a*math.exp(-b*( 180/math.pi*theta - a)))
        d3d = math.sqrt(h**2 + r**2)
        L_los = 30.9 + (22.25-0.5*math.log10(h))*math.log10(d3d) + 20*math.log10(F_CARRIER/1e9)
        L_nlos = max(L_los, 32.4 + (43.2-7.6*math.log10(h))*math.log10(d3d) + 20*math.log10(F_CARRIER/1e9))
        return P_los*L_los + (1-P_los)*L_nlos

    def _get_obs(self):
        uav_flat = self.uav_pos.flatten()
        usr_flat = np.concatenate([u.state_vec() for u in self.users])
        return np.concatenate([uav_flat, usr_flat]).astype(np.float32)

class OUActionNoise:
    def __init__(self, mu, sigma=0.1):
        self.mu, self.sigma, self.state = mu, sigma, np.zeros_like(mu)
    def __call__(self):
        dx = self.sigma*np.random.randn(*self.mu.shape)
        self.state += -self.state*0.15 + dx
        return self.state

def build_actor(input_dim, act_move_dim, act_sel_dim):
    inp = layers.Input(shape=(input_dim,))
    x  = layers.Dense(128,activation='relu')(inp)
    x  = layers.Dense(128,activation='relu')(x)
    move = layers.Dense(act_move_dim,activation='tanh')(x)
    sel  = layers.Dense(act_sel_dim,activation='softmax')(x)
    return models.Model(inp, [move,sel])

def build_critic(obs_dim, act_dim):
    inp_obs = layers.Input(shape=(obs_dim,))
    inp_act = layers.Input(shape=(act_dim,))
    x = layers.Concatenate()([inp_obs, inp_act])
    x = layers.Dense(256,activation='relu')(x)
    x = layers.Dense(256,activation='relu')(x)
    out = layers.Dense(1)(x)
    return models.Model([inp_obs,inp_act], out)

class MADDPG:
    def __init__(self, env: UAVEnv):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_move_dim = 3
        self.act_sel_dim  = env.n_users+1
        self.act_dim = self.act_move_dim + self.act_sel_dim
        self.actor = build_actor(self.obs_dim, self.act_move_dim,self.act_sel_dim)
        self.actor_target = build_actor(self.obs_dim, self.act_move_dim,self.act_sel_dim)
        self.critic = build_critic(self.obs_dim, self.act_dim)
        self.critic_target = build_critic(self.obs_dim, self.act_dim)
        self.actor_opt  = optimizers.Adam(ACTOR_LR)
        self.critic_opt = optimizers.Adam(CRITIC_LR)
        self.buf_s = deque(maxlen=MEM_CAP)
        self.buf_a = deque(maxlen=MEM_CAP)
        self.buf_r = deque(maxlen=MEM_CAP)
        self.buf_s2= deque(maxlen=MEM_CAP)
        # Uncertainty-driven exploration: noise scale adaptively increases with uncertainty
        self.noise = OUActionNoise(np.zeros(self.act_move_dim))
        self._update_target(1.0)
        self.explore_sigma = 0.1   # base sigma for OU noise

    def _update_target(self,tau):
        new = self.actor.get_weights(); tgt=self.actor_target.get_weights()
        self.actor_target.set_weights([tau*n+(1-tau)*o for n,o in zip(new,tgt)])
        new = self.critic.get_weights(); tgt=self.critic_target.get_weights()
        self.critic_target.set_weights([tau*n+(1-tau)*o for n,o in zip(new,tgt)])

    def select_action(self, obs, explore=True, uncertainty=0.0):
        move, sel = self.actor(obs[None])[0].numpy()[0], self.actor(obs[None])[1].numpy()[0]
        if explore:
            # Uncertainty-driven exploration: 더 불확실할수록 노이즈 up
            sigma = min(self.explore_sigma + 0.3*uncertainty, 1.0)
            self.noise.sigma = sigma
            move += self.noise()
            sel_idx = np.random.choice(len(sel), p=sel)
        else:
            sel_idx = int(sel.argmax())
        return np.clip(move,-1,1)*3, sel_idx

    def store(self,s,a,r,s2):
        self.buf_s.append(s); self.buf_a.append(a); self.buf_r.append(r); self.buf_s2.append(s2)

    def train(self):
        if len(self.buf_s)<BATCH: return
        idx = np.random.choice(len(self.buf_s),BATCH,replace=False)
        S  = np.stack([self.buf_s[i] for i in idx])
        A  = np.stack([self.buf_a[i] for i in idx])
        R  = np.stack([self.buf_r[i] for i in idx]).reshape(-1,1)
        S2 = np.stack([self.buf_s2[i] for i in idx])

        S = tf.convert_to_tensor(S, dtype=tf.float32)
        A = tf.convert_to_tensor(A, dtype=tf.float32)
        R = tf.convert_to_tensor(R, dtype=tf.float32)
        S2 = tf.convert_to_tensor(S2, dtype=tf.float32)

        with tf.GradientTape() as tape:
            a2_move,a2_sel = self.actor_target(S2)
            a2 = tf.concat([a2_move, a2_sel],axis=1)
            q2 = self.critic_target([S2,a2])
            y  = R + GAMMA*q2
            q  = self.critic([S,A])
            loss = tf.keras.losses.MSE(y,q)
        grads = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_weights))
        with tf.GradientTape() as tape:
            a_move,a_sel = self.actor(S)
            a = tf.concat([a_move,a_sel],axis=1)
            q = self.critic([S,a])
            loss_a = -tf.reduce_mean(q)
        grads = tape.gradient(loss_a, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_weights))
        self._update_target(TAU)

def plot_learning_curve(returns, innovations, uncertainties, fairnesses, window=20):
    import matplotlib.pyplot as plt
    N = len(returns)
    avg = np.convolve(returns, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1)
    plt.plot(range(1, N+1), returns, label='Episode Return', alpha=0.5)
    plt.plot(range(window, N+1), avg, label=f'Moving Avg({window})', linewidth=2)
    plt.xlabel('Episode'); plt.ylabel('Return'); plt.legend(); plt.grid(True)
    plt.title('Learning Curve: Total Episode Return')
    plt.subplot(2,2,2)
    plt.plot(innovations, label='Avg Innovation')
    plt.xlabel('Episode'); plt.ylabel('Innovation'); plt.title('Kalman Innovation'); plt.grid(True)
    plt.subplot(2,2,3)
    plt.plot(uncertainties, label='Avg Uncertainty', color='orange')
    plt.xlabel('Episode'); plt.ylabel('Uncertainty'); plt.title('Kalman Uncertainty'); plt.grid(True)
    plt.subplot(2,2,4)
    plt.plot(fairnesses, label='Fairness', color='green')
    plt.xlabel('Episode'); plt.ylabel('Fairness'); plt.title('Jain\'s Fairness Index'); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "learning_curve.png"))
    plt.close()


# UAV trajectory plot
def plot_trajectories(uav_logs, user_logs):
    for uav_id, traj in enumerate(uav_logs):
        xs, ys = zip(*[(p[0], p[1]) for p in traj])
        plt.plot(xs, ys, label=f'UAV{uav_id}')
    for user_id, traj in enumerate(user_logs):
        xs, ys = zip(*[(p[0], p[1]) for p in traj])
        plt.plot(xs, ys, 'k--', alpha=0.5, label=f'User{user_id}')
    plt.legend(); plt.title("UAV & User Trajectories"); plt.xlabel('x'); plt.ylabel('y'); plt.show()


# coverage heatmap
def plot_coverage_heatmap(uav_logs, area_x, area_y, bins=50):
    all_pos = np.vstack([np.array(log)[:,:2] for log in uav_logs])
    heat, xedges, yedges = np.histogram2d(all_pos[:,0], all_pos[:,1], bins=bins, range=[[0,area_x],[0,area_y]])
    plt.imshow(heat.T, origin='lower', extent=[0, area_x, 0, area_y], cmap='hot')
    plt.colorbar(label='UAV Visit Frequency')
    plt.title('UAV Coverage Heatmap'); plt.xlabel('x'); plt.ylabel('y')
    plt.show()
    

#QoS distribution
def plot_qos_boxplot(qos_by_user):
    plt.boxplot(qos_by_user)
    plt.xlabel('User'); plt.ylabel('QoS/Throughput')
    plt.title('User QoS Distribution'); plt.show()

    
def run_training():
    env = UAVEnv()
    agent = MADDPG(env)
    returns, innovations, uncertainties, fairnesses = [], [], [], []
    for ep in range(EPISODES):
        obs = env.reset()
        ep_ret = 0.0
        ep_innov = []
        ep_uncert = []
        ep_fair = []
        for t in range(STEPS):
            # uncertainty-driven exploration: 가장 최근 uncertainty 사용
            cur_uncert = np.mean([np.trace(u.kf.P) for u in env.users]) if env.users else 0
            move, sel = agent.select_action(obs, uncertainty=cur_uncert)
            act = [(move, sel) for _ in range(env.n_uav)]
            obs2, rews, done, info = env.step(act)
            r = np.mean(rews)
            a_cat = np.concatenate([move, tf.one_hot(sel, env.n_users+1).numpy()])
            agent.store(obs, a_cat, r, obs2)
            agent.train()
            obs = obs2; ep_ret += r
            if 'innovation' in info:
                ep_innov.append(info['innovation'])
                ep_uncert.append(info['uncertainty'])
                ep_fair.append(info['fairness'])
            if done: break
        returns.append(ep_ret)
        innovations.append(np.mean(ep_innov))
        uncertainties.append(np.mean(ep_uncert))
        fairnesses.append(np.mean(ep_fair))
        if (ep + 1) % 10 == 0:
            print(f"[EP {ep + 1}] return={ep_ret:.3f}, innovation={np.mean(ep_innov):.4f}, uncertainty={np.mean(ep_uncert):.4f}, fairness={np.mean(ep_fair):.4f}")
    np.save(os.path.join(SAVE_DIR, "returns.npy"), np.array(returns))
    plot_learning_curve(returns, innovations, uncertainties, fairnesses, window=20)

if __name__=="__main__":
    run_training()
