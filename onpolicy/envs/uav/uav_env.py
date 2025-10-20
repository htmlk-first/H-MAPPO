import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from gym import spaces # gym.spaces를 사용하도록 명시
from sklearn.cluster import KMeans

# entities.py로부터 모든 클래스를 import 합니다.
from .entities import UAV, Jammer, PointOfInterest

class UAVEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, args):
        # 1. 시뮬레이션 기본 파라미터 설정 (이전과 동일)
        self.num_uavs = args.num_uavs
        self.num_pois = args.num_pois
        self.num_jammers = args.num_jammers
        self.world_size = args.world_size
        self.fly_th = args.fly_th
        self.time_step = 0
        self.episode_len = args.episode_length

        # 2. 객체 생성 (이전과 동일)
        self.uavs = [UAV(start_x=np.random.rand() * self.world_size,
                         start_y=np.random.rand() * self.world_size,
                         start_z=self.fly_th, v_max=args.v_max, u_max=args.u_max)
                     for _ in range(self.num_uavs)]
        self.jammers = [Jammer(x=np.random.rand() * self.world_size,
                               y=np.random.rand() * self.world_size,
                               radius=args.jammer_radius, power=args.jammer_power)
                        for _ in range(self.num_jammers)]
        self.pois = [PointOfInterest(x=np.random.rand() * self.world_size,
                                     y=np.random.rand() * self.world_size)
                     for _ in range(self.num_pois)]

        # 3. H-MAPPO 관련 변수 (이전과 동일)
        self.num_clusters = self.num_uavs
        self.cluster_centers = np.zeros((self.num_clusters, 2))
        self.uav_goals = np.zeros((self.num_uavs, 2))

        # 4. 통신 및 에너지 모델 파라미터 (이전과 동일)
        self.sinr_threshold = 5
        self.noise_power = 1e-9
        self.energy_prop_coeff = 0.1
        self.energy_comm_trad = 0.01
        self.energy_comm_sem = 0.05

        # 5. [핵심 수정] 표준 gym.spaces를 사용하여 계층적 관측/행동 공간 정의
        # --- 상위 레벨 공간 정의 ---
        high_obs_dim = (self.num_pois * 2) + (self.num_uavs * 3)
        high_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(high_obs_dim,), dtype=np.float32)
        high_act_space = spaces.MultiDiscrete([self.num_clusters] * self.num_uavs)

        # --- 하위 레벨 공간 정의 ---
        low_obs_dim = 4 + (self.num_uavs - 1) * 2 + 2
        low_act_space = spaces.MultiDiscrete([9, 3, 2, 3])
        
        # --- 전체 공간을 spaces.Dict로 통합 ---
        self.action_space = spaces.Dict({
            'high_level': spaces.Tuple([high_act_space]),
            'low_level': spaces.Tuple([low_act_space for _ in range(self.num_uavs)])
        })
        self.observation_space = spaces.Dict({
            'high_level': spaces.Tuple([high_obs_space]),
            'low_level': spaces.Tuple([spaces.Box(low=-np.inf, high=np.inf, shape=(low_obs_dim,), dtype=np.float32) for _ in range(self.num_uavs)])
        })

        # --- 공유 관측 공간 정의 ---
        share_low_obs_dim = low_obs_dim * self.num_uavs
        share_low_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(share_low_obs_dim,), dtype=np.float32)
        self.share_observation_space = spaces.Dict({
            'high_level': spaces.Tuple([high_obs_space]),
            'low_level': spaces.Tuple([share_low_obs_space for _ in range(self.num_uavs)])
        })
        
        # 6. 시각화 설정 (이전과 동일)
        self.fig, self.ax = None, None
        self.uav_plots = [None] * self.num_uavs
        self.uav_trajectories = [[] for _ in range(self.num_uavs)]
        self.poi_plots = [None] * self.num_pois
        self.jammer_plots = [None] * self.num_jammers
        self.goal_plots = [None] * self.num_uavs

    def seed(self, seed=None):
        if seed is None: np.random.seed(1)
        else: np.random.seed(seed)
            
    def _get_full_obs(self):
        low_level_obs = self._get_low_level_obs()
        high_level_obs = self._get_high_level_obs()
        
        # [수정] observation_space에 맞는 딕셔너리 구조로 반환
        return {
            'high_level': high_level_obs,
            'low_level': low_level_obs
        }

    def reset(self):
        self.time_step = 0
        for i in range(self.num_uavs):
            self.uavs[i].pos = np.array([np.random.rand() * self.world_size, np.random.rand() * self.world_size, self.fly_th], dtype=np.float32)
            self.uavs[i].energy = self.uavs[i].max_energy
            self.uav_trajectories[i] = [self.uavs[i].pos[:2].copy()]
        for poi in self.pois:
            poi.visited = False
            poi.aoi = 0
            poi.pos = np.array([np.random.rand() * self.world_size, np.random.rand() * self.world_size])

        self._update_clusters()
        initial_goals = np.array([np.argmin([np.linalg.norm(uav.pos[:2] - cc) for cc in self.cluster_centers]) for uav in self.uavs])
        self.set_goals(initial_goals)
        
        return self._get_full_obs(), {}

    def step(self, low_level_actions):
        # ... (step 로직은 이전과 동일, 맨 아래 반환 부분만 변경) ...
        self.time_step += 1
        low_level_rewards = np.zeros(self.num_uavs)
        
        for i, uav in enumerate(self.uavs):
            move_dir_idx, speed_idx, comm_mode, sem_level = low_level_actions[i]
            dist_to_goal_before = np.linalg.norm(uav.pos[:2] - self.uav_goals[i])
            self._uav_move(uav, move_dir_idx, speed_idx)
            dist_to_goal_after = np.linalg.norm(uav.pos[:2] - self.uav_goals[i])
            prop_energy = self.energy_prop_coeff * (self.uavs[0].v_max * (speed_idx + 1) / 3)**2
            uav.energy -= prop_energy
            sinr = self._calculate_sinr(uav)
            comm_energy = 0
            if comm_mode == 0:
                comm_energy = self.energy_comm_trad
                if sinr < self.sinr_threshold: low_level_rewards[i] -= 2
            else:
                comm_energy = self.energy_comm_sem * (1 + sem_level * 0.5)
                if sinr < self.sinr_threshold: low_level_rewards[i] += 1
            uav.energy -= comm_energy
            low_level_rewards[i] += (dist_to_goal_before - dist_to_goal_after) * 0.5
            low_level_rewards[i] -= (prop_energy + comm_energy) * 0.1

        high_level_reward = 0.0
        for poi in self.pois:
            if not poi.visited:
                for uav in self.uavs:
                    if np.linalg.norm(uav.pos[:2] - poi.pos) < 5:
                        poi.visited = True
                        high_level_reward += 20
                        break
        
        done = (self.time_step >= self.episode_len) or all(p.visited for p in self.pois)
        if all(p.visited for p in self.pois):
            high_level_reward += 100
        
        rewards = {'low_level': low_level_rewards, 'high_level': high_level_reward}
        infos = {}
        
        return self._get_full_obs(), rewards, done, infos

    # ... (이하 _uav_move, _calculate_sinr, _get_low_level_obs, _get_high_level_obs,
    #  _update_clusters, set_goals, render, close 함수는 이전과 동일하게 유지) ...
    def _uav_move(self, uav, dir_idx, speed_idx):
        if dir_idx == 8: return
        angle = (dir_idx / 8) * 2 * np.pi
        speed = uav.v_max * (speed_idx + 1) / 3
        uav.pos[0] += speed * np.cos(angle)
        uav.pos[1] += speed * np.sin(angle)
        uav.pos[0] = np.clip(uav.pos[0], 0, self.world_size)
        uav.pos[1] = np.clip(uav.pos[1], 0, self.world_size)
        self.uav_trajectories[self.uavs.index(uav)].append(uav.pos[:2].copy())

    def _calculate_sinr(self, uav):
        dist_to_gbs = np.linalg.norm(uav.pos)
        path_loss = 20 * np.log10(dist_to_gbs) + 20 * np.log10(4 * np.pi * 2.4e9 / 3e8) if dist_to_gbs > 0 else 0
        jammer_interference = sum(j.power / (np.linalg.norm(uav.pos[:2] - j.pos)**2 + 1e-9) for j in self.jammers if np.linalg.norm(uav.pos[:2] - j.pos) < j.radius)
        signal_power = uav.transmit_power / (10**(path_loss/10)) if path_loss > 0 else uav.transmit_power
        sinr = signal_power / (self.noise_power + jammer_interference) if (self.noise_power + jammer_interference) > 0 else signal_power
        return 10 * np.log10(sinr) if sinr > 0 else -100

    def _get_low_level_obs(self):
        obs_list = []
        for i, uav in enumerate(self.uavs):
            self_obs = [
                uav.pos[0] / self.world_size,
                uav.pos[1] / self.world_size,
                uav.energy / uav.max_energy,
                self._calculate_sinr(uav) / 20.0
            ]
            other_uav_obs = []
            for j, other_uav in enumerate(self.uavs):
                if i == j: continue
                other_uav_obs.extend((other_uav.pos[:2] - uav.pos[:2]) / self.world_size)
            goal_obs = (self.uav_goals[i] - uav.pos[:2]) / self.world_size
            full_obs = np.concatenate([self_obs, other_uav_obs, goal_obs])
            obs_list.append(full_obs)
        return obs_list

    def _get_high_level_obs(self):
        poi_info = np.array([[p.pos[0]/self.world_size, p.pos[1]/self.world_size] for p in self.pois]).flatten()
        uav_info = np.array([[u.pos[0]/self.world_size, u.pos[1]/self.world_size, u.energy/u.max_energy] for u in self.uavs]).flatten()
        if len(poi_info) < self.num_pois * 2:
            poi_info = np.pad(poi_info, (0, self.num_pois*2 - len(poi_info)), 'constant')
        return [np.concatenate([poi_info, uav_info])]

    def _update_clusters(self):
        poi_positions = np.array([p.pos for p in self.pois if not p.visited])
        if len(poi_positions) < self.num_clusters:
            self.cluster_centers = np.array([poi.pos for poi in self.pois if not poi.visited] + 
                                            [[self.world_size/2, self.world_size/2]] * (self.num_clusters - len(poi_positions)))
            return
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10, random_state=0).fit(poi_positions)
        self.cluster_centers = kmeans.cluster_centers_

    def set_goals(self, high_level_actions):
        actions = np.array(high_level_actions).flatten()
        if self.cluster_centers is not None and len(self.cluster_centers) > 0:
            self.uav_goals = np.array([self.cluster_centers[min(act, len(self.cluster_centers)-1)] for act in actions])

    def render(self, mode='human'):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(0, self.world_size)
            self.ax.set_ylim(0, self.world_size)
            self.ax.set_aspect('equal')
            
            for i in range(self.num_uavs):
                self.uav_plots[i] = self.ax.add_patch(Circle((0, 0), 2, color=f'C{i}', label=f'UAV {i}'))
                self.goal_plots[i] = self.ax.plot([], [], 'x', color=f'C{i}', markersize=10)[0]
            for i in range(self.num_pois):
                self.poi_plots[i] = self.ax.add_patch(Rectangle((0, 0), 3, 3, color='green', alpha=0.6))
            for i in range(self.num_jammers):
                self.jammer_plots[i] = self.ax.add_patch(Circle((0, 0), self.jammers[i].radius, color='red', alpha=0.3))
            self.ax.legend()

        for i, uav in enumerate(self.uavs):
            self.uav_plots[i].center = uav.pos[:2]
            if len(self.uav_trajectories[i]) > 1:
                traj = np.array(self.uav_trajectories[i])
                self.ax.plot(traj[:, 0], traj[:, 1], '-', color=f'C{i}', alpha=0.3)

        for i, poi in enumerate(self.pois):
            self.poi_plots[i].set_xy(poi.pos - 1.5)
            self.poi_plots[i].set_visible(not poi.visited)

        for i, jammer in enumerate(self.jammers):
            self.jammer_plots[i].center = jammer.pos

        if self.uav_goals is not None:
            for i, goal in enumerate(self.uav_goals):
                self.goal_plots[i].set_data(goal[0], goal[1])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None