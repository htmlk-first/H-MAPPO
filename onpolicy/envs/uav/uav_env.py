import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from gym import spaces
from sklearn.cluster import KMeans

from .entities import UAV, Jammer, PointOfInterest
from . import parameters as params

class UAVEnv(gym.Env):
    """
    Custom Gym environment for multi-UAV operations with hierarchical control (H-MAPPO).
    
    This environment defines two levels of control:
    1.  High-Level Policy: Observes the global state (all UAVs, all POIs) and
        outputs a goal (cluster index) for each UAV.
    2.  Low-Level Policy: Each UAV observes its local state (self, other UAVs,
        goal) and outputs movement and communication actions.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, args):
        """
        Initialize the UAV environment.
        :param args: (argparse.Namespace) Configuration object with parameters
                     like num_uavs, num_pois, world_size, etc.
        """
        
        # 1. Basic simulation parameters
        self.num_uavs = args.num_uavs
        self.num_pois = args.num_pois
        self.num_jammers = args.num_jammers
        self.world_size = args.world_size
        self.fly_th = args.fly_th   # Flying altitude threshold/default
        self.time_step = 0
        self.episode_len = args.episode_length
        
        self.gbs_pos = np.array([self.world_size / 2, self.world_size / 2, 0]) # 기지국(GBS) 위치 (맵 중앙, 지면)
        self.fc = params.FC     # Carrier frequency (2.4 GHz)
        self.c = params.C       # Speed of light
        self.a_plos = params.LOS_A   
        self.b_plos = params.LOS_B   
        self.eta_los = params.ETA_LOS   
        self.eta_nlos = params.ETA_NLOS 

        # 2. Create simulation entities (Agents, Targets, Obstacles)
        self.uavs = [UAV(start_x=np.random.rand() * self.world_size,
                         start_y=np.random.rand() * self.world_size,
                         start_z=self.fly_th, v_max=args.v_max, u_max=args.u_max)
                     for _ in range(self.num_uavs)]
        # Low-level state에 p_los를 추가하기 위해 uav 객체에 p_los 변수 초기화
        for uav in self.uavs:
            uav.p_los = 0.0 # p_los (LoS/NLoS 상태)
        
        self.jammers = [Jammer(x=np.random.rand() * self.world_size,
                               y=np.random.rand() * self.world_size,
                               radius=args.jammer_radius, power=args.jammer_power)
                        for _ in range(self.num_jammers)]
        
        self.pois = [PointOfInterest(x=np.random.rand() * self.world_size,
                                     y=np.random.rand() * self.world_size)
                     for _ in range(self.num_pois)]

        # 3. H-MAPPO related variables
        self.num_clusters = self.num_uavs   # Number of clusters for POIs
        self.cluster_centers = np.zeros((self.num_clusters, 2)) # 2D coordinates of cluster centers
        self.uav_goals = np.zeros((self.num_uavs, 2))   # 2D coordinates of the goal for each UAV
        
        # set_goals 로직 변경을 위해 POI의 클러스터 할당 레이블과 인덱스 저장 변수 추가
        self.poi_cluster_labels = None
        self.unvisited_poi_indices = None

        # 4. Communication and energy model parameters
        self.sinr_threshold = params.SINR_THRESHOLD = 5 # dB, threshold for successful communication
        self.noise_power = params.NOISE_DENSITY * params.BANDWIDTH # Noise power in SINR calculation
        self.energy_prop_coeff = params.ENERGY_PROP_COEFF # Coefficient for propulsion energy
        self.energy_comm_trad = params.ENERGY_COMM_TRAD # Energy cost for traditional communication
        self.energy_comm_sem = params.ENERGY_COMM_SEM # Energy cost for semantic communication
        self.time_delta = params.TIME_DELTA # 1 time step = 1 second 가정
        
        self.P_IDLE = params.P_IDLE       # W (Hovering power)
        self.U_TIP = params.U_TIP         # m/s (Tip speed of the rotor blade)
        self.V_0 = params.V_0          # m/s (Mean rotor induced velocity in hover)
        self.D_0 = params.D_0           # Fuselage drag ratio
        self.RHO = params.RHO         # Air density (kg/m^3)
        self.S_0 = params.S_0          # Rotor solidity
        self.A = params.A           # Rotor disc area (m^2)
        
        # Fidelity 모델 파라미터
        self.sem_base_quality = params.SEM_BASE_QUALITY # l=0, 1, 2 에 매핑
        self.sem_robust_coeff = params.SEM_ROBUST_COEFF  # k
        self.sem_robust_offset = params.SEM_ROBUST_OFFSET # S_offset
        self.r_cover = params.R_COVER # PoI 방문 기본 보상
        self.w_energy = params.W_ENERGY # 에너지 페널티 가중치

        # 5. Define hierarchical observation and action spaces using gym.spaces
        
        # --- High-Level Spaces (Global Policy, 1 Agent) ---
        # (목표 설계 반영) S_high: (POI 위치) + (POI 방문 상태) + (Jammer 위치) + (UAV 상태)
        high_obs_dim = (self.num_pois * 2) + (self.num_pois) + (self.num_jammers * 2) + (self.num_uavs * 4)
        high_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(high_obs_dim,), dtype=np.float32)
        
        # Action: Assign a cluster index [0, num_clusters-1] to each UAV
        high_act_space = spaces.MultiDiscrete([self.num_clusters] * self.num_uavs)

        # --- Low-Level Spaces (Local Policy, num_uavs Agents) ---
        # (목표 설계 반영) S_low: (self_pos, energy, sinr, p_los) + (relative_pos_of_other_UAVs) + (vector_to_goal)
        # self_obs_dim = 5 (p_los 추가)
        low_obs_dim = 5 + (self.num_uavs - 1) * 2 + 2
        
        # (목표 설계 반영) A_low: [연속 경로] + [이산 통신 모드] + [이산 압축 수준]
        # 기존 MultiDiscrete([9, 3, 2, 3]) -> 연속 Box(4,)로 변경
        # action[0]: v_x_scaled (-1 to 1)
        # action[1]: v_y_scaled (-1 to 1)
        # action[2]: comm_mode_signal (-1 to 1) -> 0 또는 1로 변환
        # action[3]: sem_level_signal (-1 to 1) -> 0, 1, 2로 변환
        low_act_space_per_agent = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # --- Combine into a gym.spaces.Dict ---
        # The H-MAPPO runner expects this dictionary structure.
        # Note: The spaces are wrapped in a Tuple([space]) to match the runner's expectations.
        self.action_space = spaces.Dict({
            'high_level': spaces.Tuple([high_act_space]),
            'low_level': spaces.Tuple([low_act_space_per_agent for _ in range(self.num_uavs)])
        })
        self.observation_space = spaces.Dict({
            'high_level': spaces.Tuple([high_obs_space]),
            'low_level': spaces.Tuple([spaces.Box(low=-np.inf, high=np.inf, shape=(low_obs_dim,), dtype=np.float32) for _ in range(self.num_uavs)])
        })

        # --- Define Shared Observation Spaces (for Centralized Critic) ---
        # Low-level share_obs is all low-level obs concatenated
        share_low_obs_dim = low_obs_dim * self.num_uavs
        share_low_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(share_low_obs_dim,), dtype=np.float32)
        # High-level share_obs is the same as the high-level obs (it's already global)
        self.share_observation_space = spaces.Dict({
            'high_level': spaces.Tuple([high_obs_space]),
            'low_level': spaces.Tuple([share_low_obs_space for _ in range(self.num_uavs)])
        })
        
        # 6. Visualization setup
        self.fig, self.ax = None, None
        self.uav_plots = [None] * self.num_uavs
        self.uav_trajectories = [[] for _ in range(self.num_uavs)]
        self.poi_plots = [None] * self.num_pois
        self.jammer_plots = [None] * self.num_jammers
        self.goal_plots = [None] * self.num_uavs

    def seed(self, seed=None):
        """Seeds the environment's random number generator."""
        if seed is None: np.random.seed(1)
        else: np.random.seed(seed)
            
    def _get_full_obs(self):
        """
        Helper function to gather observations from both levels.
        :return: (dict) A dictionary matching self.observation_space structure.
        """
        low_level_obs = self._get_low_level_obs()
        high_level_obs = self._get_high_level_obs()
        
        return {
            'high_level': high_level_obs,
            'low_level': low_level_obs
        }

    def reset(self):
        """
        Resets the environment to an initial state.
        :return: (dict, dict) The initial observation dictionary and an empty info dict.
        """
        self.time_step = 0
        
        # Reset UAV positions and energy
        for i in range(self.num_uavs):
            self.uavs[i].pos = np.array([np.random.rand() * self.world_size, np.random.rand() * self.world_size, self.fly_th], dtype=np.float32)
            self.uavs[i].energy = self.uavs[i].max_energy
            self.uav_trajectories[i] = [self.uavs[i].pos[:2].copy()]
            self.uavs[i].p_los = 0.0
        
        # Reset POI positions and visited status
        for poi in self.pois:
            poi.visited = False
            poi.aoi = 0
            poi.pos = np.array([np.random.rand() * self.world_size, np.random.rand() * self.world_size])

        # Update POI clusters for the high-level policy
        self._update_clusters()
        # Assign initial goals based on nearest cluster
        initial_goals = np.array([np.argmin([np.linalg.norm(uav.pos[:2] - cc) for cc in self.cluster_centers]) for uav in self.uavs])
        self.set_goals(initial_goals)
                
        # Return the first observation
        return self._get_full_obs(), {}

    def step(self, low_level_actions):
        """
        Executes one time step of the environment's dynamics.
        :param low_level_actions: (list) A list of low-level actions, one for each UAV.
                                 Each action is a MultiDiscrete array:
                                 [move_dir, speed, comm_mode, sem_level]
        :return: (dict, dict, bool, dict) 
                 obs (dict), rewards (dict), done (bool), infos (dict)
        """
        self.time_step += 1
        low_level_rewards = np.zeros(self.num_uavs)
        
        for i, uav in enumerate(self.uavs):
            
            # --- [GEM] MODIFICATION START ---
            # 연속적인 low_level_actions를 해석 및 이산화
            
            # 1. Unpack continuous actions
            move_action = low_level_actions[i][0:2]     # [v_x_signal, v_y_signal]
            comm_signal = low_level_actions[i][2]       # comm_mode_signal
            sem_level_signal = low_level_actions[i][3]  # sem_level_signal
            
            # 2. Discretize communication actions
            # comm_signal (-1 to 1) -> comm_mode (0 or 1)
            comm_mode = 0 if comm_signal < 0 else 1
            
            # 1. Normalize signal from [-1, 1] (or even outside) to [0, 1]
            sem_level_normalized = (sem_level_signal + 1) / 2.0
            # 2. Scale to [0, 3]
            sem_level_scaled = sem_level_normalized * 3
            # 3. Convert to integer (0, 1, 2, or 3+)
            sem_level_int = int(sem_level_scaled)
            # 4. Clip to the valid range [0, 2] to prevent index error
            sem_level = np.clip(sem_level_int, 0, 2)
                        
            dist_to_goal_before = np.linalg.norm(uav.pos[:2] - self.uav_goals[i])
            
            # 1. Apply movement
            # --- [GEM] MODIFICATION START ---
            # _uav_move는 이제 연속적인 move_action을 받고 실제 속도를 반환
            current_speed = self._uav_move(uav, move_action)
            # --- [GEM] MODIFICATION END ---
            dist_to_goal_after = np.linalg.norm(uav.pos[:2] - self.uav_goals[i])
            
            # 2. Calculate propulsion energy cost
            # --- [GEM] MODIFICATION START ---
            # (기존: speed_idx 기반)
            # prop_energy = self.energy_prop_coeff * (self.uavs[0].v_max * (speed_idx + 1) / 3)**2
            # (수정: _uav_move에서 반환된 실제 속도 기반)
            prop_power = self._calculate_propulsion_power(current_speed)
            prop_energy = prop_power * self.time_delta # Energy = Power * Time
            # --- [GEM] MODIFICATION END ---
            
            uav.energy -= prop_energy
            
            # 3. Calculate communication energy cost and SINR penalty/reward
            sinr = self._calculate_sinr(uav) # (uav.p_los가 이 함수 내부에서 업데이트됨)
            comm_energy = 0
            if comm_mode == 0:  # Traditional mode
                comm_energy = self.energy_comm_trad
            else:   # Semantic mode
                # --- [GEM] MODIFICATION --- (이산화된 sem_level 사용)
                comm_energy = self.energy_comm_sem * (1 + sem_level * 0.5)
            uav.energy -= comm_energy
            
            # 현재 Fidelity 계산 및 저장 ---
            # --- [GEM] MODIFICATION --- (이산화된 comm_mode, sem_level 사용)
            uav.current_fidelity = self._calculate_fidelity(comm_mode, sem_level, sinr)
            
            # 4. Calculate low-level reward
            # Reward for moving closer to the high-level goal (이제 이 goal은 POI임)
            low_level_rewards[i] += (dist_to_goal_before - dist_to_goal_after) * 0.5
            # Penalty for energy consumption
            low_level_rewards[i] -= (prop_energy + comm_energy) * self.w_energy

        # 5. Calculate high-level reward (global)
        high_level_reward = 0.0
        for poi in self.pois:
            if not poi.visited:
                for i, uav in enumerate(self.uavs):
                    # Check if any UAV is close enough to "visit" the POI
                    if np.linalg.norm(uav.pos[:2] - poi.pos) < 5:
                        poi.visited = True
                        
                        # --- 핵심 수정: Fidelity 기반 보상 ---
                        fidelity_reward = self.r_cover * uav.current_fidelity
                        high_level_reward += fidelity_reward
                        
                        # (선택) Low-level 에이전트에게도 이 보상을 일부 공유 (Credit Assignment)
                        low_level_rewards[i] += fidelity_reward * 0.1
                        
                        # --- [GEM] MODIFICATION START ---
                        # (목표 설계 로직 변경)
                        # POI를 방문했으므로, Low-level의 목표를 이 클러스터 내의 다음 POI로 갱신
                        # (set_goals는 high_level_action을 입력으로 받으므로, 현재 할당된 클러스터 인덱스 필요)
                        # (간단한 구현: 여기서는 갱신하지 않고, 다음 high-level step (set_goals)에서 갱신되도록 둠)
                        # (더 복잡한 구현: high_level_action을 저장해 뒀다가 여기서 set_goals(i)를 호출)
                        # (현재: low_level 보상에만 반영하고 goal 갱신은 high-level 주기에 맡김)
                        # --- [GEM] MODIFICATION END ---
                        break
        
        # 6. Check for episode termination
        done = (self.time_step >= self.episode_len) or all(p.visited for p in self.pois)
        if all(p.visited for p in self.pois):
            high_level_reward += 100    # Bonus reward for completing all POIs
        
        # 7. Package rewards into a dictionary for H-MAPPO
        rewards = {'low_level': low_level_rewards, 'high_level': high_level_reward}
        infos = {}
        
        # Return the new state, rewards, done flag, and info
        return self._get_full_obs(), rewards, done, infos

    def _uav_move(self, uav, move_action):
        """
        Moves the UAV based on a continuous action vector.
        :param uav: (UAV) The UAV object to move.
        :param move_action: (np.ndarray) Shape (2,), continuous vector [-1, 1]
                            representing [v_x_signal, v_y_signal].
        :return: (float) The actual speed (magnitude) of the UAV.
        """
        
        # 1. Scale action signals to velocity vector
        speed_vec = move_action * uav.v_max
        
        # 2. Check maximum speed constraint
        current_speed = np.linalg.norm(speed_vec)
        if current_speed > uav.v_max:
            # Rescale vector to match v_max if it exceeds it
            speed_vec = speed_vec * (uav.v_max / current_speed)
            current_speed = uav.v_max
            
        # 3. Apply movement (pos = pos + v * dt)
        uav.pos[0] += speed_vec[0] * self.time_delta
        uav.pos[1] += speed_vec[1] * self.time_delta
        
        # 4. Clip position to stay within world boundaries
        uav.pos[0] = np.clip(uav.pos[0], 0, self.world_size)
        uav.pos[1] = np.clip(uav.pos[1], 0, self.world_size)
        
        # 5. Store trajectory for rendering
        self.uav_trajectories[self.uavs.index(uav)].append(uav.pos[:2].copy())
        
        # 6. Return actual speed for energy calculation
        return current_speed

    def _calculate_sinr(self, uav):
        """
        Helper function to calculate the SINR for a UAV based on 3D A2G channel model.
        (Ref: [저서 Uav Communications for 5G and Beyond.pdf] Ch. 2)
        """
        
        # --- 1. 신호 경로 계산 (UAV to GBS) ---
        
        # 3D 거리 벡터 및 스칼라 거리 계산
        vec_to_gbs = self.gbs_pos - uav.pos
        d_3d = np.linalg.norm(vec_to_gbs)
        if d_3d < 1.0: d_3d = 1.0 # 1m 미만 클리핑

        # 2D (수평) 거리 계산
        r_2d = np.linalg.norm(vec_to_gbs[:2])
        
        # 고도각(Elevation angle) 계산 (degrees)
        elevation_rad = np.arcsin(uav.pos[2] / d_3d)
        elevation_deg = np.degrees(elevation_rad)
        if elevation_deg < 0: elevation_deg = 0
        
        # LoS 확률 계산 (Sigmoid)
        p_los = 1.0 / (1.0 + self.a_plos * np.exp(-self.b_plos * (elevation_deg - self.a_plos)))
        
        # S_low에 p_los를 제공하기 위해 uav 객체에 저장
        uav.p_los = p_los
        
        # 자유 공간 경로 손실 (FSPL) 계산 (dB)
        pl_fspl_db = 20 * np.log10(d_3d) + 20 * np.log10(self.fc) + 20 * np.log10(4 * np.pi / self.c)
        
        # LoS 및 NLoS 경로 손실 (dB)
        pl_los_db = pl_fspl_db + self.eta_los
        pl_nlos_db = pl_fspl_db + self.eta_nlos
                
        # 평균 경로 손실 (dB)
        avg_path_loss_db = p_los * pl_los_db + (1 - p_los) * pl_nlos_db
        
        # 수신 신호 전력 계산 (dBm)
        tx_power_dbm = 10 * np.log10(uav.transmit_power * 1000) # W to dBm
        signal_power_dbm = tx_power_dbm - avg_path_loss_db

        # --- 2. 간섭 경로 계산 (Jammers to UAV) ---
        jammer_interference_watts = 0
        for j in self.jammers:
            jammer_pos_3d = np.array([j.pos[0], j.pos[1], 0])
            dist_to_jammer = np.linalg.norm(uav.pos - jammer_pos_3d)
            
            if dist_to_jammer < j.radius:
                lambda_sq = (self.c / self.fc)**2
                fspl_linear = lambda_sq / ((4 * np.pi * dist_to_jammer)**2 + 1e-9)
                jammer_interference_watts += j.power * fspl_linear

        # --- 3. SINR 계산 (dB) ---
        
        # 잡음 + 간섭 전력 (Watts)
        noise_plus_interference_watts = self.noise_power + jammer_interference_watts
        # 수신 신호 전력 (Watts)
        signal_power_watts = 10**((signal_power_dbm - 30) / 10) # dBm to W
        # SINR (linear)
        sinr_linear = signal_power_watts / noise_plus_interference_watts
        
        # SINR (dB)
        return 10 * np.log10(sinr_linear) if sinr_linear > 0 else -100
    
    def _calculate_fidelity(self, comm_mode, sem_level, sinr):
        """
        Calculates the data fidelity based on comm mode, sem level, and SINR.
        Implements the mathematical formulation.
        """
        if comm_mode == 0: # Traditional Mode
            # 수식 (A) 구현: Cliff-effect
            return 1.0 if sinr >= self.sinr_threshold else 0.0
        else: # Semantic Mode
            # 1. Base Quality
            base_quality = self.sem_base_quality[sem_level]
            # 2. Robustness (Sigmoid)
            robustness = 1.0 / (1.0 + np.exp(-self.sem_robust_coeff * (sinr - self.sem_robust_offset)))
            
            return base_quality * robustness
    
    def _calculate_propulsion_power(self, V):
        """
        Calculates the propulsion power (in Watts) for a UAV flying at speed V.
        Based on the model in [저서 wireless-communications-and-networking-for-unmanned-aerial-vehicles.pdf]
        and parameters from parameters.py.
        """
        V_squared = V**2
        V_fourth = V**4
        
        # 1. Blade Profile Power
        P_blade = self.P_IDLE * (1 + (3 * V_squared) / (self.U_TIP**2))
        
        # 2. Induced Power
        v_0_fourth = self.V_0**4
        # Numerical stability for V=0
        sqrt_term_inner = 1 + V_fourth / (4 * v_0_fourth)
        if sqrt_term_inner < 0: sqrt_term_inner = 0
        sqrt_term = np.sqrt(np.sqrt(sqrt_term_inner) - V_squared / (2 * self.V_0**2))
        if np.isnan(sqrt_term): sqrt_term = 0 # V가 매우 클 때 음수가 될 수 있음
        P_induced = self.P_IDLE * sqrt_term
        
        # 3. Parasite Drag Power
        P_parasite = 0.5 * self.D_0 * self.RHO * self.S_0 * self.A * (V**3)
        
        return P_blade + P_induced + P_parasite
    
    def _get_low_level_obs(self):
        """
        Generates the local observation for each low-level agent (UAV).
        :return: (list) A list of observation arrays, one for each UAV.
        """
        
        obs_list = []
        for i, uav in enumerate(self.uavs):
            # 1. Self observation (normalized)
            current_sinr = self._calculate_sinr(uav) # (uav.p_los가 여기서 업데이트됨)
            self_obs = [
                uav.pos[0] / self.world_size,
                uav.pos[1] / self.world_size,
                uav.energy / uav.max_energy,
                current_sinr / 20.0, # SINR 정규화 (임의의 값 20.0)
                uav.p_los # p_los는 0~1 값이므로 정규화 불필요
            ]
            
            # 2. Observation of other UAVs (relative positions, normalized)
            other_uav_obs = []
            for j, other_uav in enumerate(self.uavs):
                if i == j: continue
                other_uav_obs.extend((other_uav.pos[:2] - uav.pos[:2]) / self.world_size)
                
            # 3. Observation of goal (vector to goal, normalized)
            goal_obs = (self.uav_goals[i] - uav.pos[:2]) / self.world_size
            
            # Concatenate all parts into a single observation vector
            full_obs = np.concatenate([self_obs, other_uav_obs, goal_obs])
            obs_list.append(full_obs)
        return obs_list

    def _get_high_level_obs(self):
        """
        Generates the global observation for the high-level agent.
        :return: (list) A list containing one global observation array.
        """
        # (목표 설계 반영) S_high: (POI 위치) + (POI 방문 상태) + (Jammer 위치) + (UAV 상태)
        
        # 1. POI information (normalized positions)
        poi_info = np.array([[p.pos[0]/self.world_size, p.pos[1]/self.world_size] for p in self.pois]).flatten()
        
        # 2. (NEW) POI visited status (binary)
        poi_status = np.array([1.0 if p.visited else 0.0 for p in self.pois])

        # 3. (NEW) Jammer information (normalized positions)
        jammer_info = np.array([[j.pos[0]/self.world_size, j.pos[1]/self.world_size] for j in self.jammers]).flatten()
        
        # 4. UAV information (normalized positions, energy, sinr)
        uav_info_list = []
        for u in self.uavs:
            # (S_low와 일관성을 위해 _calculate_sinr을 여기서 호출)
            sinr_norm = self._calculate_sinr(u) / 20.0
            uav_info_list.extend([
                u.pos[0]/self.world_size,
                u.pos[1]/self.world_size,
                u.energy/u.max_energy,
                sinr_norm
            ])
        uav_info = np.array(uav_info_list).flatten()
        
        # Pad POI info (필요시) - (현재 로직은 방문해도 POI를 제거하지 않으므로 불필요할 수 있으나 유지)
        if len(poi_info) < self.num_pois * 2:
            poi_info = np.pad(poi_info, (0, self.num_pois*2 - len(poi_info)), 'constant')
        if len(poi_status) < self.num_pois:
            poi_status = np.pad(poi_status, (0, self.num_pois - len(poi_status)), 'constant')
            
        # Return as a list (matching the Tuple space)
        return [np.concatenate([poi_info, poi_status, jammer_info, uav_info])]

    def _update_clusters(self):
        """
        Updates the POI cluster centers using KMeans.
        This is called by `reset()` to define goals for the new episode.
        """
        # Get positions of unvisited POIs
        # (set_goals 로직 변경) unvisited POI의 '글로벌 인덱스' 저장
        self.unvisited_poi_indices = np.where([not p.visited for p in self.pois])[0]
        poi_positions = np.array([self.pois[idx].pos for idx in self.unvisited_poi_indices])
        
        # If fewer POIs remain than clusters, just use POI positions as "centers"
        if len(poi_positions) < self.num_clusters:
            # Handle the case with very few or zero POIs left
            if len(poi_positions) == 0:
                default_pos = [[self.world_size/2, self.world_size/2]] * self.num_clusters
                self.cluster_centers = np.array(default_pos)
                self.poi_cluster_labels = np.array([], dtype=int) # 레이블 없음
            else:
                extra_centers = [[self.world_size/2, self.world_size/2]] * (self.num_clusters - len(poi_positions))
                self.cluster_centers = np.array(list(poi_positions) + extra_centers)
                # POI가 클러스터보다 적으면, 각 POI가 자체 클러스터 레이블을 가짐
                self.poi_cluster_labels = np.arange(len(poi_positions))
            return
        
        # Run KMeans to find new cluster centers
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10, random_state=0).fit(poi_positions)
        self.cluster_centers = kmeans.cluster_centers_
        
        # (set_goals 로직 변경) K-Means 결과(레이블) 저장
        self.poi_cluster_labels = kmeans.labels_

    def set_goals(self, high_level_actions):
        """
        Sets the low-level goals (self.uav_goals) based on the high-level policy's
        actions (cluster indices).
        
        (목표 설계 반영)
        이제 Low-level goal은 cluster center가 아니라,
        "UAV에게 할당된 클러스터 내의 가장 가까운 미방문 POI"가 됩니다.
        
        :param high_level_actions: (np.ndarray) An array of cluster indices,
                                   one for each UAV. Shape (num_uavs,)
        """
        actions = np.array(high_level_actions).flatten()
        
        # (목표 설계 반영) 새로운 set_goals 로직
        
        if self.cluster_centers is None or len(self.cluster_centers) == 0:
             # _update_clusters가 아직 호출되지 않았거나 POI가 없는 극단적 케이스
            for i in range(self.num_uavs):
                self.uav_goals[i] = self.uavs[i].pos[:2] # 현재 위치를 목표로 설정
            return

        for i, uav in enumerate(self.uavs):
            assigned_cluster_idx = actions[i]
            
            # 1. 이 클러스터에 할당된 (아직 방문하지 않은) POI들의 인덱스를 찾습니다.
            mask_pois_in_cluster = (self.poi_cluster_labels == assigned_cluster_idx)
            
            # (self.unvisited_poi_indices는 "글로벌" POI 인덱스를 담고 있음)
            global_indices_of_pois_in_cluster = self.unvisited_poi_indices[mask_pois_in_cluster]
            
            if len(global_indices_of_pois_in_cluster) == 0:
                # 2. 만약 이 클러스터에 방문할 POI가 없다면 (모두 방문 완료),
                #    목표를 해당 클러스터의 '중심'으로 설정합니다 (대안)
                self.uav_goals[i] = self.cluster_centers[assigned_cluster_idx]
            else:
                # 3. 방문할 POI가 있다면, 이 POI들 중 UAV와 가장 가까운 POI를 찾습니다.
                poi_positions_in_cluster = [self.pois[global_idx].pos for global_idx in global_indices_of_pois_in_cluster]
                
                dists = [np.linalg.norm(uav.pos[:2] - poi_pos) for poi_pos in poi_positions_in_cluster]
                
                # 4. 가장 가까운 POI의 위치를 Low-level 목표로 설정합니다.
                nearest_poi_global_idx = global_indices_of_pois_in_cluster[np.argmin(dists)]
                self.uav_goals[i] = self.pois[nearest_poi_global_idx].pos

    def render(self, mode='human'):
        """
        Renders the environment using matplotlib.
        :param mode: 'human' (shows a plot) or 'rgb_array' (returns an image).
        """
        if self.fig is None:
            # Initialize the plot
            plt.ion()   # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(0, self.world_size)
            self.ax.set_ylim(0, self.world_size)
            self.ax.set_aspect('equal')
            
            # Create plot objects for all entities
            for i in range(self.num_uavs):
                self.uav_plots[i] = self.ax.add_patch(Circle((0, 0), 2, color=f'C{i}', label=f'UAV {i}'))
                self.goal_plots[i] = self.ax.plot([], [], 'x', color=f'C{i}', markersize=10)[0]
            for i in range(self.num_pois):
                self.poi_plots[i] = self.ax.add_patch(Rectangle((0, 0), 3, 3, color='green', alpha=0.6))
            for i in range(self.num_jammers):
                self.jammer_plots[i] = self.ax.add_patch(Circle((0, 0), self.jammers[i].radius, color='red', alpha=0.3))
            self.ax.legend()

        # Update positions of all entities
        for i, uav in enumerate(self.uavs):
            self.uav_plots[i].center = uav.pos[:2]
            # Draw trajectory
            if len(self.uav_trajectories[i]) > 1:
                traj = np.array(self.uav_trajectories[i])
                self.ax.plot(traj[:, 0], traj[:, 1], '-', color=f'C{i}', alpha=0.3)

        for i, poi in enumerate(self.pois):
            self.poi_plots[i].set_xy(poi.pos - 1.5)
            self.poi_plots[i].set_visible(not poi.visited)  # Hide visited POIs

        for i, jammer in enumerate(self.jammers):
            self.jammer_plots[i].center = jammer.pos

        # Update goal markers ('x')
        if self.uav_goals is not None:
            for i, goal in enumerate(self.uav_goals):
                self.goal_plots[i].set_data([goal[0]], [goal[1]])

        # Redraw the canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        if mode == 'human':
            plt.pause(0.01) # Short pause to update the window
            return None
        
        elif mode == 'rgb_array':
            # Convert the matplotlib canvas to an RGB numpy array
            w, h = self.fig.canvas.get_width_height()
            buf = self.fig.canvas.tostring_argb()
            image_array = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
            image_array_rgb = image_array[:, :, 1:]     # Drop the alpha channel
            return image_array_rgb
        
    def close(self):
        """Closes the rendering window."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None