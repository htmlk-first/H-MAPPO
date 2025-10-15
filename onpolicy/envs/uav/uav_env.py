import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import gymnasium as gym  # gymnasium import
from gymnasium.spaces import Box, Discrete, MultiDiscrete
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .parameters import *
from .entities import UAV, Target, Jammer


class UAVEnv:
    def __init__(self):
        # Initialize entities
        self.uavs = [UAV(i) for i in range(N_UAVS)]
        self.targets = [Target(i) for i in range(N_TARGETS)]
        self.jammers = [Jammer(i) for i in range(N_JAMMERS)]
        self.time_step = 0

        # High-level agent's state/action space (placeholder)
        self.n_clusters = N_UAVS  # One cluster per UAV for now
        self.cluster_centers = None
        self.target_clusters = None
        
        # Visualization setup
        self.fig, self.ax = None, None
        self.uav_plots = []
        self.uav_trajectories = []
        self.target_plots = []
        self.jammer_plots = []
        self.cluster_center_plots = []
        self.info_text = None

        # Low-level agent's observation space
        # [pos_x, pos_y, rem_energy, other_uav_rel_pos (3*2), rel_cluster_pos (2)]
        self.observation_space = []
        obs_dim = 3 + (N_UAVS - 1) * 2 + 2
        for _ in range(N_UAVS):
            self.observation_space.append(Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32))

        # Centralized critic's observation space (share_observation_space)
        self.share_observation_space = []
        share_obs_dim = obs_dim * N_UAVS
        for _ in range(N_UAVS):
            self.share_observation_space.append(Box(low=-np.inf, high=np.inf, shape=(share_obs_dim,), dtype=np.float32))
        
        # Low-level agent's action space (Hybrid: Continuous + Discrete)
        # Continuous part: [speed, angle]
        # Discrete part: [comm_mode, semantic_level]
        self.action_space = []
        for _ in range(N_UAVS):
            # Continuous actions for velocity
            continuous_action_space = Box(low=np.array([0, -np.pi]), high=np.array([UAV_MAX_SPEED, np.pi]), dtype=np.float32)
            # Discrete actions for communication
            discrete_action_space = MultiDiscrete([ACTION_COMM_MODE, ACTION_SEM_LEVEL])
            # We will handle this hybrid space in the runner
            self.action_space.append((continuous_action_space, discrete_action_space))

    def reset(self):
        # Reset trajectory paths for visualization
        for uav in self.uavs:
            uav.path = []
        
        # Resets the environment to an initial state.
        self.time_step = 0
        for uav in self.uavs:
            uav.reset()
            uav.path.append(uav.pos[:2].copy()) # Store initial position
        for target in self.targets:
            target.reset()

        self.jammers[0].reset(predefined_pos=[250, 250])
        self.jammers[1].reset(predefined_pos=[750, 750])

        self._update_clusters_and_assignments()

        # gymnasium 표준에 맞게 (obs, info) 튜플을 반환
        return self.get_state(), {}

    def step(self, actions):
        """
        Processes one step of the simulation.
        actions: A list of actions, one for each UAV.
                 Each action is a tuple: (velocity_vector, comm_mode, semantic_level)
        """
        rewards = np.zeros(N_UAVS)
        total_reward = 0

        # Low-level agent actions
        total_quality = 0
        total_energy_consumed = 0
        total_latency = 0

        for i, uav in enumerate(self.uavs):
            if uav.rem_energy <= 0:
                uav.path.append(uav.pos[:2].copy()) # 움직이지 않더라도 경로 기록
                continue # 에너지가 없으면 이번 스텝의 행동을 건너뜀
            
            flat_action = actions[i]

            # Debug: Print the action taken by UAV 0
            if i == 0: 
                print(f"Step {self.time_step}, UAV 0 Action: [Speed/Angle: ({flat_action[0]:.2f}, {flat_action[1]:.2f}), Comm: {np.round(flat_action[2])}, Sem: {np.round(flat_action[3])}]")
            
            # --- DE-FLATTEN THE ACTION TENSOR ---
            # Continuous part (first 2 elements): speed, angle
            cont_action = flat_action[0:2]
            speed, angle = np.abs(cont_action[0]), cont_action[1]
            vel_action = np.array([speed * np.cos(angle), speed * np.sin(angle)])

            # Discrete part (rest of elements): comm_mode, semantic_level
            # These are now floats from concatenation, so we must round and cast to int
            disc_actions = np.round(flat_action[2:]).astype(int)
            # comm_mode_action = disc_actions[0]
            # sem_level_action = disc_actions[1]
            comm_mode_action = np.clip(disc_actions[0], 0, ACTION_COMM_MODE - 1)
            sem_level_action = np.clip(disc_actions[1], 0, ACTION_SEM_LEVEL - 1)

            # 1. Update UAV position and calculate propulsion energy
            propulsion_energy = self._calculate_propulsion_energy(uav, vel_action)
            uav.rem_energy -= propulsion_energy
            uav.pos[0] += vel_action[0]
            uav.pos[1] += vel_action[1]
            uav.pos = np.clip(uav.pos, [0, 0, UAV_ALTITUDE], [AREA_WIDTH, AREA_HEIGHT, UAV_ALTITUDE])
            uav.path.append(uav.pos[:2].copy())

            # 2. Communication logic
            # Find nearest unvisited target in assigned cluster
            my_cluster_targets = [t for t_idx, t in enumerate(self.targets) if self.target_clusters[t_idx] == uav.assigned_cluster_id and not t.is_visited]

            quality = 0
            latency = 0
            comm_energy = 0

            if my_cluster_targets:
                target_dists = [np.linalg.norm(uav.pos[:2] - t.pos) for t in my_cluster_targets]
                nearest_target = my_cluster_targets[np.argmin(target_dists)]

                sinr = self._calculate_sinr(uav, nearest_target)

                # Calculate costs based on communication mode
                if comm_mode_action == 0:  # Traditional
                    quality, latency, comm_energy = self._calculate_trad_comm_costs(sinr)
                else:  # Semantic
                    semantic_lambda = list(SEMANTIC_LEVELS.values())[sem_level_action]
                    quality, latency, comm_energy = self._calculate_sem_comm_costs(sinr, semantic_lambda)

                uav.rem_energy -= comm_energy

            total_quality += quality
            total_energy_consumed += (propulsion_energy + comm_energy)
            total_latency += latency

        # 3. Calculate event-based rewards/penalties
        event_reward = self._calculate_event_rewards()

        # 4. Formulate the final reward
        reward = (W_QUALITY * total_quality) - \
                 (W_ENERGY * total_energy_consumed) - \
                 (W_LATENCY * total_latency) + \
                 event_reward

        # 5. High-level agent action (periodic re-clustering)
        if self.time_step % 50 == 0:
            self._update_clusters_and_assignments()

        self.time_step += 1
        done_for_all = self.time_step >= SIM_TIME_STEPS or all(t.is_visited for t in self.targets)

        obs = self.get_state()
        dones = [done_for_all for _ in range(N_UAVS)]
        infos = {}

        rewards = [reward for _ in range(N_UAVS)]

        return obs, rewards, dones, infos

    def render(self, mode='human'):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            
            uav_colors = ['blue', 'red', 'green', 'purple']
            for i in range(N_UAVS):
                plot = self.ax.scatter([], [], c=uav_colors[i], marker='^', s=100, label=f'UAV {i}', zorder=3)
                self.uav_plots.append(plot)
                traj, = self.ax.plot([], [], c=uav_colors[i], linestyle='-', linewidth=1, zorder=2)
                self.uav_trajectories.append(traj)

            self.target_plots = self.ax.scatter([], [], c='gray', marker='o', s=50, label='Target', zorder=1)
            
            for jammer in self.jammers:
                circle = patches.Circle(jammer.pos, 150, color='orangered', alpha=0.3, zorder=0)
                self.ax.add_patch(circle)
            
            self.cluster_center_plots = self.ax.scatter([], [], c='black', marker='x', s=100, label='Cluster Center', zorder=4)
            self.info_text = self.ax.text(0.02, 1.02, '', transform=self.ax.transAxes)
            self.ax.legend(loc='upper right')

        self.ax.set_xlim(0, AREA_WIDTH)
        self.ax.set_ylim(0, AREA_HEIGHT)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        
        uav_positions = np.array([uav.pos[:2] for uav in self.uavs])
        for i in range(N_UAVS):
            self.uav_plots[i].set_offsets(uav_positions[i])
            traj_data = np.array(self.uavs[i].path)
            self.uav_trajectories[i].set_data(traj_data[:,0], traj_data[:,1])

        target_positions = np.array([t.pos for t in self.targets])
        target_colors = ['limegreen' if t.is_visited else 'gray' for t in self.targets]
        self.target_plots.set_offsets(target_positions)
        self.target_plots.set_color(target_colors)

        if self.cluster_centers is not None:
            self.cluster_center_plots.set_offsets(self.cluster_centers)
        
        info_str = f'Time: {self.time_step} / {SIM_TIME_STEPS}\n'
        for i, uav in enumerate(self.uavs):
            info_str += f'UAV {i} Energy: {uav.rem_energy:.0f}\n'
        self.info_text.set_text(info_str)

        self.fig.canvas.draw()
        plt.pause(0.01)
       
        if mode == 'rgb_array':
            self.fig.canvas.draw()
            image_buf = self.fig.canvas.tostring_argb()
            image = np.frombuffer(image_buf, dtype='uint8')
            # 4채널(ARGB)로 reshape한 후, 3채널(RGB)만 슬라이싱
            width, height = self.fig.canvas.get_width_height()
            image = image.reshape(height, width, 4)[:, :, :3]
            return image

    # --- Helper methods for physics and communication models ---

    def _calculate_propulsion_energy(self, uav, velocity_action):
        """Calculates non-linear propulsion energy."""
        speed = np.linalg.norm(velocity_action)
        if speed == 0:
            return P_IDLE  # Hovering power

        power = P_IDLE * (1 + 3 * speed**2 / U_TIP**2) + \
                (0.5 * D_0 * RHO * S_0 * A * speed**3)
        return power  # Energy for 1 time step (J)

    def _calculate_sinr(self, uav, target):
        """Calculates SINR from a target to a UAV."""
        dist_3d = np.linalg.norm(uav.pos - np.append(target.pos, 0))
        dist_2d = np.linalg.norm(uav.pos[:2] - target.pos)

        elevation_angle = np.arctan(UAV_ALTITUDE / dist_2d) * 180 / np.pi
        p_los = 1 / (1 + LOS_A * np.exp(-LOS_B * (elevation_angle - LOS_A)))

        path_loss_los = 20 * np.log10(dist_3d) + 20 * np.log10(4 * np.pi * 915e6 / 3e8)
        path_loss_nlos = path_loss_los + 20

        avg_path_loss = p_los * (10**(-path_loss_los/10)) + (1 - p_los) * (10**(-path_loss_nlos/10))
        channel_gain = 1 / avg_path_loss

        signal_power = TARGET_TX_POWER * channel_gain

        interference = 0
        for jammer in self.jammers:
            jam_dist = np.linalg.norm(uav.pos[:2] - jammer.pos)
            jam_path_loss = 20 * np.log10(jam_dist) + 20 * np.log10(4 * np.pi * 915e6 / 3e8)
            jam_gain = 1 / (10**(-jam_path_loss/10))
            interference += JAMMER_TX_POWER * jam_gain

        noise = NOISE_DENSITY * BANDWIDTH
        sinr = signal_power / (interference + noise)
        return sinr

    def _calculate_trad_comm_costs(self, sinr):
        sinr_db = 10 * np.log10(sinr)
        quality = 1 / (1 + np.exp(-QUALITY_FUNC_COEFF * (sinr_db - SINR_THRESHOLD)))

        if quality < 0.5:
            return 0, SIM_TIME_STEPS, 0

        rate = BANDWIDTH * np.log2(1 + sinr)
        latency = DATA_PACKET_SIZE / rate
        comm_energy = UAV_TX_POWER * latency
        return quality, latency, comm_energy

    def _calculate_sem_comm_costs(self, sinr, semantic_lambda):
        quality = (1 - np.sqrt(semantic_lambda)) * (1 / (1 + np.exp(-0.1 * (10 * np.log10(sinr)))))

        proc_delay = PROC_DELAY_COEFF / semantic_lambda
        proc_energy = PROC_ENERGY_COEFF / (semantic_lambda**2)

        rate = BANDWIDTH * np.log2(1 + sinr)
        trans_latency = (DATA_PACKET_SIZE * semantic_lambda) / rate

        latency = proc_delay + trans_latency
        comm_energy = proc_energy + (UAV_TX_POWER * trans_latency)

        return quality, latency, comm_energy

    def _calculate_event_rewards(self):
        event_reward = 0

        for uav in self.uavs:
            for target in self.targets:
                if np.linalg.norm(uav.pos[:2] - target.pos) < 20:
                    if not target.is_visited:
                        target.is_visited = True
                        event_reward += R_COVER
                    else:
                        event_reward -= P_REVISIT

        for i in range(N_UAVS):
            for j in range(i + 1, N_UAVS):
                dist = np.linalg.norm(self.uavs[i].pos - self.uavs[j].pos)
                if dist < UAV_SAFE_DISTANCE:
                    event_reward -= P_COLLISION

        if all(t.is_visited for t in self.targets):
            event_reward += R_COMPLETE

        if self.time_step >= SIM_TIME_STEPS:
            event_reward -= P_TIMEOUT

        return event_reward

    def _update_clusters_and_assignments(self):
        """High-level agent's placeholder logic for clustering."""
        target_positions = np.array([t.pos for t in self.targets])
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=10).fit(target_positions)
        self.target_clusters = kmeans.labels_
        self.cluster_centers = kmeans.cluster_centers_

        uav_positions = np.array([u.pos[:2] for u in self.uavs])
        dist_matrix = cdist(uav_positions, self.cluster_centers)

        assigned_clusters = [-1] * N_UAVS
        for i in range(self.n_clusters):
            uav_idx = np.argmin(dist_matrix[:, i])
            while uav_idx in assigned_clusters:
                dist_matrix[uav_idx, i] = np.inf
                uav_idx = np.argmin(dist_matrix[:, i])
            assigned_clusters[uav_idx] = i

        for i, uav in enumerate(self.uavs):
            uav.assigned_cluster_id = assigned_clusters[i]

    def get_state(self):
        """Returns the current state of the environment for all agents."""
        states = []
        for i, uav in enumerate(self.uavs):
            self_state = [
                uav.pos[0] / AREA_WIDTH, uav.pos[1] / AREA_HEIGHT,
                uav.rem_energy / UAV_MAX_ENERGY
            ]

            other_uav_states = []
            for j, other_uav in enumerate(self.uavs):
                if i == j:
                    continue
                rel_pos = (other_uav.pos - uav.pos) / AREA_WIDTH
                other_uav_states.extend([rel_pos[0], rel_pos[1]])

            if uav.assigned_cluster_id != -1 and self.cluster_centers is not None:
                my_cluster_center = self.cluster_centers[uav.assigned_cluster_id]
                rel_cluster_pos = (my_cluster_center - uav.pos[:2]) / AREA_WIDTH
            else:
                rel_cluster_pos = np.zeros(2)

            state = np.concatenate([self_state, other_uav_states, rel_cluster_pos])
            states.append(state)
        return states

    def close(self):
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            self.fig, self.ax = None, None
