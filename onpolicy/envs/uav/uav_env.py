import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from gym import spaces
from sklearn.cluster import KMeans

from .entities import UAV, Jammer, PointOfInterest

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

        # 2. Create simulation entities (Agents, Targets, Obstacles)
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

        # 3. H-MAPPO related variables
        self.num_clusters = self.num_uavs   # Number of clusters for POIs
        self.cluster_centers = np.zeros((self.num_clusters, 2)) # 2D coordinates of cluster centers
        self.uav_goals = np.zeros((self.num_uavs, 2))   # 2D coordinates of the goal for each UAV

        # 4. Communication and energy model parameters
        self.sinr_threshold = 5 # dB, threshold for successful communication
        self.noise_power = 1e-9 # Noise power in SINR calculation
        self.energy_prop_coeff = 0.1 # Coefficient for propulsion energy
        self.energy_comm_trad = 0.01 # Energy cost for traditional communication
        self.energy_comm_sem = 0.05 # Energy cost for semantic communication

        # 5. Define hierarchical observation and action spaces using gym.spaces
        
        # --- High-Level Spaces (Global Policy, 1 Agent) ---
        # Obs: (POI_positions) + (UAV_states)
        high_obs_dim = (self.num_pois * 2) + (self.num_uavs * 3)
        high_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(high_obs_dim,), dtype=np.float32)
        # Action: Assign a cluster index [0, num_clusters-1] to each UAV
        high_act_space = spaces.MultiDiscrete([self.num_clusters] * self.num_uavs)

        # --- Low-Level Spaces (Local Policy, num_uavs Agents) ---
        # Obs: (self_pos, energy, sinr) + (relative_pos_of_other_UAVs) + (vector_to_goal)
        low_obs_dim = 4 + (self.num_uavs - 1) * 2 + 2
        low_act_space_per_agent = spaces.Box(low=-np.inf, high=np.inf, shape=(low_obs_dim,), dtype=np.float32)
        # Action: (Move Direction [0-8], Speed [0-2], Comm Mode [0-1], Sem Level [0-2])
        low_act_space = spaces.MultiDiscrete([9, 3, 2, 3])
        
        # --- Combine into a gym.spaces.Dict ---
        # The H-MAPPO runner expects this dictionary structure.
        # Note: The spaces are wrapped in a Tuple([space]) to match the runner's expectations.
        self.action_space = spaces.Dict({
            'high_level': spaces.Tuple([high_act_space]),
            'low_level': spaces.Tuple([low_act_space for _ in range(self.num_uavs)])
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
            # Unpack the low-level action for this UAV
            move_dir_idx, speed_idx, comm_mode, sem_level = low_level_actions[i]
            dist_to_goal_before = np.linalg.norm(uav.pos[:2] - self.uav_goals[i])
            
            # 1. Apply movement
            self._uav_move(uav, move_dir_idx, speed_idx)
            dist_to_goal_after = np.linalg.norm(uav.pos[:2] - self.uav_goals[i])
            
            # 2. Calculate propulsion energy cost
            # Simplified energy model: proportional to v^2
            prop_energy = self.energy_prop_coeff * (self.uavs[0].v_max * (speed_idx + 1) / 3)**2
            uav.energy -= prop_energy
            
            # 3. Calculate communication energy cost and SINR penalty/reward
            sinr = self._calculate_sinr(uav)
            comm_energy = 0
            if comm_mode == 0:  # Traditional mode
                comm_energy = self.energy_comm_trad
                if sinr < self.sinr_threshold: low_level_rewards[i] -= 2    # Penalty for low SINR in trad mode
            else:   # Semantic mode
                comm_energy = self.energy_comm_sem * (1 + sem_level * 0.5)
                if sinr < self.sinr_threshold: low_level_rewards[i] += 1    # Small reward for attempting sem comm in low SINR
            uav.energy -= comm_energy
            
            # 4. Calculate low-level reward
            # Reward for moving closer to the high-level goal
            low_level_rewards[i] += (dist_to_goal_before - dist_to_goal_after) * 0.5
            # Penalty for energy consumption
            low_level_rewards[i] -= (prop_energy + comm_energy) * 0.1

        # 5. Calculate high-level reward (global)
        high_level_reward = 0.0
        for poi in self.pois:
            if not poi.visited:
                for uav in self.uavs:
                    # Check if any UAV is close enough to "visit" the POI
                    if np.linalg.norm(uav.pos[:2] - poi.pos) < 5:
                        poi.visited = True
                        high_level_reward += 20     # Reward for visiting a new POI
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

    def _uav_move(self, uav, dir_idx, speed_idx):
        if dir_idx == 8: return     # 8 is the 'stay' action
        
        # Convert discrete direction index to an angle
        angle = (dir_idx / 8) * 2 * np.pi
        # Convert discrete speed index to a speed value
        speed = uav.v_max * (speed_idx + 1) / 3
        
        # Update 2D position
        uav.pos[0] += speed * np.cos(angle)
        uav.pos[1] += speed * np.sin(angle)
        
        # Clip position to stay within world boundaries
        uav.pos[0] = np.clip(uav.pos[0], 0, self.world_size)
        uav.pos[1] = np.clip(uav.pos[1], 0, self.world_size)
        
        # Store trajectory for rendering
        self.uav_trajectories[self.uavs.index(uav)].append(uav.pos[:2].copy())

    def _calculate_sinr(self, uav):
        """Helper function to calculate the SINR for a UAV."""
        # Simplified path loss model
        dist_to_gbs = np.linalg.norm(uav.pos) # Assuming gBS is at (0,0,0)
        path_loss = 20 * np.log10(dist_to_gbs) + 20 * np.log10(4 * np.pi * 2.4e9 / 3e8) if dist_to_gbs > 0 else 0
        
        # Calculate interference from all jammers within their radius
        jammer_interference = 0
        for j in self.jammers:
            dist_to_jammer = np.linalg.norm(uav.pos[:2] - j.pos)
            if dist_to_jammer < j.radius:
                jammer_interference += j.power / (dist_to_jammer**2 + 1e-9) # 1e-9 to avoid division by zero

        signal_power = uav.transmit_power / (10**(path_loss/10)) if path_loss > 0 else uav.transmit_power
        
        # Calculate SINR
        denominator = self.noise_power + jammer_interference
        sinr = signal_power / denominator if denominator > 0 else signal_power
        
        # Convert to dB
        return 10 * np.log10(sinr) if sinr > 0 else -100

    def _get_low_level_obs(self):
        """
        Generates the local observation for each low-level agent (UAV).
        :return: (list) A list of observation arrays, one for each UAV.
        """
        
        obs_list = []
        for i, uav in enumerate(self.uavs):
            # 1. Self observation (normalized)
            self_obs = [
                uav.pos[0] / self.world_size,
                uav.pos[1] / self.world_size,
                uav.energy / uav.max_energy,
                self._calculate_sinr(uav) / 20.0
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
        # 1. POI information (normalized positions)
        poi_info = np.array([[p.pos[0]/self.world_size, p.pos[1]/self.world_size] for p in self.pois]).flatten()
        
        # 2. UAV information (normalized positions and energy)
        uav_info = np.array([[u.pos[0]/self.world_size, u.pos[1]/self.world_size, u.energy/u.max_energy] for u in self.uavs]).flatten()
        
        # Pad POI info if some POIs have been visited (to maintain fixed obs dim)
        if len(poi_info) < self.num_pois * 2:
            poi_info = np.pad(poi_info, (0, self.num_pois*2 - len(poi_info)), 'constant')
            
        # Return as a list (matching the Tuple space)
        return [np.concatenate([poi_info, uav_info])]

    def _update_clusters(self):
        """
        Updates the POI cluster centers using KMeans.
        This is called by `reset()` to define goals for the new episode.
        """
        # Get positions of unvisited POIs
        poi_positions = np.array([p.pos for p in self.pois if not p.visited])
        
        # If fewer POIs remain than clusters, just use POI positions as "centers"
        if len(poi_positions) < self.num_clusters:
            # Handle the case with very few or zero POIs left
            if len(poi_positions) == 0:
                default_pos = [[self.world_size/2, self.world_size/2]] * self.num_clusters
                self.cluster_centers = np.array(default_pos)
            else:
                extra_centers = [[self.world_size/2, self.world_size/2]] * (self.num_clusters - len(poi_positions))
                self.cluster_centers = np.array(list(poi_positions) + extra_centers)
            return
        
        # Run KMeans to find new cluster centers
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10, random_state=0).fit(poi_positions)
        self.cluster_centers = kmeans.cluster_centers_

    def set_goals(self, high_level_actions):
        """
        Sets the low-level goals (self.uav_goals) based on the high-level policy's
        actions (cluster indices).
        This method is called by the H_UAVRunner.
        :param high_level_actions: (np.ndarray) An array of cluster indices,
                                   one for each UAV. Shape (num_uavs,)
        """
        actions = np.array(high_level_actions).flatten()
        if self.cluster_centers is not None and len(self.cluster_centers) > 0:
            # Map each UAV's action (cluster index) to a 2D coordinate
            self.uav_goals = np.array([self.cluster_centers[min(act, len(self.cluster_centers)-1)] for act in actions])

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