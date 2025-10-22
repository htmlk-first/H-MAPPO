import numpy as np
from .parameters import *

class UAV:
    """
    Represents a single UAV (Unmanned Aerial Vehicle) agent in the environment.
    This class holds the state information for one agent.
    """
    def __init__(self, start_x, start_y, start_z, v_max, u_max):
        """
        Initializes the UAV's state.
        :param start_x: Initial x-coordinate.
        :param start_y: Initial y-coordinate.
        :param start_z: Initial z-coordinate (altitude).
        :param v_max: Maximum velocity allowed for the UAV.
        :param u_max: Maximum acceleration allowed for the UAV.
        """
        # Kinematic state
        self.pos = np.array([start_x, start_y, start_z], dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.acc = np.zeros(3, dtype=np.float32)
        
        # Kinematic constraints
        self.v_max = v_max
        self.u_max = u_max
        
        # --- State variables for communication and energy ---
        self.max_energy = 1000.0    # Maximum energy capacity (example value)
        self.energy = self.max_energy   # Current remaining energy
        self.comm_mode = 0   # 0: Traditional, 1: Semantic
        self.sem_level = 0   # Semantic communication level
        self.transmit_power = 1.0   # Communication transmission power
        
    def reset(self):
        """
        Resets the UAV's state to a new random position and full energy.
        (Note: This method seems unused; uav_env.py handles reset manually).
        """
        self.pos = np.array([np.random.uniform(0, AREA_WIDTH),
                               np.random.uniform(0, AREA_HEIGHT),
                               UAV_ALTITUDE])
        self.velocity = np.zeros(2)
        self.rem_energy = UAV_MAX_ENERGY
        self.assigned_cluster_id = -1
        
class Jammer:
    """
    Represents a Jammer obstacle in the environment.
    Jammers interfere with UAV communication within their radius.
    """
    def __init__(self, x, y, radius, power):
        """
        Initializes the Jammer's properties.
        :param x: Jammer's x-coordinate.
        :param y: Jammer's y-coordinate.
        :param radius: Effective jamming radius.
        :param power: Jamming signal power.
        """
        self.pos = np.array([x, y], dtype=np.float32)   # Jammer's 2D position
        self.radius = radius    # Jammer's effective radius
        self.power = power  # Jammer's signal power

class PointOfInterest:
    def __init__(self, x, y):
        """
        Represents a Point of Interest (PoI) target in the environment.
        UAVs are rewarded for visiting these points.
        """
        self.pos = np.array([x, y], dtype=np.float32)   # PoI's 2D position
        self.visited = False    # Flag to track if the PoI has been visited
        self.aoi = 0    # Age of Information (AoI) for this PoI