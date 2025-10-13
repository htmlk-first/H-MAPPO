import numpy as np
from .parameters import *

class UAV:
    def __init__(self, id):
        self.id = id
        self.pos = np.zeros(3)  # [x, y, z]
        self.velocity = np.zeros(2) # [vx, vy]
        self.rem_energy = UAV_MAX_ENERGY
        self.assigned_cluster_id = -1

    def reset(self):
        self.pos = np.array([np.random.uniform(0, AREA_WIDTH),
                               np.random.uniform(0, AREA_HEIGHT),
                               UAV_ALTITUDE])
        self.velocity = np.zeros(2)
        self.rem_energy = UAV_MAX_ENERGY
        self.assigned_cluster_id = -1

class Target:
    def __init__(self, id):
        self.id = id
        self.pos = np.zeros(2) # [x, y]
        self.is_visited = False

    def reset(self):
        self.pos = np.array([np.random.uniform(0, AREA_WIDTH),
                               np.random.uniform(0, AREA_HEIGHT)])
        self.is_visited = False

class Jammer:
    def __init__(self, id):
        self.id = id
        self.pos = np.zeros(2) # [x, y]

    def reset(self, predefined_pos=None):
        if predefined_pos is not None:
            self.pos = np.array(predefined_pos)
        else:
            self.pos = np.array([np.random.uniform(0, AREA_WIDTH),
                                   np.random.uniform(0, AREA_HEIGHT)])