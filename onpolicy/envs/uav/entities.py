import numpy as np
from .parameters import *

class UAV:
    def __init__(self, start_x, start_y, start_z, v_max, u_max):
        self.pos = np.array([start_x, start_y, start_z], dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.acc = np.zeros(3, dtype=np.float32)
        
        self.v_max = v_max
        self.u_max = u_max
        
        # [추가] 통신 및 에너지 상태 변수
        self.max_energy = 1000.0 # 최대 에너지 (예시 값)
        self.energy = self.max_energy
        self.comm_mode = 0  # 0: Traditional, 1: Semantic
        self.sem_level = 0
        self.transmit_power = 1.0 # 통신 출력 (예시 값)

    def reset(self):
        self.pos = np.array([np.random.uniform(0, AREA_WIDTH),
                               np.random.uniform(0, AREA_HEIGHT),
                               UAV_ALTITUDE])
        self.velocity = np.zeros(2)
        self.rem_energy = UAV_MAX_ENERGY
        self.assigned_cluster_id = -1
        
class Jammer:
    def __init__(self, x, y, radius, power):
        """
        Jammer 객체 초기화
        :param x: Jammer의 x 좌표
        :param y: Jammer의 y 좌표
        :param radius: 재밍 유효 반경
        :param power: 재밍 신호 출력
        """
        self.pos = np.array([x, y], dtype=np.float32)
        self.radius = radius
        self.power = power

class PointOfInterest:
    def __init__(self, x, y):
        """
        Point of Interest (관심 지점) 객체 초기화
        :param x: PoI의 x 좌표
        :param y: PoI의 y 좌표
        """
        self.pos = np.array([x, y], dtype=np.float32)
        self.visited = False
        self.aoi = 0 # Age of Information