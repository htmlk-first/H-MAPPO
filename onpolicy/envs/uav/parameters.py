import numpy as np

# Simulation Parameters
AREA_WIDTH = 1000  # meters
AREA_HEIGHT = 1000  # meters
SIM_TIME_STEPS = 500
N_UAVS = 4
N_TARGETS = 20
N_JAMMERS = 2

# UAV Parameters
UAV_ALTITUDE = 100  # meters
UAV_MAX_SPEED = 20  # m/s
UAV_MAX_ENERGY = 10000.0  # Joules (Default: 10 kJ)
UAV_SAFE_DISTANCE = 50 # meters for collision penalty

# Propulsion Energy Model Parameters (for non-linear model)
# 현실적인 값으로 추후 논문 서베이를 통해 튜닝 필요
P_IDLE = 80.0  # W (Hovering power)
U_TIP = 120    # m/s (Tip speed of the rotor blade)
V_0 = 4.03     # m/s (Mean rotor induced velocity in hover)
D_0 = 0.6      # Fuselage drag ratio
RHO = 1.225    # Air density (kg/m^3)
S_0 = 0.05     # Rotor solidity
A = 0.503      # Rotor disc area (m^2)


# Communication Parameters
BANDWIDTH = 10e6  # 10 MHz
NOISE_DENSITY = 10**(-20.4)  # -174 dBm/Hz in Watts/Hz
UAV_TX_POWER = 0.1  # 20 dBm in Watts
TARGET_TX_POWER = 0.2  # 23 dBm in Watts
JAMMER_TX_POWER = 1.0  # 30 dBm in Watts
LOS_A = 9.61
LOS_B = 0.16
NLOS_MULTIPLIER = 0.2 # NLoS path loss is 5x worse than LoS
DATA_PACKET_SIZE = 1e6 # 1 Mbit

# Semantic Communication Parameters
SEMANTIC_LEVELS = {
    "high": 0.05,  # High compression
    "mid": 0.1,
    "low": 0.2
}
PROC_DELAY_COEFF = 0.005
PROC_ENERGY_COEFF = 0.025
QUALITY_FUNC_COEFF = 2.0
SINR_THRESHOLD = 5 # dB

# MARL Parameters
# Action space definitions
ACTION_V = 1 # Velocity magnitude (continuous)
ACTION_THETA = 1 # Velocity angle (continuous)
ACTION_COMM_MODE = 2 # Comm mode (discrete: 0=Trad, 1=Sem)
ACTION_SEM_LEVEL = 3 # Semantic level (discrete: 0=High, 1=Mid, 2=Low)

# Reward Function Weights
W_QUALITY = 1.0
W_ENERGY = 0.1
W_LATENCY = 0.2

# Event-based Rewards/Penalties
R_COVER = 10.0
R_COMPLETE = 200.0
P_TIMEOUT = 100.0
P_REVISIT = 20.0
P_COLLISION = 200.0