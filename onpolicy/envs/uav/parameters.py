import numpy as np

# --- Simulation Parameters ---
# Defines the basic properties of the simulation world.

AREA_WIDTH = 1000       # meters
AREA_HEIGHT = 1000      # meters
SIM_TIME_STEPS = 500    # Maximum number of steps per episode
N_UAVS = 4              # Default number of UAVs (agents)
N_TARGETS = 20          # Default number of Points of Interest (PoIs)
N_JAMMERS = 2           # Default number of Jammers
TIME_DELTA = 1.0        # Duration of a single time step in seconds

# --- UAV Parameters ---
# Defines the physical properties and constraints of the UAV agents.

UAV_ALTITUDE = 100      # meters (Fixed flying altitude)
UAV_MAX_SPEED = 20      # m/s
UAV_MAX_ENERGY = 10000.0 # Joules (Default: 10 kJ)
UAV_SAFE_DISTANCE = 50   # meters (Used for collision penalty)

# --- Propulsion Energy Model Parameters (for non-linear model) ---
# These parameters define a more complex energy model.
# (Note: These seem intended for a different model than the one
# currently implemented in uav_env.py, which uses a simpler model).
# Korean Comment: "Needs tuning with realistic values, possibly through future paper surveys."
P_IDLE = 80.0           # W (Hovering power)
U_TIP = 120             # m/s (Tip speed of the rotor blade)
V_0 = 4.03              # m/s (Mean rotor induced velocity in hover)
D_0 = 0.6               # Fuselage drag ratio
RHO = 1.225             # Air density (kg/m^3)
S_0 = 0.05              # Rotor solidity
A = 0.503               # Rotor disc area (m^2)

# --- Simplified Energy Model (uav_env.py) ---
# Parameters used directly by the current uav_env.py implementation
ENERGY_COMM_TRAD = 0.01   # Energy cost for traditional communication
ENERGY_COMM_SEM = 0.05    # Base energy cost for semantic communication
ENERGY_PROP_COEFF = 0.1  # Coefficient for propulsion energy calculation

# --- Communication Parameters ---
# Defines parameters for the wireless communication model (SINR calculation).

BANDWIDTH = 10e6        # 10 MHz
NOISE_DENSITY = 10**(-20.4) # -174 dBm/Hz in Watts/Hz
UAV_TX_POWER = 0.1      # 20 dBm in Watts (UAV's transmission power)
TARGET_TX_POWER = 0.2   # 23 dBm in Watts (Target's transmission power)
JAMMER_TX_POWER = 1.0   # 30 dBm in Watts (Jammer's transmission power)
LOS_A = 9.61            # Path loss coefficient for Line-of-Sight (LoS)
LOS_B = 0.16            # Path loss exponent for LoS
NLOS_MULTIPLIER = 0.2   # Non-Line-of-Sight (NLoS) path loss multiplier (e.g., 0.2 means 5x worse)
DATA_PACKET_SIZE = 1e6  # 1 Mbit (Size of data to be collected/transmitted)

# --- Channel Model Parameters (uav_env.py) ---
FC = 2.4e9              # Carrier frequency (2.4 GHz)
C = 3e8                 # Speed of light (m/s)
ETA_LOS = 1.0           # Additional path loss for LoS (dB)
ETA_NLOS = 20.0         # Additional path loss for NLoS (dB)

# --- Semantic Communication Parameters ---
# Parameters related to the semantic communication mode.

SEMANTIC_LEVELS = {
    "high": 0.05,       # High compression (5% of original data)
    "mid": 0.1,         # Medium compression (10%)
    "low": 0.2          # Low compression (20%)
}
PROC_DELAY_COEFF = 0.005  # Coefficient for processing delay
PROC_ENERGY_COEFF = 0.025 # Coefficient for processing energy consumption
QUALITY_FUNC_COEFF = 2.0  # Coefficient for calculating data quality
SINR_THRESHOLD = 5      # dB (Threshold for successful communication)

# --- Fidelity Model Parameters (uav_env.py) ---
SEM_BASE_QUALITY = np.array([0.6, 0.75, 0.9]) # Base quality for sem_level 0, 1, 2
SEM_ROBUST_COEFF = 0.5  # 'k' - Steepness of the fidelity sigmoid curve
SEM_ROBUST_OFFSET = 0.0 # 'S_offset' - SINR offset for the fidelity sigmoid curve

# --- MARL (Multi-Agent Reinforcement Learning) Parameters ---

# Action space definitions (Note: These are for reference;
# the actual space is defined in uav_env.py)
ACTION_V = 1            # Velocity magnitude (continuous)
ACTION_THETA = 1        # Velocity angle (continuous)
ACTION_COMM_MODE = 2    # Comm mode (discrete: 0=Trad, 1=Sem)
ACTION_SEM_LEVEL = 3    # Semantic level (discrete: 0=High, 1=Mid, 2=Low)

# --- Reward Function Weights ---
# Weights for different components of the reward signal.

W_QUALITY = 1.0         # Weight for data quality
W_ENERGY = 0.1          # Weight for energy consumption (penalty)
W_LATENCY = 0.2         # Weight for latency (penalty)

# --- Event-based Rewards/Penalties ---
# Discrete rewards/penalties for specific events.

R_COVER = 10.0          # Reward for covering a PoI
R_COMPLETE = 200.0      # Reward for completing all PoIs
P_TIMEOUT = 100.0       # Penalty for running out of time
P_REVISIT = 20.0        # Penalty for revisiting an already-covered PoI
P_COLLISION = 200.0     # Penalty for colliding (getting too close) to another UAV