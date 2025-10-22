import numpy as np
from .uav_env import UAVEnv
from .parameters import *

# Note: This script appears to be a simple test runner for debugging the UAVEnv
# environment using RANDOM actions. 
# It does NOT use the H-MAPPO policy.
# The actual training is initiated by `onpolicy/scripts/train/train_uav.py`.

def main():
    """
    Main function to run a simple test loop on the UAVEnv.
    """
    
    # NOTE: The current `uav_env.py` __init__ requires an 'args' object, 
    # but this script calls it without one. This script might be
    # outdated or intended for an older version of uav_env.py.
    # To run this, you might need to pass a placeholder 'args' object.
    env = UAVEnv()
    
    # Run for a few test episodes
    for episode in range(3):
        # Reset the environment at the start of each episode
        states = env.reset()
        done = False
        total_reward = 0
        time_step = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        
        # Loop until the episode is done (e.g., max steps reached or goal achieved)
        while not done:
            # --- Placeholder for Agent's Action Selection ---
            # This section generates random actions for each UAV, 
            # ignoring the 'states' received from the environment.
            actions = []
            for i in range(N_UAVS):     # N_UAVS is imported from parameters.py
                # Generate a random continuous action for velocity
                # (This is different from the MultiDiscrete action space
                # defined in the current uav_env.py)
                speed = np.random.uniform(0, UAV_MAX_SPEED)
                angle = np.random.uniform(0, 2 * np.pi)
                vel_action = np.array([speed * np.cos(angle), speed * np.sin(angle)])
                
                # Generate random discrete actions
                # (These also seem based on parameters.py, not the gym space)
                comm_mode_action = np.random.randint(ACTION_COMM_MODE)
                sem_level_action = np.random.randint(ACTION_SEM_LEVEL)
                
                # Append the tuple of actions for this UAV
                actions.append((vel_action, comm_mode_action, sem_level_action))
            # ---------------------------------------------

            # Apply the randomly generated actions to the environment
            # Note: The 'actions' structure here might not match what the
            # H-MAPPO-compatible uav_env.py expects from its 'low_level' space.
            next_states, reward, done, _ = env.step(actions)
            
            # Accumulate the (likely global) reward
            total_reward += reward
            time_step += 1
            
            # Log progress every 100 steps
            if time_step % 100 == 0:
                print(f"Time Step: {time_step}, Cumulative Reward: {total_reward:.2f}")

        # Episode finished
        print(f"Episode Finished after {time_step} steps. Final Reward: {total_reward:.2f}")
        
        # Print the final energy of the first UAV
        # Note: The `entities.py` file defines this as `uav.energy`, 
        # not `uav.rem_energy`, indicating this script may be out of sync.
        print(f"UAV 0 Final Energy: {env.uavs[0].rem_energy:.2f}")


if __name__ == '__main__':
    main()