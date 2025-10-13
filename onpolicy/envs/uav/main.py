import numpy as np
from .uav_env import UAVEnv
from .parameters import *

def main():
    env = UAVEnv()
    
    # Run for a few episodes
    for episode in range(3):
        states = env.reset()
        done = False
        total_reward = 0
        time_step = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        
        while not done:
            # --- Placeholder for Agent's Action Selection ---
            # For now, generate random actions for each UAV
            actions = []
            for i in range(N_UAVS):
                # Random continuous action for velocity
                speed = np.random.uniform(0, UAV_MAX_SPEED)
                angle = np.random.uniform(0, 2 * np.pi)
                vel_action = np.array([speed * np.cos(angle), speed * np.sin(angle)])
                
                # Random discrete actions
                comm_mode_action = np.random.randint(ACTION_COMM_MODE)
                sem_level_action = np.random.randint(ACTION_SEM_LEVEL)
                
                actions.append((vel_action, comm_mode_action, sem_level_action))
            # ---------------------------------------------

            next_states, reward, done, _ = env.step(actions)
            
            total_reward += reward
            time_step += 1
            
            if time_step % 100 == 0:
                print(f"Time Step: {time_step}, Cumulative Reward: {total_reward:.2f}")

        print(f"Episode Finished after {time_step} steps. Final Reward: {total_reward:.2f}")
        # Print final status of a UAV
        print(f"UAV 0 Final Energy: {env.uavs[0].rem_energy:.2f}")


if __name__ == '__main__':
    main()