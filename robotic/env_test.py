import gymnasium as gym
import panda_gym

import time

if __name__ == "__main__":
    done = False
    env = gym.make('PandaReach-v3', render_mode='human')
    obs, info = env.reset()
    
    for _ in range(1000): 
        action = env.action_space.sample()
        obs_, rew, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info  = env.reset()

        # time.sleep(.05)
    env.close()



