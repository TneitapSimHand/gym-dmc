import gym 
import sys
import numpy as np 
import argparse
import gym_dmc # env rigisteration
from dm_control.suite import ALL_TASKS
print(*ALL_TASKS, sep="\n")

# handPhy23: gym==0.26.1
# stableBaseline3: gym==0.21.0

if __name__ == "__main__":

    assert gym.__version__ == '0.21.0', "Unmatched Gym version"
    assert gym_dmc.DMC_IS_REGISTERED, "dmc env has not been successfully registered"
    # env_name = "BipedalWalkerHardcore-v3"
    # env = gym.make(env_name)
    
    env_name = "Quadruped-walk-v1" #"Walker-stand-v1" # "Quadruped-walk-v1"
    env = gym.make(env_name, height=480, width=640, frame_skip=4, space_dtype=np.float32)

    test_episode = 20
    for epi_i in range(test_episode):
        print("episode %02d"%(epi_i))
        step_count = 0; done = False
        obs = env.reset()
        while not done: 
            action = np.random.randn(*env.action_space.shape) # random policy
            obs, reward, done, infos = env.step(action)
            env.render("human")
            step_count +=1
            if done: print("ctrl step accu: ", step_count)