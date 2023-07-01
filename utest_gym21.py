import gym 
import sys
import numpy as np 
import argparse
import gym_dmc # env rigisteration
from dm_control import suite

def get_all_envs():
    for domain_name, task_name in suite.BENCHMARKING:
        print(domain_name, task_name)

assert gym.__version__ == '0.21.0', "Unmatched Gym version"
assert gym_dmc.DMC_IS_REGISTERED, "dmc env has not been successfully registered"

print("========== Gym Standard Env ==========")
print(*(gym.envs.registry.env_specs.keys()), sep="\t")

print("========== DMC mujoco Env ==========")
print(*suite.ALL_TASKS, sep="\t")

if __name__ == "__main__":

    
    # env_name = "BipedalWalkerHardcore-v3"
    # env = gym.make(env_name)
    
    env_name = "Humanoid-stand-v1" #"Walker-stand-v1" # "Quadruped-walk-v1"
    env = gym.make(env_name, height=480, width=640, space_dtype=np.float32, frame_skip=4)
    print("wrapper re-order observations: ", env.env.env.observation_space.keys())
    print("gym wrapped obs dim: ", env.observation_space.shape)
    print("gym wrapped act dim: ", env.action_space.shape)
    test_episode = 20
    for epi_i in range(test_episode):
        print("episode %02d"%(epi_i))
        step_count = 0; done = False
        obs = env.reset()
        while not done: 
            action = np.random.randn(*env.action_space.shape) # random policy
            obs, reward, done, infos = env.step(action) # gym21: 4 outs
            env.render("human")
            step_count +=1
            if done: print("ctrl step accu: ", step_count)