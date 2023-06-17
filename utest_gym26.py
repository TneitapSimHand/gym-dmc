import gym 
import sys
import numpy as np 
import argparse
import gym_dmc # env rigisteration
from dm_control.suite import ALL_TASKS

assert gym.__version__ == '0.26.1', "Unmatched Gym version"
assert gym_dmc.DMC_IS_REGISTERED, "dmc env has not been successfully registered"
    
print("========== Gym Standard Env ==========")
print(*(gym.envs.registry.keys()), sep="\t")

print("========== DMC mujoco Env ==========")
print(*ALL_TASKS, sep="\t")

if __name__ == "__main__":

    # gym naive (openAI)
    # env_name = "CartPole-v1" # CartPole-v1 
    # env = gym.make(env_name, render_mode = "human") # BipedalWalker-v3, BipedalWalkerHardcore-v3

    # py_mujoco (openAI)
    # env_name = "HalfCheetah-v4" # Humanoid-v3, HalfCheetah-v4, Ant-v3
    # env = gym.make(env_name, render_mode = "human")
    
    # dmc + mujoco (Deepmind)
    env_name = "Quadruped-walk-v1" #"Walker-stand-v1"
    env = gym.make(env_name, height=480, width=640, frame_skip=4, space_dtype=np.float32)
    print("reorder observations: ", env.env.env.observation_space.keys())

    test_episode = 20
    for epi_i in range(test_episode):
        print("episode %02d"%(epi_i))
        step_count = 0; done = False
        obs, infos = env.reset()
        while not done: 
            action = np.random.randn(*env.action_space.shape) # random policy
            obs, reward, done, trunc, infos = env.step(action)
            env.render()
            step_count +=1
            if done: print("ctrl step accu: ", step_count)