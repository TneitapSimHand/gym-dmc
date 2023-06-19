import numpy as np 
import mujoco
from dm_control import suite
from dm_control import viewer
import cv2 

RENDER_MODE = "human_freeview"
# RENDER_MODE = "human_fixedview"

class Policy:
    def __init__(self, env):
        self.env = env

    def get_random_act(self, time_step):
        del time_step  # Unused.
        return np.random.uniform(low=self.env.action_spec().minimum,
                                high=self.env.action_spec().maximum,
                                size=self.env.action_spec().shape) # shape (21, )

if __name__ == "__main__":


    env = suite.load("walker", "stand")
        
    poli = Policy(env)

    time_step_data = env.reset()
    if RENDER_MODE == "human_freeview":
        viewer.launch(env, policy=poli.get_random_act)

    elif RENDER_MODE == "human_fixedview":
        succeed_case_count = 0
        epoch_time = 0
        print("time interval for simulation: ",env.physics.timestep())
        print("time interval for control: ",  env.control_timestep())

        while True:
            # obs_Arr = np.concatenate([val.reshape(-1) for key, val in time_step_data.observation.items()], dtype=np.float32)
            # print("[%.3f]obs: "%(env.physics.time()), obs_Arr)
            action=poli.get_random_act(time_step_data) # np.random.randn(*env.action_spec().shape) # random policy

            # action = np.array([-1,1,-1,1,-1,1]).astype(np.float32)
            time_step_data = env.step(action)

            camera0_frame = env.physics.render(camera_id=0, height=480, width=640)
            camera1_frame = env.physics.render(camera_id=1, height=480, width=640)
            # print(camera0.shape)
            cv2.imshow("cam0", camera0_frame[:,:,::-1])
            cv2.imshow("cam1", camera1_frame[:,:,::-1])
            cv2.waitKey(1)

            reward = time_step_data.reward
            if reward is None: reward = 0 
            if abs(reward - 1) < 1e-3:
                print("task succeed! reward = ", reward)
                succeed_case_count +=1
                time_step_data = env.reset()

            if epoch_time > 1000:
                print("time out! reward = ", reward)
                time_step_data = env.reset()
                epoch_time = 0

            next_obs_Arr = np.concatenate([val.reshape(-1) for key, val in time_step_data.observation.items()], dtype=np.float32)
            obs_Arr = next_obs_Arr
            epoch_time +=1 
