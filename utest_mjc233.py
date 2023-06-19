'''
mujoco: pure physics; (sim_timestep)
dm_control: deepmind wrapped Env Class (control_timestep + sim_timestep)
gym: most popular Env Class (control_timestep + sim_timestep)
'''
import os 
import numpy as np 
import mujoco
import dm_control.suite as suite
import cv2 

assert mujoco.__version__ == '2.3.3', "Unmatched mujoco version"


if __name__ == "__main__":

    filename = os.path.join(suite.__path__[0], "walker.xml")
    assert os.path.exists(filename), "invalid mjcf dir"
    model = mujoco.MjModel.from_xml_path(filename) # static data
    data = mujoco.MjData(model) # dynamic data

    data.qpos = np.zeros(model.nq)
    data.qvel = np.zeros(model.nv)

    renderer = mujoco.Renderer(model, height=480, width=640)
    while (data.time < 10):
        data.ctrl = np.ones(model.nu)
        mujoco.mj_step(model, data)
        renderer.update_scene(data)
        img = renderer.render()
        cv2.imshow("img", img[:,:,::-1])
        cv2.waitKey(1)
        print("[%.3f]obv qvel: "%(data.time), data.xpos)
