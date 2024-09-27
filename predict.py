import myosuite
import mujoco
import gym
import numpy as np
import os
from IPython.display import HTML
from base64 import b64encode
from typing import Optional, Union, List, Tuple, Dict, Literal
from tqdm import tqdm
import matplotlib.pyplot as plt
from myosuite.envs.myo.myobase import register_env_with_variants
from myosuite.envs.myo.base_v0 import BaseV0
import collections
import warnings
import skvideo.io

from stable_baselines3.common import env_checker
from stable_baselines3 import PPO



def write_video(video_path: str, frames: List[np.ndarray], **kwargs):
    dir_path = os.path.dirname(video_path)
    os.makedirs(dir_path, exist_ok=True)
    skvideo.io.vwrite(video_path, np.asarray(frames),**kwargs)

def show_video(
    video_path: str,
    video_width: Optional[int] = 720
):

  video_file = open(video_path, "r+b").read()

  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  return HTML(f"""<video autoplay width={video_width} controls><source src="{video_url}"></video>""")


env = gym.make('CenterReachOut-v0')

obs = env.reset()
print(obs)

model = PPO.load('myosuite/agents/outputs/2024-09-17/00-23-57/CenterReachOut-v0_PPO_model.zip')

frames = []
for _ in range(500):
   action, _ = model.predict(obs, deterministic=True)
   obs, reward, done, info = env.step(action=action)
#    print(env.sim.renderer.render_offscreen())
   frames.append(env.sim.renderer.render_offscreen(camera_id=0))

write_video('video/preview.mp4', frames)
# model.

# for _ in range(1000): 
#     env.mj_render()
#     env.step(env.action_space)
# env.close()
# import time

# time.sleep(5)