# %env MUJOCO_GL=egl
import mujoco

from IPython.display import HTML
from base64 import b64encode

def show_video(video_path, video_width = 400):

  video_file = open(video_path, "r+b").read()

  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  return HTML(f"""<video autoplay width={video_width} controls><source src="{video_url}"></video>""")


import myosuite
from myosuite.utils import gym
import skvideo.io
import numpy as np
import os

env = gym.make('myoElbowPose1D6MRandom-v0')


