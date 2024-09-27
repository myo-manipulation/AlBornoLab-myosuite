import gym
import myosuite
import deprl

env = gym.make("myoLegWalk-v0")
policy = deprl.load_baseline(env)

N = 5 # number of episodes
for i in range(N):
    obs = env.reset()
    while True:
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        env.mj_render()
        if done:
            break