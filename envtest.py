import gym
import mtxenv
env = gym.make('mtxenv/GridWorld-v0',size=5)
# env = gym.make('PongNoFrameskip-v4')
print(env.reset())
env.step(3)
