from gym.envs.registration import register

register(
    id='mtxenv/GridWorld-v0',
    entry_point='mtxenv.envs:GridWorldEnv',
    max_episode_steps=300,
)