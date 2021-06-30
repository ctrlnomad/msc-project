from gym.envs.registration import register

register(
    id='MetaCausalBandits-v0',
    entry_point='causal_env.envs:MetaCausalBanditsEnv',
)