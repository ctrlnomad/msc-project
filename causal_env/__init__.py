from gym.envs.registration import register

register(
    id='CausalMnistBanditsEnv-v0',
    entry_point='causal_env.envs:CausalMnistBanditsEnv',
)
