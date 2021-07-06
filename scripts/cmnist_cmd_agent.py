import gym

from dataclasses import dataclass
from argparse_dataclass import ArgumentParser

from causal_env.envs import CausalMnistBanditsConfig

@dataclass
class Options(CausalMnistBanditsConfig):
  seed: int = 5000
  debug: bool = False


if __name__ == '__main__':
  parser = ArgumentParser(Options)
  config = parser.parse_args()

  env = gym.make('CausalMnistBanditsEnv-v0')
  env.init(config)

  timestep = env.reset()

  done = False
  while not done:
    print(f'context: {timestep.context}')
    action = input('please input your action: \n>>>')

    try:
      action = int(action)
      assert 0 <= action <= config.num_arms, 'not a valid input'
    except:
      continue

    timestep = env.step(action)
    print(f'your reward is: [{timestep.reward:.3f}]')
    print(f'treatments: {timestep.treatments}')
    
  best_action = input('what is the best action? \n >>>')
  print(f'the individual treatment effects were: {env.digit_ITEs} argmax is {env.digit_ITEs.argmax()}')
  #print(f'regret is {env.calculate_regret(best_action)}')

