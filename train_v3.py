"""
Modified from:
https://github.com/ray-project/ray/blob/7f1bacc7dc9caf6d0ec042e39499bbf1d9a7d065/rllib/examples/action_masking.py
"""

from gym.spaces import Box, Discrete, Tuple
import ray
from ray import tune
from ray.rllib.agents import ppo
#from ray.rllib.examples.env.action_mask_env import ActionMaskEnv
from environment import game_1010_v1
from ray.rllib.examples.models.action_mask_model import ActionMaskModel
from ray.tune.logger import pretty_print


TRAINING_ITER_STOP = 100000
TIMESTEPS_TOTAL = 1000000
EPISODE_REWARD_MEAN = 10000000

ray.init()#num_cpus=0, local_mode=True)

# main part: configure the ActionMaskEnv and ActionMaskModel
config = {
    # random env with 100 discrete actions and 5x [-1,1] observations
    # some actions are declared invalid and lead to errors
    "env": game_1010_v1,
    "env_config": {
        "action_space": Discrete(300),
        "observation_space": Box(0, 20, (103,), dtype=int),
    },
    # the ActionMaskModel retrieves the invalid actions and avoids them
    "model": {
        "custom_model": ActionMaskModel,
        # disable action masking according to CLI
        "custom_model_config": {"no_masking": False},
    },
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": 0,
    # Run with tracing enabled for tfe/tf2?
    #"eager_tracing": args.eager_tracing,
    "horizon": 1000000
}

stop = {
    "training_iteration": TRAINING_ITER_STOP,
    "timesteps_total": TIMESTEPS_TOTAL,
    "episode_reward_mean": EPISODE_REWARD_MEAN,
}

# manual training loop (no Ray tune)
ppo_config = ppo.DEFAULT_CONFIG.copy()
ppo_config.update(config)
trainer = ppo.PPOTrainer(config=ppo_config, env=game_1010_v1)
# run manual training loop and print results after each iteration
for idx in range(TRAINING_ITER_STOP):
    results = trainer.train()
    print(f"-----EPISODE {idx}-----")
    print(f"Reward min/max/mean = {results['episode_reward_min']}/{results['episode_reward_max']}/{results['episode_reward_mean']}")
    lengths = results["hist_stats"]['episode_lengths']
    print(f"Length min/max/mean = {min(lengths)}/{max(lengths)}/{results['episode_len_mean']}")

    # stop training if the target train steps or reward are reached
    if (
        results["episode_reward_mean"] >= EPISODE_REWARD_MEAN
    ):
        break

# manual test loop
print("Finished training. Running manual test/inference loop.")
# prepare environment with max 10 steps
config["env_config"]["max_episode_len"] = 10
env = game_1010_v1(config["env_config"])
obs = env.reset()
done = False
# run one iteration until done
print(f"ActionMaskEnv with {config['env_config']}")
while not done:
    action = trainer.compute_single_action(obs)
    next_obs, reward, done, _ = env.step(action)
    # observations contain original observations and the action mask
    # reward is random and irrelevant here and therefore not printed
    print(f"Obs: {obs}, Action: {action}")
    obs = next_obs

'''
# run with tune for auto trainer creation, stopping, TensorBoard, etc.
else:
    results = tune.run(args.run, config=config, stop=stop, verbose=2)
'''
print("Finished successfully without selecting invalid actions.")
ray.shutdown()