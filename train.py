# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:28:44 2022

@author: Peter
"""

import ray
from ray.rllib.agents import ppo
from environment import game_1010_v0

config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 3
config["horizon"] = 1000000
"""
# Configure the algorithm.
config = {
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 3,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "tf",
    
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "vf_share_layers": True
    },
    # Set up a separate evaluation worker set for the
    # `trainer.evaluate()` call after training (see below).
    "evaluation_num_workers": 0,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": True,
    },
    "num_gpus": 0
}
"""
ray.init()
# Create our RLlib Trainer.
trainer = ppo.PPOTrainer(env=game_1010_v0, config=config)

# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.
for idx in range(100):
    results = trainer.train()
    print(f"-----EPISODE {idx}-----")
    print(f"Reward min/max/mean = {results['episode_reward_min']}/{results['episode_reward_max']}/{results['episode_reward_mean']}")
    print(f"Length mean = {results['episode_len_mean']}")

# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
trainer.evaluate()