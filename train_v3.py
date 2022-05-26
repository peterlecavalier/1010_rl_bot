"""
Modified from:
https://github.com/ray-project/ray/blob/7f1bacc7dc9caf6d0ec042e39499bbf1d9a7d065/rllib/examples/action_masking.py
"""

from gym.spaces import Box, Discrete
import ray
from ray.rllib.agents import ppo
#from ray.rllib.examples.env.action_mask_env import ActionMaskEnv
from environment import game_1010_v1
from ray.rllib.examples.models.action_mask_model import ActionMaskModel
from datetime import datetime
import matplotlib.pyplot as plt


TRAINING_ITER_STOP = 100000
EPISODE_LEN_MEAN = 1000000
EPISODE_REWARD_MEAN = 10000000

ray.init()

# main part: configure the ActionMaskEnv and ActionMaskModel
config = {
    "env": game_1010_v1,
    'gamma': 0.99, 
    'lr': 5e-05,
    "env_config": {
        "action_space": Discrete(300),
        "observation_space": Box(0, 20, (103,), dtype=int),
    },
    # the ActionMaskModel retrieves the invalid actions and avoids them
    "model": {
        "custom_model": ActionMaskModel,
        # keep masking enabled
        "custom_model_config": {"no_masking": False},
        "fcnet_hiddens": [512, 512, 512],
        #"fcnet_activation": "relu",
    },
    "num_gpus": 0,
    "num_workers": 4,
    # max steps in any one episode is 1000000
    "horizon": EPISODE_LEN_MEAN,
}
# config["tf_session_args"]["intra_op_parallelism_threads"] = 64
# config["tf_session_args"]["inter_op_parallelism_threads"] = 64
# config["local_tf_session_args"]["intra_op_parallelism_threads"] = 64
# config["local_tf_session_args"]["inter_op_parallelism_threads"] = 64
#config["num_envs_per_worker"] = 10
#config["vf_clip_param"] = 10000
#config['create_env_on_driver'] = True
#config['render_env'] = True


# stop dict is used if training with Ray tune
stop = {
    "training_iteration": TRAINING_ITER_STOP,
    "episode_reward_mean": EPISODE_REWARD_MEAN,
}

# manual training loop (no Ray tune)
ppo_config = ppo.DEFAULT_CONFIG.copy()
ppo_config.update(config)
trainer = ppo.PPOTrainer(config=ppo_config, env=game_1010_v1)

policy = trainer.get_policy()
policy.model.internal_model.base_model.summary()

time = str(datetime.now())
time = time[:10] + "_" + time[11:-7]
time = time.replace(":", "_")

f = open(f"./training_output_{time}.txt", "w")
f.close()

total_lengths = []
total_rewards = []

# run manual training loop and print results after each iteration
for idx in range(TRAINING_ITER_STOP):
    results = trainer.train()
    print(f"-----EPISODE {idx}-----")
    print(f"Reward min/max/mean = {results['episode_reward_min']}/{results['episode_reward_max']}/{results['episode_reward_mean']}")
    lengths = results["hist_stats"]['episode_lengths']
    print(f"Length min/max/mean = {min(lengths)}/{max(lengths)}/{results['episode_len_mean']}")

    total_lengths.append(results['episode_len_mean'])
    total_rewards.append(results['episode_reward_mean'])

    f = open(f"./training_output_{time}.txt", "a")
    f.write(f"-----EPISODE {idx}-----\n")
    f.write(f"Reward min/max/mean = {results['episode_reward_min']}/{results['episode_reward_max']}/{results['episode_reward_mean']}\n")
    f.write(f"Length min/max/mean = {min(lengths)}/{max(lengths)}/{results['episode_len_mean']}")
    f.close()

    # Save the checkpoint every 10 iterations
    if idx % 10 == 0:
        trainer.save(f"./checkpoints/{time}")

    # Create a plot of rewards and lengths
    fig, axs = plt.subplots(2, 1, sharey=False, sharex=True)
    axs[0].plot(total_rewards)
    axs[1].plot(total_lengths)
    axs[0].set_ylabel('Average episode reward')
    axs[1].set_ylabel('Average episode length')
    axs[1].set_xlabel('Iteration #')

    # Save the plot
    plt.savefig(f"./checkpoints/{time}/summary_plots.png", bbox_inches='tight')
    plt.close()

    # stop training if the target train steps or reward are reached
    if (
        results["episode_reward_mean"] >= EPISODE_REWARD_MEAN
        or results['episode_len_mean'] >= EPISODE_LEN_MEAN
    ):
        # if we haven't already saved this time, save
        if idx % 10 != 0:
            trainer.save(f"./checkpoints/{time}")
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