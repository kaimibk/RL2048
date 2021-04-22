from ray import tune
from ray.tune.registry import register_env
# from ray.rllib.agents.ppo import PPOTrainer as trainer
from ray.rllib.agents.dqn import DQNTrainer as trainer
from env import Base2048Env

def gym_env(env_config):
    return Base2048Env()

register_env("2048-v0", gym_env)

tune.run(
    trainer,
    config={
        "env": "2048-v0",
    },
    name="2048-v0-DQN",
    # resume=True,
    checkpoint_freq=1000,
    checkpoint_at_end=True
)