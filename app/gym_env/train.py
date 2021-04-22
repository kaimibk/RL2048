from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from env import Base2048Env

def gym_env(env_config):
    return Base2048Env()

register_env("2048-v0", gym_env)

tune.run(
    PPOTrainer,
    config={
        "env": "2048-v0",
    },
)