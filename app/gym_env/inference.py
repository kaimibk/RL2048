from env import Base2048Env
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.dqn import DQNTrainer as trainer
import time

ray.init()
env = Base2048Env()

def gym_env(env_config):
    return Base2048Env()

register_env("2048-v0", gym_env)

agent = trainer(env="2048-v0")
agent.restore("/home/ray/ray_results/2048-v0-DQN/DQN_2048-v0_21666_00000_0_2021-04-22_13-12-58/checkpoint_3000/checkpoint-3000")

state = env.reset()

sum_reward = 0

for step in range(10000):
    # time.sleep(0.01)

    env.render()

    action = agent.compute_action(state)
    state, reward, done, info = env.step(action)
    sum_reward += reward
    print(f"STEP {step} \t ACTION: {action} \t REWARD: {sum_reward}")

    if done:
        print("DONE")
        break


