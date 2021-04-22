from env import Base2048Env
import time

env = Base2048Env()

sum_reward = 0

for step in range(100):
    time.sleep(0.1)

    env.render()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    sum_reward += reward
    print(f"STEP {step} \t ACTION: {action} \t REWARD: {sum_reward}")

    if done:
        print("DONE")
        break


