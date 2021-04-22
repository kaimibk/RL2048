## Reinforcement Learning on 2048

The goal of this repository is to use the [RLlib](https://github.com/ray-project/ray#rllib-quick-start) library from [Ray](https://github.com/ray-project/ray) to solve the game of [2048](https://play2048.co/). The primary focus of this repository is not to dive into the RL theory and application, rather to showcase the tools that come with Ray for monitoring and scaling your training in a multi-container [Docker](https://www.docker.com/) environment.

The 2048 environment used in this repository comes from [HERE](https://github.com/activatedgeek/gym-2048).

### TODOs:
- [] Render the game board for inspection.
- [] Use Curriculum learning to periodically make the game board larger or the "2048" target higher.
- [] Build sample grafana dashboard .
- [] Create/mount training config yaml file.

## How to use

Assuming you have docker-compose installed:
1. Stand up the container stack using `docker-compose up`. If this works, the Ray webservices and Tensorboard should be active.
2. In another terminal, run `docker-compose exec ray-rllib python gym_env/train.py` to start the training process. Alternatively, to test rendering run `docker-compose exec ray-rllib python gym_env/basic.py` to visualize random actions.
3. When you are done, use `ctrl+c` to stop the process (both the training and stack). And use `docker-compose down` to spin down the project containers.

Additionally, visit the web services described below to monitor the training process.

## Docker-Compose Services

Upon starting this stack, the following web services will be made availble:
- [Ray Dashboard](https://docs.ray.io/en/master/ray-dashboard.html): `localhost:8265`
    - Used to monitor Ray clusters, including error logs, hardware utilization, etc.
- [Ray Monitoring](https://docs.ray.io/en/master/ray-metrics.html): `localhost:8080`
    - A Prometheus endpoint containing various metrics captured by Ray.
- [Prometheus](https://prometheus.io/): `localhost:8080/metrics`
    - Prometheus endpoint with some default metrics
- [Grafana](https://grafana.com/): `localhost:3000`
    - A monitoring tool that parses the Prometheus endpoints and visualizes their metrics.
- [Tensorboard](https://www.tensorflow.org/tensorboard): `localhost:6006`
    - Visualization tool to monitoring training progress and experimentation.