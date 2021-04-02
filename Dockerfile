FROM rayproject/ray-ml:latest-gpu
LABEL maintainer="kaimibk@gmail.com"

COPY ./GymEnv /app/GymEnv
COPY ./Training /app/Training
COPY ./scripts /app/scripts
COPY ./requirements /app/requirements

RUN mkdir -p ~/ray_results

CMD ray start \
    --head \
    --port 6379 \
    --dashboard-port 8265 --dashboard-host 0.0.0.0 \
    --metrics-export-port=8080 \
    && tensorboard --logdir=~/ray_results --host 0.0.0.0 --port 6006