FROM rayproject/ray-ml:latest-gpu
LABEL maintainer="kaimibk@gmail.com"

ENV PYTHONPATH="${PYTHONPATH}:/app/"

RUN sudo apt-get update
RUN sudo apt-get install -y libx11-dev
RUN sudo apt-get install -y python3-tk

COPY ./gym_env /app/gym_env

RUN mkdir -p ~/ray_results

CMD ray start \
    --head \
    --port 6379 \
    --dashboard-port 8265 --dashboard-host 0.0.0.0 \
    --metrics-export-port=8080 \
    && tensorboard --logdir=~/ray_results --host 0.0.0.0 --port 6006
