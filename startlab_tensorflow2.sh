#!/bin/bash
PORT=8888
NOTEBOOKS=/home/alessioarcara/CNAF/anomaly-detection_failure-prediction-CNAF-INFN/tirocinio/notebooks

docker run -d \
   --gpus all \
   --name test-tf-gpu \
   -p ${PORT}:8888 \
   -v "${NOTEBOOKS}":/home/jovyan/notebooks \
   --user "$(id -u):$(id -g)" \
   tensorflow/tensorflow:latest-gpu-jupyter
