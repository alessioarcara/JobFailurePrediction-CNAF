#!/bin/bash
PORT=8888
NOTEBOOKS=/home/alessioarcara/CNAF/anomaly-detection_failure-prediction-CNAF-INFN/tirocinio/notebooks

docker run -d \
   -p ${PORT}:8888 \
   -v "${NOTEBOOKS}":/home/jovyan/notebooks \
   --gpus 'all' \
   --user "$(id -u)" --group-add users \
jupyter/tensorflow-notebook

# docker run -d \
#    -e GRANT_SUDO=yes \
#    -e NB_UID=`id -u` \
#    -e NB_GID=`id -g` \
#    -e NB_USER="sdp" \
#    --user root \
#    --group-add users \
#    -p ${PORT}:8888 \
#    --gpus all \
#    -v "${DATA}":/home/jovyan/data \
#    -v "${NOTEBOOKS}":/home/jovyan/work \
#    jupyter/tensorflow-notebook

#   -e GEN_CERT=yes
#farm-registry.cr.cnaf.infn.it/sdp/mlsdpjl:v1
# jupyter/datascience-notebook
