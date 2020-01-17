## This docker is supposed to be run on a GPU, otherwise testing/training will take for-ever.
##
## Storage estimates on /work/: 75 GB

FROM ubuntu:18.04


## Install General Requirements
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        cmake \
        git \
        vim \
        wget \
        nano \
        python3-dev \
        python3-venv \
        python3-pip \
        software-properties-common

WORKDIR /work

# copy entire directory where docker file is into docker container at /work
COPY . /work/

RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install --upgrade numpy sentencepiece scipy pandas tensorflow==1.15

RUN chmod 777 download_models.sh
RUN chmod 777 train.sh
RUN chmod 777 test.sh

RUN /bin/bash ./download_models.sh
