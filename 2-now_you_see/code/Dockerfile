## Client: Docker Engine - Community Version 19.03.5
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.7
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         curl \
         wget \
         ca-certificates && rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy scipy mkl mkl-include ninja cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN mkdir -p /root/.cache/torch/transformers
RUN mkdir -p /work
RUN mkdir -p /work/models

COPY . /work/
WORKDIR /work

RUN pip install --no-cache-dir -r requirements.txt

## GEt my home trained models
RUN wget --tries=0 --read-timeout=10 --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zM2UE16vwwCbRigps71lVKu7RgdH7zVn' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zM2UE16vwwCbRigps71lVKu7RgdH7zVn" -O models8935.tar.gz && rm -rf /tmp/cookies.txt

## GEt base pretrained models (tar -czvf transformers.tar.gz transformers/)
RUN wget --tries=0 --read-timeout=10 --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qoHIu5fJfg_HQitfBgALV6ThF-us2HtY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1qoHIu5fJfg_HQitfBgALV6ThF-us2HtY" -O transformers.tar.gz && rm -rf /tmp/cookies.txt

## unpack base torch models to default path
RUN tar -zxvf /work/transformers.tar.gz -C /root/.cache/torch

## and unpack my models
RUN tar -zxvf /work/models8935.tar.gz -C /work

RUN chmod 777 test.sh
RUN chmod 777 train.sh
ENTRYPOINT ["/bin/bash"]
#time ./train.sh path/to/train/and/test/csv/files
#time ./test.sh path/to/train/and/test/csv/files path/to/out/dir/for/solution.csv