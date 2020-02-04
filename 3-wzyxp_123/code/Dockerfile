FROM pytorch/pytorch:latest

RUN mkdir /work
COPY . /work/
WORKDIR /work

RUN chmod 777 train.sh
RUN chmod 777 test.sh
