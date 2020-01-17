#!/bin/sh

apt-get update -y
apt-get upgrade -y
apt-get install wget -y
apt-get install zip -y

python -m pip install -r /work/code/roberta/pytorch_transformers/transformers-master/requirements.txt
python -m pip install tensorboard
python -m pip install sklearn

mkdir /work/workspace/bert_data
python /work/code/pre/pre_one.py /work/workspace/eventcounts.tsv /data/train.csv /work/workspace/bert_data/train.tsv
python /work/code/pre/pre_one.py /work/workspace/eventcounts.tsv /data/test.csv /work/workspace/bert_data/dev.tsv

cd /work/code

python /work/code/roberta/pytorch_transformers/transformers-master/examples/run_glue.py \
 --model_type=roberta \
 --model_name_or_path=roberta-large \
 --task_name=cdc \
 --do_train \
 --do_eval \
 --data_dir=/work/workspace/bert_data/ \
 --max_seq_length=50  \
 --learning_rate=2e-5   --num_train_epochs=3.0  --save_steps=4000 --output_dir /work/models/1/ --seed=98987

python /work/code/roberta/pytorch_transformers/transformers-master/examples/run_glue.py \
 --model_type=roberta \
 --model_name_or_path=roberta-large \
 --task_name=cdc \
 --do_train \
 --do_eval \
 --data_dir=/work/workspace/bert_data/ \
 --max_seq_length=50  \
 --learning_rate=2e-5   --num_train_epochs=3.0  --save_steps=4000 --output_dir /work/models/2/ --seed=665

python /work/code/roberta/pytorch_transformers/transformers-master/examples/run_glue.py \
 --model_type=roberta \
 --model_name_or_path=roberta-large \
 --task_name=cdc \
 --do_train \
 --do_eval \
 --data_dir=/work/workspace/bert_data/ \
 --max_seq_length=50  \
 --learning_rate=2e-5   --num_train_epochs=3.0  --save_steps=4000 --output_dir /work/models/3/ --seed=2874

python /work/code/roberta/pytorch_transformers/transformers-master/examples/run_glue.py \
 --model_type=roberta \
 --model_name_or_path=roberta-large \
 --task_name=cdc \
 --do_train \
 --do_eval \
 --data_dir=/work/workspace/bert_data/ \
 --max_seq_length=50  \
 --learning_rate=2e-5   --num_train_epochs=3.0  --save_steps=4000 --output_dir /work/models/4/ --seed=73

python /work/code/roberta/pytorch_transformers/transformers-master/examples/run_glue.py \
 --model_type=roberta \
 --model_name_or_path=roberta-large \
 --task_name=cdc \
 --do_train \
 --do_eval \
 --data_dir=/work/workspace/bert_data/ \
 --max_seq_length=50  \
 --learning_rate=2e-5   --num_train_epochs=3.0  --save_steps=4000 --output_dir /work/models/5/ --seed=330

python /work/code/roberta/pytorch_transformers/transformers-master/examples/run_glue.py \
 --model_type=roberta \
 --model_name_or_path=roberta-large \
 --task_name=cdc \
 --do_train \
 --do_eval \
 --data_dir=/work/workspace/bert_data/ \
 --max_seq_length=50  \
 --learning_rate=2e-5   --num_train_epochs=3.0  --save_steps=4000 --output_dir /work/models/6/ --seed=5

python /work/code/roberta/pytorch_transformers/transformers-master/examples/run_glue.py \
 --model_type=roberta \
 --model_name_or_path=roberta-large \
 --task_name=cdc \
 --do_train \
 --do_eval \
 --data_dir=/work/workspace/bert_data/ \
 --max_seq_length=50  \
 --learning_rate=2e-5   --num_train_epochs=3.0  --save_steps=4000 --output_dir /work/models/7/ --seed=15777

python /work/code/roberta/pytorch_transformers/transformers-master/examples/run_glue.py \
 --model_type=roberta \
 --model_name_or_path=roberta-large \
 --task_name=cdc \
 --do_train \
 --do_eval \
 --data_dir=/work/workspace/bert_data/ \
 --max_seq_length=50  \
 --learning_rate=2e-5   --num_train_epochs=3.0  --save_steps=4000 --output_dir /work/models/8/ --seed=9914

mkdir /work/workspace/trained_models

mkdir /work/workspace/trained_models/1
cp /work/models/1/pytorch_model.bin /work/workspace/trained_models/1
cp /work/models/1/config.json /work/workspace/trained_models/1
cp /work/models/1/training_args.bin /work/workspace/trained_models/1
cd /work/workspace/trained_models/1
zip -r 1.zip ./*
cd /work

mkdir /work/workspace/trained_models/2
cp /work/models/2/pytorch_model.bin /work/workspace/trained_models/2
cp /work/models/2/config.json /work/workspace/trained_models/2
cp /work/models/2/training_args.bin /work/workspace/trained_models/2
cd /work/workspace/trained_models/2
zip -r 2.zip ./*
cd /work

mkdir /work/workspace/trained_models/3
cp /work/models/3/pytorch_model.bin /work/workspace/trained_models/3
cp /work/models/3/config.json /work/workspace/trained_models/3
cp /work/models/3/training_args.bin /work/workspace/trained_models/3
cd /work/workspace/trained_models/3
zip -r 3.zip ./*
cd /work

mkdir /work/workspace/trained_models/4
cp /work/models/4/pytorch_model.bin /work/workspace/trained_models/4
cp /work/models/4/config.json /work/workspace/trained_models/4
cp /work/models/4/training_args.bin /work/workspace/trained_models/4
cd /work/workspace/trained_models/4
zip -r 4.zip ./*
cd /work

mkdir /work/workspace/trained_models/5
cp /work/models/5/pytorch_model.bin /work/workspace/trained_models/5
cp /work/models/5/config.json /work/workspace/trained_models/5
cp /work/models/5/training_args.bin /work/workspace/trained_models/5
cd /work/workspace/trained_models/5
zip -r 5.zip ./*
cd /work

mkdir /work/workspace/trained_models/6
cp /work/models/6/pytorch_model.bin /work/workspace/trained_models/6
cp /work/models/6/config.json /work/workspace/trained_models/6
cp /work/models/6/training_args.bin /work/workspace/trained_models/6
cd /work/workspace/trained_models/6
zip -r 6.zip ./*
cd /work

mkdir /work/workspace/trained_models/7
cp /work/models/7/pytorch_model.bin /work/workspace/trained_models/7
cp /work/models/7/config.json /work/workspace/trained_models/7
cp /work/models/7/training_args.bin /work/workspace/trained_models/7
cd /work/workspace/trained_models/7
zip -r 7.zip ./*
cd /work

mkdir /work/workspace/trained_models/8
cp /work/models/8/pytorch_model.bin /work/workspace/trained_models/8
cp /work/models/8/config.json /work/workspace/trained_models/8
cp /work/models/8/training_args.bin /work/workspace/trained_models/8
cd /work/workspace/trained_models/8
zip -r 8.zip ./*
cd /work

cp /work/workspace/trained_models/1/1.zip /wdata/
cp /work/workspace/trained_models/2/2.zip /wdata/
cp /work/workspace/trained_models/3/3.zip /wdata/
cp /work/workspace/trained_models/4/4.zip /wdata/
cp /work/workspace/trained_models/5/5.zip /wdata/
cp /work/workspace/trained_models/6/6.zip /wdata/
cp /work/workspace/trained_models/7/7.zip /wdata/
cp /work/workspace/trained_models/8/8.zip /wdata/

mkdir /work/workspace/results

mkdir /work/workspace/results/checkpoint-1
cp /work/models/1/dev.probs.res /work/workspace/results/checkpoint-1/dev.probs.res

mkdir /work/workspace/results/checkpoint-2
cp /work/models/2/dev.probs.res /work/workspace/results/checkpoint-2/dev.probs.res

mkdir /work/workspace/results/checkpoint-3
cp /work/models/3/dev.probs.res /work/workspace/results/checkpoint-3/dev.probs.res

mkdir /work/workspace/results/checkpoint-4
cp /work/models/4/dev.probs.res /work/workspace/results/checkpoint-4/dev.probs.res

mkdir /work/workspace/results/checkpoint-5
cp /work/models/5/dev.probs.res /work/workspace/results/checkpoint-5/dev.probs.res

mkdir /work/workspace/results/checkpoint-6
cp /work/models/6/dev.probs.res /work/workspace/results/checkpoint-6/dev.probs.res

mkdir /work/workspace/results/checkpoint-7
cp /work/models/7/dev.probs.res /work/workspace/results/checkpoint-7/dev.probs.res

mkdir /work/workspace/results/checkpoint-8
cp /work/models/8/dev.probs.res /work/workspace/results/checkpoint-8/dev.probs.res

python /work/code/post/ensemble_n_res_by_probs_softmax.py /work/workspace/results/ 8 /work/workspace/merged_dev.res

python /work/code/post/post_process_pytorch_res.py /work/workspace/eventcounts.tsv /work/workspace/merged_dev.res /work/workspace/solution.csv

cp /work/workspace/solution.csv /wdata/
