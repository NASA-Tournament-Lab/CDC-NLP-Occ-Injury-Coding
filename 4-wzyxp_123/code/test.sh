#!/bin/sh

apt-get update -y
apt-get upgrade -y
apt-get install wget -y
apt-get install zip -y

python -m pip install -r /work/code/roberta/pytorch_transformers/transformers-master/requirements.txt
python -m pip install tensorboard
python -m pip install sklearn

mkdir /work/workspace/bert_data

python /work/code/pre/pre_one.py /work/workspace/eventcounts.tsv /data/test.csv /work/workspace/bert_data/dev.tsv

#download trained models
mkdir /work/models/
cd /work/models/

mkdir /work/models/1/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1CUBMJxbOgEbsYAfeCEFbstwdJwUrA61G' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1CUBMJxbOgEbsYAfeCEFbstwdJwUrA61G" -O /work/models/1/1.zip && rm -rf /tmp/cookies.txt
cd 1/
unzip 1.zip
cp /work/workspace/vocabs/* ./
cd ..

mkdir /work/models/2/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1CigopROPwOhw_SZzyZbO5dc3fhmnVvkg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1CigopROPwOhw_SZzyZbO5dc3fhmnVvkg" -O /work/models/2/2.zip && rm -rf /tmp/cookies.txt
cd 2/
unzip 2.zip
cp /work/workspace/vocabs/* ./
cd ..

mkdir /work/models/3/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xA-5bemlB9EfRnW9pkXPn_lv457h5ykS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xA-5bemlB9EfRnW9pkXPn_lv457h5ykS" -O /work/models/3/3.zip && rm -rf /tmp/cookies.txt
cd 3/
unzip 3.zip
cp /work/workspace/vocabs/* ./
cd ..


mkdir /work/models/4/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=17tihcDvIdbrv_-9fg4xESSjqA0UOtuqF' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=17tihcDvIdbrv_-9fg4xESSjqA0UOtuqF" -O /work/models/4/4.zip && rm -rf /tmp/cookies.txt
cd 4/
unzip 4.zip
cp /work/workspace/vocabs/* ./
cd ..

mkdir /work/models/5/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1x4Oixl3LSbnJ6uCYlXMoRCrnrP8V7oKY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1x4Oixl3LSbnJ6uCYlXMoRCrnrP8V7oKY" -O /work/models/5/5.zip && rm -rf /tmp/cookies.txt
cd 5/
unzip 5.zip
cp /work/workspace/vocabs/* ./
cd ..

mkdir /work/models/6/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10RFeTmkwe0E5q5-KfaMJ7EAEEadG_n6J' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10RFeTmkwe0E5q5-KfaMJ7EAEEadG_n6J" -O /work/models/6/6.zip && rm -rf /tmp/cookies.txt
cd 6/
unzip 6.zip
cp /work/workspace/vocabs/* ./
cd ..

mkdir /work/models/7/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zQrycU6Rl-q2-tNrL3OiN4YKU-rplgAL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zQrycU6Rl-q2-tNrL3OiN4YKU-rplgAL" -O /work/models/7/7.zip && rm -rf /tmp/cookies.txt
cd 7/
unzip 7.zip
cp /work/workspace/vocabs/* ./
cd ..

mkdir /work/models/8/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=167Exhx-kk8ayzPxbb9eHJ4aLM4EQauKJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=167Exhx-kk8ayzPxbb9eHJ4aLM4EQauKJ" -O /work/models/8/8.zip && rm -rf /tmp/cookies.txt
cd 8/
unzip 8.zip
cp /work/workspace/vocabs/* ./
cd ..

python /work/code/roberta/pytorch_transformers/transformers-master/examples/run_glue.py \
 --model_type=roberta \
 --model_name_or_path=roberta-large \
 --task_name=cdc \
 --do_eval \
 --data_dir=/work/workspace/bert_data/ \
 --max_seq_length=50  \
 --learning_rate=2e-5   --num_train_epochs=3.0 --per_gpu_eval_batch_size=32  --output_dir /work/models/1/

python /work/code/roberta/pytorch_transformers/transformers-master/examples/run_glue.py \
 --model_type=roberta \
 --model_name_or_path=roberta-large \
 --task_name=cdc \
 --do_eval \
 --data_dir=/work/workspace/bert_data/ \
 --max_seq_length=50  \
 --learning_rate=2e-5   --num_train_epochs=3.0 --per_gpu_eval_batch_size=32  --output_dir /work/models/2/

python /work/code/roberta/pytorch_transformers/transformers-master/examples/run_glue.py \
 --model_type=roberta \
 --model_name_or_path=roberta-large \
 --task_name=cdc \
 --do_eval \
 --data_dir=/work/workspace/bert_data/ \
 --max_seq_length=50  \
 --learning_rate=2e-5   --num_train_epochs=3.0 --per_gpu_eval_batch_size=32  --output_dir /work/models/3/

python /work/code/roberta/pytorch_transformers/transformers-master/examples/run_glue.py \
 --model_type=roberta \
 --model_name_or_path=roberta-large \
 --task_name=cdc \
 --do_eval \
 --data_dir=/work/workspace/bert_data/ \
 --max_seq_length=50  \
 --learning_rate=2e-5   --num_train_epochs=3.0 --per_gpu_eval_batch_size=32  --output_dir /work/models/4/


python /work/code/roberta/pytorch_transformers/transformers-master/examples/run_glue.py \
 --model_type=roberta \
 --model_name_or_path=roberta-large \
 --task_name=cdc \
 --do_eval \
 --data_dir=/work/workspace/bert_data/ \
 --max_seq_length=50  \
 --learning_rate=2e-5   --num_train_epochs=3.0 --per_gpu_eval_batch_size=32  --output_dir /work/models/5/

 python /work/code/roberta/pytorch_transformers/transformers-master/examples/run_glue.py \
 --model_type=roberta \
 --model_name_or_path=roberta-large \
 --task_name=cdc \
 --do_eval \
 --data_dir=/work/workspace/bert_data/ \
 --max_seq_length=50  \
 --learning_rate=2e-5   --num_train_epochs=3.0 --per_gpu_eval_batch_size=32  --output_dir /work/models/6/

 python /work/code/roberta/pytorch_transformers/transformers-master/examples/run_glue.py \
 --model_type=roberta \
 --model_name_or_path=roberta-large \
 --task_name=cdc \
 --do_eval \
 --data_dir=/work/workspace/bert_data/ \
 --max_seq_length=50  \
 --learning_rate=2e-5   --num_train_epochs=3.0 --per_gpu_eval_batch_size=32  --output_dir /work/models/7/

 python /work/code/roberta/pytorch_transformers/transformers-master/examples/run_glue.py \
 --model_type=roberta \
 --model_name_or_path=roberta-large \
 --task_name=cdc \
 --do_eval \
 --data_dir=/work/workspace/bert_data/ \
 --max_seq_length=50  \
 --learning_rate=2e-5   --num_train_epochs=3.0 --per_gpu_eval_batch_size=32  --output_dir /work/models/8/


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

