#/bin/sh
test -d ~/workspace/data_tmp || mkdir data_tmp

python preprocessing.py --train_file=$1 --phase train --data_dir=~/data --output_dir data_tmp/

test -d ~/workspace/models && rm -rf  ~/workspace/models

test -d ~/workspace && mkdir ~/workspace
mkdir ~/workspace/models

python run_glue.py   --model_type roberta   --model_name_or_path roberta-large   --task_name CoLA   --do_train   --do_lower_case   --data_dir data_tmp/   --max_seq_length 50 --per_gpu_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 3.0  --output_dir ~/workspace/models/run1/ --overwrite_output_dir

python run_glue.py   --model_type roberta   --model_name_or_path roberta-large   --task_name CoLA   --do_train   --do_lower_case   --data_dir data_tmp/   --max_seq_length 50 --per_gpu_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 4.0  --output_dir ~/workspace/models/run2/ --overwrite_output_dir


python run_glue.py   --model_type roberta   --model_name_or_path roberta-large   --task_name CoLA   --do_train   --do_lower_case   --data_dir data_tmp/   --max_seq_length 40 --per_gpu_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 3.0  --output_dir ~/workspace/models/run3/ --overwrite_output_dir

python run_glue.py   --model_type roberta   --model_name_or_path roberta-base   --task_name CoLA   --do_train   --do_lower_case   --data_dir data_tmp/   --max_seq_length 50 --per_gpu_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 3.0  --output_dir ~/workspace/models/run4/ --overwrite_output_dir

python run_glue.py   --model_type roberta   --model_name_or_path roberta-base   --task_name CoLA   --do_train   --do_lower_case   --data_dir data_tmp/   --max_seq_length 50 --per_gpu_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 4.0  --output_dir ~/workspace/models/run5/ --overwrite_output_dir
