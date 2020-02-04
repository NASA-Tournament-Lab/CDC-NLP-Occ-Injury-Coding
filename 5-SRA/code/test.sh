#/bin/sh
test -d ~/workspace/data_tmp || mkdir data_tmp

python preprocessing.py --test_file=$1 --phase test --output_dir data_tmp/

test -d ~/workspace/models || (echo "No models directory found" && exit -1)

python run_glue.py   --model_type roberta   --model_name_or_path ~/workspace/models/run1/    --task_name CoLA   --do_predict   --do_lower_case   --data_dir data_tmp/   --max_seq_length 50   --output_dir ~/workspace/models/run1/

python run_glue.py   --model_type roberta   --model_name_or_path ~/workspace/models/run2/   --task_name CoLA   --do_predict   --do_lower_case   --data_dir data_tmp/   --max_seq_length 50   --output_dir ~/workspace/models/run2/

python run_glue.py   --model_type roberta   --model_name_or_path ~/workspace/models/run3/   --task_name CoLA   --do_predict   --do_lower_case   --data_dir data_tmp/   --max_seq_length 40   --output_dir ~/workspace/models/run3/

python run_glue.py   --model_type roberta   --model_name_or_path ~/workspace/models/run4/   --task_name CoLA   --do_predict   --do_lower_case   --data_dir data_tmp/   --max_seq_length 50   --output_dir ~/workspace/models/run4/

python run_glue.py   --model_type roberta   --model_name_or_path ~/workspace/models/run5/   --task_name CoLA   --do_predict   --do_lower_case   --data_dir data_tmp/   --max_seq_length 50   --output_dir ~/workspace/models/run5/

python eval.py --data_dir ~/data --result_dir=~/workspace/models --runs 5 --output_file $2