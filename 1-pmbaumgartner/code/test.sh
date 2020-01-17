

#!/bin/bash

export TASK_NAME=COLA
export DATA_DIR=/wdata/data/
export TEMP_PREDS_DIR=$DATA_DIR/tmppreds

if [ -d "/submission/model" ] 
then
    echo "Submission models exist." 
    export MODELS_DIR=/submission/model
    echo "Using submission model dir: $MODELS_DIR"
else
    echo "Submission model directory missing"
    export MODELS_DIR=/wdata/model
    echo "Using new trained model dir: $MODELS_DIR"
fi

if [ -d $TEMP_PREDS_DIR ] 
then
    rm -rf $TEMP_PREDS_DIR
fi

if [[ -z $1 || -z $2 ]]; then
    echo "Missing arguments. File is called as: test.sh <test-csv-file> <output_path>"
else
    echo "Generating predictions using file $1."
    echo "Exporting to $2 when finished."
    echo "Starting preprocessing. Expected duration: 1 min per 3,500 rows"
    python 02_process_test.py $1 $DATA_DIR
    echo "Evaluation dataset preprocessing complete. Saved in $DATA_DIR"

    model_dirs=($MODELS_DIR/*/)
    echo "N MODEL DIRS FOUND: ${#model_dirs[@]}"
    for f in "${model_dirs[@]}"
    do
        echo "GENERATING PREDICTIONS. MODEL ID: $f"
        python run_model.py \
            --model_type roberta \
            --model_name_or_path $f \
            --task_name $TASK_NAME \
            --do_eval \
            --do_lower_case \
            --data_dir $DATA_DIR \
            --max_seq_length 56 \
            --per_gpu_eval_batch_size 128 \
            --output_dir $f \
            --pred_output_dir $TEMP_PREDS_DIR/$f
    done 
    python 03_combine_preds.py $TEMP_PREDS_DIR $1 $2 
fi
