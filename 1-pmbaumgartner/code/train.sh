#!/bin/bash

export OUTPUT_DIR=/wdata/model
export DATA_DIR=/wdata/data/
export SUBMISSION_MODEL_DIR=/submission/model


if [ -z "$1" ]; then
    echo "train.csv path not passed as argument"
else
    if [ -d "$SUBMISSION_MODEL_DIR" ] 
    then
        echo "Submission models directory exists at $SUBMISSION_MODEL_DIR. DELETING" 
        rm -rf $SUBMISSION_MODEL_DIR
        echo "Directory $SUBMISSION_MODEL_DIR DELETED."
    else
        echo "Submission model folder not present. Taking no action." 
    fi

    echo "PROCESSING DATA"
    python 01_process_train.py $1 $DATA_DIR

    if [ -z $2 ]; then
        echo "TRAINING 7 MODELS (DEFAULT):"
        for ((i=1;i<=7;i++))
        do
            export SEED=$RANDOM
            echo "TRAINING MODEL. SEED: $SEED"
            python run_model.py \
            --model_type roberta \
            --model_name_or_path roberta-large \
            --task_name COLA \
            --do_train \
            --do_lower_case \
            --data_dir $DATA_DIR \
            --max_seq_length 56 \
            --per_gpu_train_batch_size 8 \
            --gradient_accumulation_steps 4 \
            --learning_rate 3e-5 \
            --num_train_epochs 3.0 \
            --save_steps 0 \
            --output_dir $OUTPUT_DIR/$SEED \
            --fp16 \
            --seed $SEED
        done
    else
        echo "TRAINING $2 MODELS"
        for ((i=1;i<=$2;i++))
        do
            export SEED=$RANDOM
            echo "TRAINING MODEL. SEED: $SEED"
            python run_model.py \
            --model_type roberta \
            --model_name_or_path roberta-large \
            --task_name COLA \
            --do_train \
            --do_lower_case \
            --data_dir $DATA_DIR \
            --max_seq_length 56 \
            --per_gpu_train_batch_size 8 \
            --gradient_accumulation_steps 4 \
            --learning_rate 3e-5 \
            --num_train_epochs 3.0 \
            --save_steps 0 \
            --output_dir $OUTPUT_DIR/$SEED \
            --fp16 \
            --seed $SEED
        done
    fi
fi
