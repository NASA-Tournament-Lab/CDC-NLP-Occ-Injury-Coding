#!/bin/bash
echo "$@"

mkdir -p models
rm -rf models/*

for i in {0..2}
    do
    echo "BERT"
    python3 bert.py --input_dir $1 --bag $i

    echo "XLNET"
    python3 xlnet.py --input_dir $1 --bag $i

    echo "RoBERTa"
    python3 roberta.py --input_dir $1 --bag $i

    echo "RoBERTa large"
    python3 large_roberta.py --input_dir $1 --bag $i

    echo "fp16 RoBERTa large"
    python3 large_roberta_fp16.py --input_dir $1 --bag $i
    done
