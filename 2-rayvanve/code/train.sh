#!/bin/bash -x
source ./parameters.sh

# Training is supposed to be run on a GPU/TPU, otherwise it will never finish.

echo "Make sure you pass the train file in the first argument!!"

# Remove leftover stuff.
rm -rf ${outdir}

# Create a directory holding the `data`, i.e. train and test processed.
mkdir -p ${datadir}
python3 src/preprocessing.py --train_file "$1" --output_dir ${datadir}

# Create our own SPM model.
mkdir -p ${outdir}/cdc_models/
cd ${outdir}/cdc_models/
python3 ${home}/src/tokenizer.py --split_by_whitespace=true --vocab_size=${cdc_vocab_size} --model_prefix ${cdc_spm} --train_file ${datadir}/train.txt

# Go back to the work directory.
cd ${home}

# We create an ensemble of 4 models.
for i in 1 2 3 4 
do

# Now first train our custom embedding only, not using any ALBERT code.
python3 -m albert.${script} ${parameters} \
    --do_train=true \
    --cdc_only=true \
    --train_batch_size=32 \
    --learning_rate=5e-4 \
    --num_train_epochs=5 \
    --save_checkpoints_steps=6000 \
    --output_dir=${outdir}/train_runs/${cdc_model}/${cdc_embed_size}/${i} \
    --data_examples=${outdir}/examples/${cdc_spm}

python3 -m albert.${script} ${parameters} \
    --do_train=true \
    --init_checkpoint=${modeldir}/${model}/variables/variables \
    --cdc_init_checkpoint=${outdir}/train_runs/${cdc_model}/${cdc_embed_size}/${i}/model.ckpt-6000 \
    --train_batch_size=32 \
    --learning_rate=3e-5 \
    --num_train_epochs=3 \
    --save_checkpoints_steps=5000 \
    --output_dir=${outdir}/train_runs/${model}/${i}/ \
    --data_examples=${outdir}/examples/${cdc_spm}
done
