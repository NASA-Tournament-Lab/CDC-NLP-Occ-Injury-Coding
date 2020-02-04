#!/bin/bash -x
source ./parameters.sh

# Ideally, this is run on a GPU/TPU, otherwise it will take forever.

mkdir -p ${datadir}
python3 src/preprocessing.py --test_file "$1" --output_dir ${datadir}

# Remove some leftover stuff.
rm -rf ${outdir}/examples/
rm -rf ${outdir}/test_runs/

# Go back to the work directory.
cd ${home}

# Evaluate the entire ensemble of models.
for i in 1 2 3 4
do
python3 -m albert.${script} ${parameters} \
    --do_predict=true \
    --init_checkpoint=${outdir}/train_runs/${model}/${i}/model.ckpt-14433 \
    --output_dir=${outdir}/test_runs/${model}/${i}/ \
    --data_examples=${outdir}/examples/${cdc_spm}
done

# Use the ensemble output to start evaluating.
python3 src/evaluation.py --data_dir ${datadir} --output_dir ${outdir}/test_runs/${model}
cp ${outdir}/test_runs/${model}/solution.csv "$2"
