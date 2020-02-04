# Set the base directory.
home=/work

# Downloaded albert models
modeldir="${home}/data/albert_models"

# Download trained models
outdir="${home}/data/generated_models"

# Store processed data
datadir="${home}/data/challenge_data"

# Basis model of ALBERT to use
model="xxlarge"

# Set the parameters for our embedding model.
cdc_vocab_size=500
cdc_embed_size=16

# Set some convenient parameters for later.
cdc_spm="${cdc_vocab_size}_with_whitespace"
cdc_model="cdc_${cdc_spm}"
script="run_classifier_sp_v2"

# Parameters to call the script
parameters="\
--task_name=cola \
--data_dir=${datadir} \
--albert_config_file=${modeldir}/${model}/assets/albert_config.json \
--vocab_file=${modeldir}/${model}/assets/30k-clean.vocab \
--spm_model_file=${modeldir}/${model}/assets/30k-clean.model \
--cdc_spm_model_file=${outdir}/cdc_models/${cdc_spm}.model \
--cdc_embedding_size=${cdc_embed_size} \
--cdc_vocab_size=${cdc_vocab_size} \
--max_seq_length=44 "
