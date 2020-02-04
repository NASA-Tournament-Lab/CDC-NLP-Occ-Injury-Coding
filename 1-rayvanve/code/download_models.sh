source ./parameters.sh
# First, download the necessary albert model files.
mkdir -p ${modeldir}/${model}
cd ${modeldir}/${model}
wget -O 2.tar.gz https://tfhub.dev/google/albert_${model}/2\?tf-hub-format\=compressed
tar -xf 2.tar.gz
rm 2.tar.gz

# Here, we download the generated model files.
mkdir -p ${outdir}/
cd ${outdir}
wget https://storage.googleapis.com/codeforces_cdc/trained_models.tar.gz
tar -xvf trained_models.tar.gz
rm trained_models.tar.gz
