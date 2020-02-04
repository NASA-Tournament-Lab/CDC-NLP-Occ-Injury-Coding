### Prerequisites

- CUDA drivers ==10.0 and CUDNN 7 are assumed to be installed on machine
- nvidia-docker is installed on machine  [docker run --gpus all ... should work]

###checking GPU
- Open python terminal inside container using `python` command
- Run `import torch`
- Run `torch.cuda.device_count()`
- It should return 1 in which case GPU is being used otherwise there is something wrong with driver version or nvidia-docker installation

###Running test.sh

- The signature of test.sh `test file`  `output file path`
- Running this script should be done in about an hour and produces a file similar to solution.csv

**Assumptions**
- It is assumed that code_descriptions.csv is present in `~\data` folder along with test.csv(or any testing file)

###Running train.sh

- The signature of test.sh `train_file`
- Running this script should be done in 30-32 hours and produces 5 models in `~/workspace/models` directory.

**Assumptions**
- It is assumed that code_descriptions.csv is present in `~\data` folder along with train.csv(or any training file)

