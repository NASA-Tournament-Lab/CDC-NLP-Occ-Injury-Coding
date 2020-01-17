## CDC Text Classification Marathon @ TopCoder (https://www.topcoder.com/challenges/30103825)
team "now_you_see"
Pavel Blinov (blinoff.pavel@gmail.com), Data Scientist at Sberbank AI Lab, Moscow, Russia

My solution is the wighted avarage of 15 models
It uses up to 10.5 GB of GPU RAM, should be ok with NVIDIA Tesla K80.
At docker build phase trained and base models would be downloaded, 30 GB of disk space should be enough.

### How to run
Build the docker image
```bash
sudo docker build --network=host --no-cache -t now_you_see .
```
Run it like this
```bash
sudo docker run --gpus '"device=0"' --rm -it -v path_to_train_AND_test_csv_files:/data:ro -v writable_outer_directory_to_store_the_solution_file:/work/solution now_you_see
```

Once inside the container:
run
```bash
time ./test.sh /data /work/solution
```
for the inference with trained models. The solution.csv file will be at ```/work/solution``` directory.

run
```bash
time ./train.sh /data
```
to retrain models and store them in ```/work/models``` directory.
