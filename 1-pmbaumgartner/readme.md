# CDC Text Classification Marathon: pmbaumgartner's solution

For any questions, email Peter Baumgartner at pbaumgartner@rti.org.

**Table of Contents**
- [CDC Text Classification Marathon: pmbaumgartner's solution](#cdc-text-classification-marathon-pmbaumgartners-solution)
  - [Summary](#summary)
    - [Architecture](#architecture)
    - [Preprocessing](#preprocessing)
  - [Running the Models](#running-the-models)
    - [Training a new Ensemble (`train.sh`)](#training-a-new-ensemble-trainsh)
      - [Performance Expectations for Retraining](#performance-expectations-for-retraining)
      - [Parameter Variability](#parameter-variability)
      - [Unstable Model Training (Model Predicts only Majority Class)](#unstable-model-training-model-predicts-only-majority-class)
    - [Generating Predictions (`test.sh`)](#generating-predictions-testsh)
    - [Assumptions and Debugging](#assumptions-and-debugging)
      - [Expectations](#expectations)
      - [Dockerfile](#dockerfile)
      - [Out of Memory Issues](#out-of-memory-issues)
  - [Future Improvements](#future-improvements)

## Summary
### Architecture
My final submission is an ensemble of `RoBERTa-large` models. Of all single models I submitted, `RoBERTa-large` had consistently the best single-model performance. Ensembling these models adds a significant improvement to the F1 score over a single model. In my testing, including more models in the ensemble increased performance. There are details on the relationship between Number of Models Trained and F1 Score on the provisional data in the section titled [Training a new Ensemble (train.sh)](#training-a-new-ensemble-trainsh).

### Preprocessing
There is a preprocessing script with various regular expressions to clean up the input data.

I developed the regular expressions by these by first looking at the benchmark script provided and copying those. Then I started looking through the data to find any acronyms or shorthand I didn't understand and doing some research and intuition around what it might mean. This was probably the most difficult stage of the model, because much of the shorthand used is not standard or general acronyms, but very specific to this domain and not documented publicly.

I think this preprocessing and translating from shorthand to full grammatical English is critically important, and would further benefit from having a subject matter expert continue to develop more of the substitutions. It is important because the base language models like BERT are trained on a corpus of grammatically correct, full english sentences. While the default tokenization and embeddings will still operate pretty well on non-english words or acronyms, not using complete sentences or regular grammar means you're leaving information contained in the full language model on the table.

Additionally, language models also depend heavily on sentence structure. Because of this, I added "fake" sentence structure in two ways:
- Adding a period and a space prior to the word "diagnosis", since the diagnosis was usually separate from the event description.
- Appending each text with a period.

As a final step, the text is run through [symspell](https://github.com/mammothb/symspellpy). This algorithm performs a spell check and also un-compounds unintentionally compounded words (e.g. `WHILEWORKING` → `WHILE WORKING`), which happens frequently within the data.

## Running the Models
### Training a new Ensemble (`train.sh`)
When retraining models, to remain in "The allowed time limit for the train.sh script is 2 days." restriction, you should train as many models as possible in 48 hours. In my testing **training 1 model on a `NVIDIA TITAN X (Pascal) (12GB)` took roughly 4h 15m.**, meaning for the competition rules you could train **11** models within the training time budget, which includes time for data preprocessing. **Training time budget assumes training on a GPU**

To play it safe, the default is to train **7** models to ensure the competition time limit rules are adhered to and performance is above second place submission. You can increase the number of models trained  by passing an optional second argument to `train.sh` indicating the number of models you want to train, which would look like: `train.sh <train-csv-file> <n-models>`. This also means you can create a single model with `train.sh <train-csv-file> 1`. Each new model will be named by random seed, so you could also use `train.sh <train-csv-file> 1` 12 times to train an ensemble of 12 models.

**To get as close as possible to final submission performance, the maximum number of models available in the train time budget should be trained.** 


#### Performance Expectations for Retraining
To identify performance expectations on ensembles fewer than 12 models, I randomly sampled 3 sets of 3, 5, 7, 9, and 11 models from the full ensemble and documented their scores on the provisional dataset. The table below describes the Mean F1 and individual ensemble model scores for the number of models included.

| N Models 	| Mean F1	| Set 1  	| Set 2  	| Set 3  	|
|---------:	|-------	|-------	|-------	|-------	|
| **3**    	| 89.33 	| 89.39 	| 89.27 	| 89.32 	|
| **5**    	| 89.35 	| 89.36 	| 89.42 	| 89.26 	|
| **7**    	| 89.41 	| 89.33 	| 89.46 	| 89.45 	|
| **9**    	| 89.45 	| 89.46 	| 89.44 	| 89.44 	|
| **11**   	| 89.45 	| 89.46 	| 89.45 	| 89.45 	|

Given these results, to ensure performance that would beat the current second place model, at least 9 models should be trained. If that is not possible, 7 is a reasonable number of models, though there was a set of 7 models with a score of 89.33, which is close to the current second best model.

#### Parameter Variability
When retraining an ensemble of 12 models from scratch, I received a provisional score of `89.18444`, so expect some variability in the a retrained solution. Based on my analysis, more diversity among models arises from a combination of learning rate and epochs. The train script is set to a `3e-5` learning rate. If a new ensemble is underperforming, try an ensemble of models with various learning rates or training epochs. I have seen `2e-5`, `3e-5`, and `5e-5` used in the literature. When using a combined ensemble of 24 models (12 from submission, 12 from new training with learning rate `2e-5`), the score was `89.40392`, indicating these additional models don't improve performance if all are tuned to a `2e-5` learning rate.

The learning rate and training epochs for each submission model are included below:

| seed  	| epochs 	| lr    	|
|-------	|--------	|-------	|
| 128   	| 3      	| 2e-5  	|
| 14394 	| 3      	| 2e-5  	|
| 18036 	| 3      	| 2e-5  	|
| 29109 	| 3      	| 2e-5  	|
| 23344 	| 3      	| 2e-5  	|
| 7598  	| 3      	| 2e-5  	|
| 20548 	| 3      	| 2e-5  	|
| 10621 	| 3      	| 3e-5  	|
| 23166 	| 3      	| 3e-5  	|
| 25417 	| 3      	| 3e-5  	|
| 2497  	| 3      	| 3e-5  	|
| 16080 	| 4      	| 2e-5  	|

#### Unstable Model Training (Model Predicts only Majority Class)
The training process can be unstable due to the class imbalance. There were a handful of times in testing where I would train a model that would only output the majority class (71). I hypothesize this is due to the relationship between the batch size and rate of each of the classes. That is, the smaller the batch size, the more likely it is that a mini-batch contains examples of all one class. When this happens, the gradient calculation has issues. This issue can be avoided by increasing the effective batch size or retraining a new model with the same parameters.

### Generating Predictions (`test.sh`)
Since this is an ensemble model, the `test.sh` script generates predictions from all trained models available. It will look for **all** model folders in `/submission/model/` inside the container, where the previously trained models are. If that folder doesn't exist, it will look for models in the write approved folder for the competition, `/wdata/model/`, and assume that new models have been trained. 

For a single model, prediction takes about 10 minutes, so to make sure inference happens in an appropriate amount of time, make sure `n_models × 10 minutes` is less than your inference time budget (`12 hours / 720 min` in competition rules). 

If you restart the container and then want to run inference using non-submission models with `test.sh`, make sure you delete the `/submission/model/` folder first with `rm -rf /submission/model/`. Then, be sure your new models are at `/wdata/model`. These directories are hardcoded into the `test.sh` script (though you're welcome to change them and point them at a new folder of models). You need to do this because `test.sh` will check for the existence of `/submission/model/` first and use those models if that folder exists.

### Assumptions and Debugging

#### Expectations
1. `/data/` is mounted, read only, and contains the original `train.csv` and `test.csv` from the competition.
2. `/wdata/` is mounted, write accessible, and will be used to store any files generated from training or inference..
   1. Intermediate data will be stored in `/wdata/data`. The intermediate processed data is: `/wdata/data/train.tsv` for training and `/wdata/data/dev.tsv` for inference. The model will also cache the input features here when it trains for the first time.
   2. Model weights will be stored in `/wdata/model/`. They are 1.2GB per model.
   3. Prediction outputs will be stored in `/wdata/solution.csv` 
3. `test.sh` will use models in `/submission/model/` if it exists. These are the models used for the submission. 
   1. If `/submission/model/` does not exist, because it was deleted as a part of the `train.sh` script requirement: "As its first step, train.sh must delete your home-made models shipped with your submission.", `test.sh` will look in `/wdata/model/` folder. This folder is used to store new models that are trained.
   2. You can always rerun the image to restore the submission models.
4. Training and test data is should be in same format as the competition data, namely a `csv` with headers `text` and `event`. The columns are `text` representing the input text and `event` representing the label (if training).
5. Docker needs to be run with access to the GPU on a machine with CUDA. This means passing `--runtime=nvidia` and possibly `-e NVIDIA_VISIBLE_DEVICES=0` to the `docker run` command. It will still run without these flags, but training will be slow.
6. The compiled models are `15.27 GB` in total, so be sure there is additionally at least `20 GB` of hard drive space available to the docker container when it is built so that the model files can be extracted.

#### Dockerfile
Since there are several large model files required, I created a specific docker image containing all the model files and had it built via dockerhub so that the build can be cached and downloaded. This prevents any of the model downloads failing incrementally, making the final model unavailable.  You can view this pre-submission dockerfile at https://hub.docker.com/r/pbaumgartner/cdc-text-classification.

The dockerfile at `/code/dockerfile` has an initial long running step that extracts the compressed model files. The current file being extracted is echo'd to the terminal, so progress can be monitored, but expect this step to take a bit of time on the first build.

#### Out of Memory Issues
If you're certain that training is causing out-of-memory errors, you can change a few parameters in the `train.sh` script to reduce the batch size. The training process takes advantage of gradient accumulation, which allows for smaller batch sizes while maintaining the same "effective batch size". The two lines that can be adjusted to accommodate this are:

```
train.sh
...
    --per_gpu_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
...
```

`per_gpu_train_batch_size × gradient_accumulation_steps` is the effective per-GPU batch size. Original models were trained using an effective batch size of 32. If you're lowering the `per_gpu_train_batch_size`, make sure that you raise `gradient_accumulation_steps` to accommodate an effective batch size of 32 to replicate training conditions.

## Future Improvements

In my experimentation, there were a few other approaches that saw minor improvements, but due to restrictions on model size or training time were not feasible to be evaluated further. I am happy to provide code, model files, or discuss these more if curious. The additional approaches are:

1. **Data Augmentation:** There were slight, non-significant improvements when using a `RoBERTa-base` model with a 1x augmented dataset. An augmented dataset was created by preprocessing the data, tokenizing the data, and randomly substituting each token with a 0.1 probability with a token "similar" to that token in the GloVe 300D word embeddings model. However, adding a 1x augmented dataset also doubles the training time.
2. **Language Model Pre-training**: Continuing the language pre-training on the corpus is another idea. I also saw slight, non-significant improvements when doing unsupervised language model pre-training on just the text from the training and test datasets. 
3. **Ensemble Optimization**:  The ensemble could be further optimized by pruning poor performing models by evaluation on a known test set. In addition, since models output the logits for each class, those could be used as features in model stacking, where a model (usually gradient boosted trees) uses the model predictions as features in the final prediction. 
