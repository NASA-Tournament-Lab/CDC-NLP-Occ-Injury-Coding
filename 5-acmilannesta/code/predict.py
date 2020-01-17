from keras.layers import *
from keras.models import Model
from keras_bert import get_custom_objects, calc_train_steps, Tokenizer, AdamWarmup, load_trained_model_from_checkpoint
from tqdm import tqdm
from absl import flags
import pandas as pd
import numpy as np
import codecs, gc, boto3, os

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", None, "The input data dir")
flags.DEFINE_string("output_dir", None, "The output data dir")

# read dataset
train = pd.read_csv('/data/train.csv')
test = pd.read_csv(FLAGS.data_dir)

# parameter setting
MAXLEN = 142
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_CLASSES = 48
LR = 5e-5
MIN_LR = 0
config_path = '/wdata/uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/wdata/uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/wdata/uncased_L-12_H-768_A-12/vocab.txt'

# load dictionaries for tokenizer
token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = Tokenizer(token_dict)

# preprocess data for training
def convert_data(data_df, branch='training'):
    data_df = data_df.reset_index(drop=True)
    global tokenizer
    indices = []
    for i in tqdm(range(len(data_df))):
        ids, segments = tokenizer.encode(data_df.loc[i, 'text'])
        indices.append(ids)
    aux = data_df[['age', 'sex']].apply(lambda x: (x - min(x)) / (max(x)-min(x)))
    if branch=='training':
        targets = data_df['event_idx']
        return indices, np.array(targets), np.array(aux)
    else:
        return indices, np.array(aux)

# data generator for bucket padding
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])

class data_generator:
    def __init__(self, data, batch_size=BATCH_SIZE, branch='train'):
        self.data = data
        self.batch_size = batch_size
        self.branch = branch
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            if self.branch == 'train':
                np.random.shuffle(self.data)
            for i in range(self.steps):
                d = self.data[i * self.batch_size: (i + 1) * self.batch_size]
                X1 = seq_padding([x[0] for x in d])
                X2 = np.zeros_like(X1)
                if self.branch == 'test':
                    aux = np.array([x[1] for x in d])
                    yield [X1, X2, aux]
                else:
                    Y = np.array([x[1] for x in d])
                    aux = np.array([x[2] for x in d])
                    yield [X1, X2, aux], Y

# build model structure
def model_build(len_train):
    global NUM_CLASSES
    global BATCH_SIZE
    global NUM_EPOCHS
    global MIN_LR
    global LR

    bert_model = load_trained_model_from_checkpoint(
        config_path,
        checkpoint_path,
        seq_len = MAXLEN,
        trainable=True
    )

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    aux_in = Input(shape=(2, ))

    inputs = bert_model([x1_in, x2_in])
    bert = Lambda(lambda x: x[:, 0])(inputs)
    dense = concatenate([bert, aux_in])
    outputs = Dense(NUM_CLASSES, activation='softmax')(dense)
    model = Model([x1_in, x2_in, aux_in], outputs)

    decay_steps, warmup_steps = calc_train_steps(
        len_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
    )

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=AdamWarmup(
            decay_steps=decay_steps,
            warmup_steps=warmup_steps,
            lr=LR,
            min_lr=MIN_LR,
            ),
        metrics=['sparse_categorical_accuracy']
    )
    del bert_model
    gc.collect()
    return model
model = model_build(len(train))

# predict
test_indices, test_aux = convert_data(test, branch='test')
pred = np.zeros((len(test), NUM_CLASSES))
for i in range(1, 16):
    print('Fold - {:} model loading'.format(i))
    model = model.load_weights('/wdata/model-oof-'+str(i)+'.h5')
    print('Model loading completed')
    test_D = data_generator(list(zip(test_indices, test_aux)), branch='test')
    pred += model.predict_generator(test_D.__iter__(), len(test_D), verbose=1) / 15

# import auxilary predictions
os.environ['AWS_SHARED_CREDENTIALS_FILE'] = 'AWS.txt'
s3 = boto3.Session(profile_name='default').client('s3')
s3.download_file(Bucket='acmilannesta', Key='final/pred_xlnet.npy', Filename='pred_xlnet.npy')
s3.download_file(Bucket='acmilannesta', Key='final/pred_large_uncased.npy', Filename='pred_large_uncased.npy')
s3.download_file(Bucket='acmilannesta', Key='final/pred_large10.npy', Filename='pred_large10.npy')

# output submission file
final_pred = pred + np.load('pred_xlnet.npy') + np.load('pred_large_uncased.npy') + np.load('pred_large10')
test['event'] = np.argmax(final_pred, 1)
test['event'] = test['event'].map({x: y for x, y in enumerate(np.sort(train.event.unique()))})
test.to_csv(FLAGS.output_dir, index=False)
