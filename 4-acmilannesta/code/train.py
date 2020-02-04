
import numpy as np
import pandas as pd
import os, gc
import codecs
from absl import flags
from keras.layers import *
from keras.models import Model
from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm import tqdm
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, AdamWarmup, calc_train_steps

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", None, "The input data dir")

# remove home-made model
for i in range(1, 16):
    os.system('rm /wdata/model-oof'+str(i)+'.h5')

train = pd.read_csv(FLAGS.data_dir)
# Event weight
wt = pd.DataFrame(train.event.value_counts()/len(train)).rename(columns={'event': 'weight'})
wt['event'] = wt.index
train = train.merge(wt, how='left', on='event')
# Reassign eventcode
train['event_idx'] = train.event.map({y:x for x, y in enumerate(np.sort(train.event.unique()))})
# Assign weight frequency
train['wt_freq'] = np.where(train.weight < 0.01, 1, np.where(train.weight < 0.05, 2, 3))



""" Parameter setting"""

MAXLEN = 142
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_CLASSES = 48
LR = 5e-5
MIN_LR = 0
config_path = '/wdata/uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/wdata/uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/wdata/uncased_L-12_H-768_A-12/vocab.txt'



"""## Tokenize train and validation set"""

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = Tokenizer(token_dict)

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


"""## Data Generator"""
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

"""##Model Assemble"""
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


kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
idx = [x for x in kf.split(train, train.wt_freq)]


for i, (tr_idx, val_idx) in enumerate(idx, 1):
    print('\nFold - {:}\n'.format(i))
    tr, val = train.loc[tr_idx], train.loc[val_idx]
    tr_x, tr_y, tr_aux = convert_data(tr)
    # val_x, val_y, val_aux = convert_data(val)
    model = model_build(len_train=len(tr_x))
    train_D = data_generator(list(zip(tr_x, tr_y, tr_aux)))
    # valid_D = data_generator(list(zip(val_x, val_y, val_aux)), branch='valid')
    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=NUM_EPOCHS,
    )
    model.save('/wdata/model-oof-'+str(i)+'.h5')
    del model
    gc.collect()

