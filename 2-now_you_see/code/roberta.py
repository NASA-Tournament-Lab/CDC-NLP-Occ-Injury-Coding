# -*- coding: utf-8 -*-
from __future__ import print_function
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0';
import torch
torch.cuda.is_available()

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.data import TensorDataset, Dataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification
from transformers.optimization import AdamW, WarmupLinearSchedule
import io

import numpy as np
import pandas as pd
from collections import Counter
import re
import codecs
import random
import argparse
import gc

from xlnet import clean_text

SEED = 369
random.seed(369)
np.random.seed(369)
MAX_LEN = 45
model_name = 'roberta-base'

import string
from sklearn.metrics import classification_report
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prepare_features(tokenizer, seq_1, max_seq_length=32, zero_pad=False, include_CLS_token=True, include_SEP_token=True):
    ## Tokenzine Input
    tokens_a = tokenizer.tokenize(seq_1)

    ## Truncate
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    ## Initialize Tokens
    tokens = []
    if include_CLS_token:
        tokens.append(tokenizer.cls_token)
    ## Add Tokens and separators
    for token in tokens_a:
        tokens.append(token)

    if include_SEP_token:
        tokens.append(tokenizer.sep_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    ## Input Mask 
    input_mask = [1] * len(input_ids)
    ## Zero-pad sequence lenght
    if zero_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
    #return torch.tensor(input_ids).unsqueeze(0), input_mask
    return input_ids, input_mask

if __name__== '__main__':
    cwd = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--bag')
    args = parser.parse_args()

    input_dir = args.input_dir
    bag = args.bag

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    df = pd.read_csv( os.path.join(input_dir, 'train.csv') )
    print (df.shape)
    
    d = dict(df.event.value_counts())
    i = 0
    l2i = {}
    for k in d:
        #if d[k]>=25:
        if d[k]>=10:
            l2i[k] = i
            i += 1
    i2l = {l2i[l]:l for l in l2i}
    num_class = len(l2i)
    print (num_class)

    df.text = pd.Series(clean_text(df.text))
    labels = df.event.to_list()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (device)

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(i2l))
    model = model.cuda()

    mask = df.event.apply(lambda s: s in l2i).values
    print (df.shape)
    df = df.iloc[mask, :]
    df.reset_index(inplace=True, drop=True)
    print (df.shape)

    input_ids, input_mask, labels = [], [], []
    for txt, l in zip(df.text, df.event):
        ids, mask = prepare_features(tokenizer, txt, zero_pad=True, max_seq_length=MAX_LEN)
        l = l2i[l]
        input_ids.append(ids)
        input_mask.append(mask)
        labels.append(l)

    # Convert all of our data into torch tensors, the required datatype for our model
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    input_mask = torch.tensor(input_mask)

    train_batch_size = 128#256
    train_data = TensorDataset(input_ids, input_mask, labels)
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=train_batch_size, num_workers=4)

    gradient_accumulation_steps = 1
    num_train_epochs = 6#5
    t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    weight_decay = 0
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    learning_rate = 6e-5#5e-5
    adam_epsilon = 1e-8
    warmup_steps = 0
    max_grad_norm = 1.0
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    # Train!
    print ("***** Running training *****")
    print ("  Num examples = ", len(train_data))
    print ("  Num Epochs = ", num_train_epochs)
    print ("  Gradient Accumulation steps = ", gradient_accumulation_steps)
    print ("  Total optimization steps = ", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_steps = 0
    model.zero_grad()
    set_seed(369)
    for _ in range(int(num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            optimizer.zero_grad()
            
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()
            
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            print (step, tr_loss/global_step)
        # print train loss per epoch
        print ("Train loss: {}".format(tr_loss/global_step))

    out_dir = 'models/{}_{}'.format(model_name, bag)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model.save_pretrained(out_dir)
