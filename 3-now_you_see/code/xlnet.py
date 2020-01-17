# -*- coding: utf-8 -*-
from __future__ import print_function
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0';
import torch
torch.cuda.is_available()
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
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

SEED = 369
random.seed(369)
np.random.seed(369)
MAX_LEN = 50
model_file_address = 'xlnet-base-cased'

import math
import string
import torch.nn.functional as F

import torch
import os
from tqdm import tqdm, trange
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer

def clean_from_punct_and_reverse(s):
    parts = s.lower().split(' ')
    s = s+' '+' '.join(parts[::-1])    
    return s

# Kludge to prevent isalph() from checking for non-ASCII characters
def is_alpha(char):
    return char in string.ascii_lowercase

def keep_alphanumeric(doc):
    doc = [char for char in doc]
    out = ''
    for char in doc:
        good = is_alpha(char) or char.isnumeric() or char in [' ', '.']
        if good:
            out += char
        else:
            out += ' '
    return out

def remove_special(docs):
    docs = [doc.lower() for doc in docs]
    docs = [keep_alphanumeric(doc) for doc in docs]
    return docs

# Converts numerals to text, i.e. 1 to 'one'
def remove_numerals(docs):
    docs = [doc.replace('0', ' zero ') for doc in docs]
    docs = [doc.replace('1', ' one ') for doc in docs]
    docs = [doc.replace('2', ' two ') for doc in docs]
    docs = [doc.replace('3', ' three ') for doc in docs]
    docs = [doc.replace('4', ' four ') for doc in docs]
    docs = [doc.replace('5', ' five ') for doc in docs]
    docs = [doc.replace('6', ' six ') for doc in docs]
    docs = [doc.replace('7', ' seven ') for doc in docs]
    docs = [doc.replace('8', ' eight ') for doc in docs]
    docs = [doc.replace('9', ' nine ') for doc in docs]
    return docs

# Removes extra whitespace for a list of text strings. Kludgy, but useful.
def remove_whitespace(docs):
    docs = [' '.join(doc.split()) for doc in docs]
    docs = [doc.rstrip() for doc in docs]
    docs = [doc.lstrip() for doc in docs]
    return docs

# Removes 'nan' strings for a list of strings
def remove_nan(docs):
    docs = [doc.replace('nan', '') for doc in docs]
    return docs

# Function that cleans up a free-text column
def clean_text(docs):
    docs = [doc.lower() for doc in docs]
    docs = remove_special(docs)
    docs = [doc.replace('yom', ' year old male ') for doc in docs]
    docs = [doc.replace('yof', ' year old female ') for doc in docs]
    docs = [doc.replace('ym', ' year old male ') for doc in docs]
    docs = [doc.replace('yf', ' year old female ') for doc in docs]
    docs = [doc.replace('yowm', ' year old male ') for doc in docs]
    docs = [doc.replace('yowf', ' year old female ') for doc in docs]
    docs = [doc.replace('yo m', ' year old male ') for doc in docs]
    docs = [doc.replace('y o m', ' year old male ') for doc in docs]
    docs = [doc.replace('yo f', ' year old female ') for doc in docs]
    docs = [doc.replace('y o f', ' year old female ') for doc in docs]
    docs = [doc.replace(' yo ', ' year old ') for doc in docs]
    docs = [doc.replace('dx', ' diagnosis ') for doc in docs]
    docs = [doc.replace(' d x ', ' diagnosis ') for doc in docs]
    docs = [doc.replace(' c o ', ' complains of ') for doc in docs]
    docs = [doc.replace('bibems', ' brought in by ems ') for doc in docs]
    docs = [doc.replace(' pt ', ' patient ') for doc in docs]
    docs = [doc.replace(' pts ', ' patients ') for doc in docs]
    docs = [doc.replace(' lac ', ' laceration ') for doc in docs]
    docs = [doc.replace(' lt ', ' left ') for doc in docs]
    docs = [doc.replace(' rt ', ' right ') for doc in docs]
    docs = [doc.replace(' sus ', ' sustained ') for doc in docs]
    docs = [doc.replace('fx', ' fracture ') for doc in docs]
    docs = [doc.replace('bldg', ' building ') for doc in docs]
    docs = [doc.replace(' s p ', ' status post ') for doc in docs]
    docs = [doc.replace(' w ', ' with ') for doc in docs]
    docs = [doc.replace(' gsw ', ' gun shot wound ') for doc in docs]
    docs = [doc.replace(' etoh ', ' ethanol ') for doc in docs]
    docs = [doc.replace(' loc ', ' loss of consciousness ') for doc in docs]
    docs = [doc.replace('pta', ' prior to arrival ') for doc in docs]
    docs = [doc.replace(' x ', ' for ') for doc in docs]
    docs = [doc.replace(' chi ', ' closed head injury ') for doc in docs]
    docs = [doc.replace(' 2 2 ', ' secondary to ') for doc in docs]
    docs = [doc.replace('lbp', ' low blood pressure ') for doc in docs]
    docs = [doc.replace(' htn ', ' hypertension ') for doc in docs]
    docs = [doc.replace(' pw ', ' puncture wound ') for doc in docs]
    docs = remove_whitespace(docs)
    return docs

def get_input_data(tokenizer, sentences):
    full_input_ids = []
    full_input_masks = []
    full_segment_ids = []

    SEG_ID_A   = 0
    SEG_ID_B   = 1
    SEG_ID_CLS = 2
    SEG_ID_SEP = 3
    SEG_ID_PAD = 4

    UNK_ID = tokenizer.encode("<unk>")[0]
    CLS_ID = tokenizer.encode("<cls>")[0]
    SEP_ID = tokenizer.encode("<sep>")[0]
    MASK_ID = tokenizer.encode("<mask>")[0]
    EOD_ID = tokenizer.encode("<eod>")[0]

    for i, sentence in enumerate(sentences):
        # Tokenize sentence to token id list
        tokens_a = tokenizer.encode(sentence)
        
        # Trim the len of text
        if(len(tokens_a)>MAX_LEN-2):
            tokens_a = tokens_a[:MAX_LEN-2]
            
        tokens = []
        segment_ids = []
        
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(SEG_ID_A)
            
        # Add <sep> token 
        tokens.append(SEP_ID)
        segment_ids.append(SEG_ID_A)
        
        # Add <cls> token
        tokens.append(CLS_ID)
        segment_ids.append(SEG_ID_CLS)
        
        input_ids = tokens
        
        # The mask has 0 for real tokens and 1 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [0] * len(input_ids)

        # Zero-pad up to the sequence length at fornt
        if len(input_ids) < MAX_LEN:
            delta_len = MAX_LEN - len(input_ids)
            input_ids = [0] * delta_len + input_ids
            input_mask = [1] * delta_len + input_mask
            segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

        assert len(input_ids) == MAX_LEN
        assert len(input_mask) == MAX_LEN
        assert len(segment_ids) == MAX_LEN
        
        full_input_ids.append(input_ids)
        full_input_masks.append(input_mask)
        full_segment_ids.append(segment_ids)
    return full_input_ids, full_input_masks, full_segment_ids

# Set acc funtion
def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

if __name__== '__main__':
    cwd = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--bag')
    args = parser.parse_args()

    input_dir = args.input_dir
    bag = args.bag

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

    mask = df.event.apply(lambda s: s in l2i).values
    print (df.shape)
    df = df.iloc[mask, :]
    df.reset_index(inplace=True, drop=True)
    print (df.shape)
    num_train = df.shape[0]

    df.text = pd.Series(clean_text(df.text))
    df.text = df.text.apply( clean_from_punct_and_reverse )

    sentences = df.text.to_list()
    labels = df.event.to_list()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (device)

    # With cased model, set do_lower_case = False
    tokenizer = XLNetTokenizer.from_pretrained(model_file_address)
    
    full_input_ids, full_input_masks, full_segment_ids = get_input_data(tokenizer, sentences)

    # Make label into id
    tags = [l2i.get(lab, 0) for lab in labels]

    inputs = torch.tensor(full_input_ids)
    tags = torch.tensor(tags)
    masks = torch.tensor(full_input_masks)
    segs = torch.tensor(full_segment_ids)

    # Set batch num
    batch_size = 118

    # Set token embedding, attention embedding, segment embedding
    train_data = TensorDataset(inputs, masks, segs, tags)
    train_sampler = RandomSampler(train_data)
    # Drop last can make batch training better for the last one
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, drop_last=True)

    model = XLNetForSequenceClassification.from_pretrained(model_file_address, num_labels=len(l2i))
    model.to(device)

    # Set epoch and grad max num
    epochs = 4
    max_grad_norm = 1.0
    # Cacluate train optimiazaion num
    num_train_optimization_steps = int( math.ceil(len(inputs) / batch_size) / 1) * epochs

    # Fine tune model all layer parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    #optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)
    optimizer = AdamW(optimizer_grouped_parameters, lr=4e-5)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=num_train_optimization_steps)

    print("***** Running training *****")
    print("Num examples = %d"%(len(inputs)))
    print("Batch size = %d"%(batch_size))
    print("Num steps = %d"%(num_train_optimization_steps))
    for _ in range(epochs):
        model.train()
        
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_segs, b_labels = batch
            
            # forward pass
            outputs = model(input_ids =b_input_ids, token_type_ids=b_segs, input_mask = b_input_mask, labels=b_labels)
            loss, logits = outputs[:2]
            
            # backward pass
            loss.backward()
            
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            
            # update parameters
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            #print (step)
            print (step, tr_loss/nb_tr_steps)
            #if step>10: break
            
        # print train loss per epoch
        print ("Train loss: {}".format(tr_loss/nb_tr_steps))
    out_dir = 'models/{}_{}'.format(model_file_address, bag)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model.save_pretrained(out_dir)
