# -*- coding: utf-8 -*-
from __future__ import print_function
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0';
import torch
torch.cuda.is_available()
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
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
bert_name = 'bert-base-uncased'

def clean_from_punct_and_reverse(s):
    parts = s.lower().split(' ')
    s = s+' '+' '.join(parts[::-1])    
    return s

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
    num_train = df.shape[0]
    
    result = []
    for age, s in zip(df.age, df.text):
        elems = re.findall(r'^'+str(age), s, flags=re.MULTILINE)
        s = re.sub(r'^'+str(age), '', s, flags=re.MULTILINE)
        if len(elems)==0:
            s = re.sub(str(age), '', s, flags=re.MULTILINE)
        result.append( s.strip().lower() )
    df['text'] = result

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

    df.text = df.text.apply( clean_from_punct_and_reverse )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (device)
    # Create sentence and label lists
    sentences = df.text.values

    # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
    labels = np.array([l2i.get(l, 0) for l in df.event.values])

    tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    print ("Tokenize the first sentence:")
    print (tokenized_texts[0])

    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="pre")

    # Create attention masks
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    # Convert all of our data into torch tensors, the required datatype for our model
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    attention_masks = torch.tensor(attention_masks)

    batch_size = 128#256
    train_data = TensorDataset(input_ids, attention_masks, labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, num_workers=4)

    model = BertForSequenceClassification.from_pretrained(bert_name, num_labels=len(i2l))
    model = model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    gradient_accumulation_steps = 1
    epochs = 3#2
    num_total_steps = len(train_dataloader) // gradient_accumulation_steps * epochs

    # Parameters:
    lr = 3e-5
    max_grad_norm = 1.0
    #num_total_steps = 1000
    num_warmup_steps = 0#100

    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)
    # # Parameters:
    # lr = 3e-5
    # max_grad_norm = 1.0
    # num_total_steps = 1000
    # num_warmup_steps = 100

    # # This variable contains all of the hyperparemeter information our training loop needs
    # optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)

    # Store our loss and accuracy for plotting
    train_loss = []
    # Number of training epochs (authors recommend between 2 and 4)
    for _ in range(epochs):
        # Training
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()
        
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        
        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss, logits = outputs[:2]
            train_loss.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            print (step, tr_loss/nb_tr_steps)
            #if step>2: break
        print ("Train loss: {}".format(tr_loss/nb_tr_steps))
        
    bert_out_dir = 'models/{}_{}'.format(bert_name, bag)
    if not os.path.exists(bert_out_dir):
        os.makedirs(bert_out_dir)
    model.save_pretrained(bert_out_dir)
