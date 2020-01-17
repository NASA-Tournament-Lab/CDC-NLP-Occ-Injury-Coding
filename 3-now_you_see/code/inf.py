# -*- coding: utf-8 -*-
from __future__ import print_function
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0';
import torch
torch.cuda.is_available()

import re
import gc
import numpy as np
import pandas as pd
from collections import Counter
import random
import time
import pickle
import argparse
from scipy.special import softmax

from xlnet import clean_text, get_input_data
from roberta import prepare_features

START_TIME = time.time()
SEED = 369
random.seed(369)
np.random.seed(369)

from sklearn.preprocessing import scale
from keras.preprocessing import text, sequence
from keras import backend as K

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

maxlen = 50
MODELS_DIR = 'models'
bert_name = 'bert-base-uncased'
model_file_address = 'xlnet-base-cased'

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification

def clean_from_punct_and_reverse(s):
    parts = s.lower().split(' ')
    s = s+' '+' '.join(parts[::-1])    
    return s

if __name__== '__main__':
    cwd = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    #device = torch.device('cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv( os.path.join(input_dir, 'train.csv') )
    print (df.shape)
    num_train = df.shape[0]
    
    d = dict(df.event.value_counts())
    i = 0
    l2i = {}
    for k in d:
        #if d[k]>=25:
        if d[k]>=10:
            l2i[k] = i
            i += 1
    i2l = {l2i[l]:l for l in l2i}

    ############ predict with bert models
    df = pd.read_csv( os.path.join(input_dir, 'test.csv') )
    print (df.shape)
    result = []
    for age, s in zip(df.age, df.text):
        elems = re.findall(r'^'+str(age), s, flags=re.MULTILINE)
        s = re.sub(r'^'+str(age), '', s, flags=re.MULTILINE)
        if len(elems)==0:
            s = re.sub(str(age), '', s, flags=re.MULTILINE)
        result.append( s.strip().lower() )
    df['text'] = result

    df.text = df.text.apply( clean_from_punct_and_reverse )
    
    sentences = df.text.values
    # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
    tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=maxlen, dtype="long", truncating="post", padding="pre")
    # Create attention masks
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask) 
    
    prediction_inputs = torch.tensor(input_ids)
    print (prediction_inputs.shape)
    prediction_masks = torch.tensor(attention_masks)
    prediction_data = TensorDataset(prediction_inputs, prediction_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=2048, num_workers=4)
    total = len(prediction_dataloader)

    pred_bert = []
    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('bert')]
    for dir_name in model_files:
        print (dir_name)
        model = BertForSequenceClassification.from_pretrained(os.path.join(MODELS_DIR, dir_name))
        model.to(device)
        # Prediction on test set
        # Put model in evaluation mode
        model.eval()

        # Tracking variables 
        predictions = []
        # Predict 
        for step, batch in enumerate(prediction_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = logits[0]
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            # Store predictions and true labels
            predictions.append(logits)
            print ('{} of {}'.format(step, total))
        print (len(predictions))
        predictions = np.vstack(predictions)
        pred_bert.append( softmax(predictions, axis=1) )
    pred_bert = np.array(pred_bert)
    print (pred_bert)

    del prediction_inputs, prediction_masks, prediction_data, prediction_sampler, prediction_dataloader
    gc.collect()

    ############ predict with xlnet models
    df = pd.read_csv( os.path.join(input_dir, 'test.csv') )
    print (df.shape)
    df.text = pd.Series(clean_text(df.text))
    df.text = df.text.apply( clean_from_punct_and_reverse )
    sentences = df.text.to_list()
    # With cased model, set do_lower_case = False
    tokenizer = XLNetTokenizer.from_pretrained(model_file_address)

    full_input_ids, full_input_masks, full_segment_ids = get_input_data(tokenizer, sentences)

    inputs = torch.tensor(full_input_ids)
    masks = torch.tensor(full_input_masks)
    segs = torch.tensor(full_segment_ids)

    # Set token embedding, attention embedding, segment embedding
    xlnet_data = TensorDataset(inputs, masks, segs)
    # Drop last can make batch training better for the last one
    dataloader = DataLoader(xlnet_data, sampler=SequentialSampler(xlnet_data), batch_size=2048, num_workers=4)
    total = len(dataloader)

    pred_xlnet = []
    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('xlnet')]
    for dir_name in model_files:
        print (dir_name)
        model = XLNetForSequenceClassification.from_pretrained( os.path.join(MODELS_DIR, dir_name) )
        model.to(device)
        # Prediction on test set
        # Put model in evaluation mode
        model.eval()

        # Tracking variables 
        predictions = []
        # Predict 
        for step, batch in enumerate(dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_segs = batch

            with torch.no_grad():
                outputs = model(input_ids=b_input_ids, token_type_ids=b_segs, input_mask=b_input_mask)
                logits = outputs[0]

            # Get textclassification predict result
            logits = logits.detach().cpu().numpy()
            predictions.append(logits)
            print ('{} of {}'.format(step, total))
        print (len(predictions))
        predictions = np.vstack(predictions)
        pred_xlnet.append( softmax(predictions, axis=1) )
    pred_xlnet = np.array(pred_xlnet)
    print (pred_xlnet)

    ############ predict with roberta models
    df = pd.read_csv( os.path.join(input_dir, 'test.csv') )
    print (df.shape)
    df.text = pd.Series(clean_text(df.text))
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    input_ids, input_mask = [], []
    for txt in df.text:
        ids, mask = prepare_features(tokenizer, txt, zero_pad=True, max_seq_length=45)
        input_ids.append(ids)
        input_mask.append(mask)

    # Convert all of our data into torch tensors, the required datatype for our model
    input_ids = torch.tensor(input_ids)
    input_mask = torch.tensor(input_mask)

    train_data = TensorDataset(input_ids, input_mask)
    dataloader = DataLoader(train_data, sampler=SequentialSampler(train_data), batch_size=2048, num_workers=4)
    total = len(dataloader)

    pred_roberta = []
    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('roberta')]
    for dir_name in model_files:
        print (dir_name)
        model = RobertaForSequenceClassification.from_pretrained( os.path.join(MODELS_DIR, dir_name) )
        model.to(device)
        model.eval()
        # Tracking variables
        predictions = []
        # Predict 
        for step, batch in enumerate(dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            predictions.append(logits)
            print ('{} of {}'.format(step, total))
        print (len(predictions))
        predictions = np.vstack(predictions)
        pred_roberta.append( softmax(predictions, axis=1) )
    pred_roberta = np.array(pred_roberta)
    print (pred_roberta)

    ############ predict with large-roberta models
    from large_roberta import *

    df = pd.read_csv( os.path.join(input_dir, 'test.csv') )
    print (df.shape)
    df.text = pd.Series(clean_text(df.text))

    pred_lroberta = []
    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('large-roberta')]
    for dir_name in model_files:
        print (dir_name)
        learn = load_learner('.', file=os.path.join(MODELS_DIR, dir_name))
        learn.data.add_test(df.text.values)
        
        pred, y = learn.get_preds(ds_type=DatasetType.Test)
        pred = pred.cpu().detach().numpy()
        pred_lroberta.append( pred )

    pred_lroberta = np.mean( np.array(pred_lroberta), axis=0 )

    ##combine models
    pred = []
    for i in range(pred_bert.shape[0]):
        curr = .3*pred_bert[i]+.2*pred_xlnet[i]+.5*pred_roberta[i]
        pred.append(curr)
    pred = np.mean( np.array(pred), axis=0 )
    print ([i for i in pred[0]])
    print ([i for i in pred_lroberta[0]])

    pred = .3*pred+.7*pred_lroberta

    pred = np.argmax(pred, axis=1)
    event = [int(i2l[i]) for i in pred]

    df = pd.read_csv( os.path.join(input_dir, 'test.csv') )
    df['event'] = event
    df.to_csv(os.path.join(output_dir,'solution.csv'), index=False)
