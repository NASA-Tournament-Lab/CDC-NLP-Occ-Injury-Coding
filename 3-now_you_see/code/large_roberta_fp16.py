# -*- coding: utf-8 -*-
from __future__ import print_function
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0';
import torch
torch.cuda.is_available()

import torch.nn as nn

import numpy as np
import pandas as pd
from collections import Counter
import re
import codecs
import random
import string
import joblib
import argparse
import gc

from xlnet import clean_text

SEED = 1
random.seed(1)
np.random.seed(1)
MAX_LEN = 32
model_name = 'roberta-large'

from fastai.text import *
from fastai.metrics import *
from transformers import RobertaTokenizer, RobertaModel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Creating a config object to store task specific information
class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

class FastAiRobertaTokenizer(BaseTokenizer):
    """Wrapper around RobertaTokenizer to be compatible with fastai"""
    def __init__(self, tokenizer: RobertaTokenizer, max_seq_len: int=128, **kwargs): 
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len 
    def __call__(self, *args, **kwargs): 
        return self 
    def tokenizer(self, t:str) -> List[str]: 
        """Adds Roberta bos and eos tokens and limits the maximum sequence length""" 
        return ["<s>"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["</s>"]

# Setting up pre-processors
class RobertaTokenizeProcessor(TokenizeProcessor):
    def __init__(self, tokenizer):
         super().__init__(tokenizer=tokenizer, include_bos=False, include_eos=False)

class RobertaNumericalizeProcessor(NumericalizeProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, vocab=fastai_roberta_vocab, **kwargs)

def get_roberta_processor(tokenizer:Tokenizer=None, vocab:Vocab=None):
    """
    Constructing preprocessors for Roberta
    We remove sos and eos tokens since we add that ourselves in the tokenizer.
    We also use a custom vocabulary to match the numericalization with the original Roberta model.
    """
    return [RobertaTokenizeProcessor(tokenizer=tokenizer), NumericalizeProcessor(vocab=vocab)]

# Creating a Roberta specific DataBunch class
class RobertaDataBunch(TextDataBunch):
    "Create a `TextDataBunch` suitable for training Roberta"
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', bs:int=64, val_bs:int=None, pad_idx=1,
               pad_first=True, device:torch.device=None, no_check:bool=False, backwards:bool=False, 
               dl_tfms:Optional[Collection[Callable]]=None, **dl_kwargs) -> DataBunch:
        "Function that transform the `datasets` in a `DataBunch` for classification. Passes `**dl_kwargs` on to `DataLoader()`"
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)
        collate_fn = partial(pad_collate, pad_idx=pad_idx, pad_first=pad_first, backwards=backwards)
        train_sampler = SortishSampler(datasets[0].x, key=lambda t: len(datasets[0][t][0].data), bs=bs)
        train_dl = DataLoader(datasets[0], batch_size=bs, sampler=train_sampler, drop_last=True, **dl_kwargs)
        dataloaders = [train_dl]
        for ds in datasets[1:]:
            lengths = [len(t) for t in ds.x.items]
            sampler = SortSampler(ds.x, key=lengths.__getitem__)
            dataloaders.append(DataLoader(ds, batch_size=val_bs, sampler=sampler, **dl_kwargs))
        return cls(*dataloaders, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)

class RobertaTextList(TextList):
    _bunch = RobertaDataBunch
    _label_cls = TextList

# defining our model architecture 
class CustomRobertaModel(nn.Module):
    def __init__(self,num_labels=2):
        super(CustomRobertaModel,self).__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(config.roberta_model_name)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels) # defining final output layer
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _ , pooled_output = self.roberta(input_ids, token_type_ids, attention_mask) # 
        logits = self.classifier(pooled_output)        
        return logits

if __name__== '__main__':
    cwd = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--bag')
    args = parser.parse_args()

    input_dir = args.input_dir
    bag = int(args.bag)

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

    mask = df.event.apply(lambda s: s in l2i).values
    print (df.shape)
    df = df.iloc[mask, :]
    df.reset_index(inplace=True, drop=True)
    print (df.shape)
    df.event = df.event.apply(lambda s: l2i[s])
    set_seed(11+bag)

    config = Config(
        testing=False,
        seed = 10,
        roberta_model_name=model_name, # can also be exchnaged with roberta-large 
        max_lr=3e-5,
        epochs=5,
        use_fp16=False,
        bs=32,
        max_seq_len=32,
        num_labels = len(i2l),
        hidden_dropout_prob=.05,
        hidden_size=1024, #  for roberta-large
        start_tok = "<s>",
        end_tok = "</s>",
    )

    feat_cols = "text"
    label_cols = "event"

    # create fastai tokenizer for roberta
    roberta_tok = RobertaTokenizer.from_pretrained(model_name)

    fastai_tokenizer = Tokenizer(tok_func=FastAiRobertaTokenizer(roberta_tok, max_seq_len=config.max_seq_len), 
                                 pre_rules=[], post_rules=[])

    # create fastai vocabulary for roberta
    path = Path('models')
    roberta_tok.save_vocabulary(path)

    with open( os.path.join(path, 'vocab.json'), 'r') as f:
        roberta_vocab_dict = json.load(f)
        
    fastai_roberta_vocab = Vocab(list(roberta_vocab_dict.keys()))

    # loading the tokenizer and vocab processors
    processor = get_roberta_processor(tokenizer=fastai_tokenizer, vocab=fastai_roberta_vocab)

    # creating our databunch 
    data = RobertaTextList.from_df(df, ".", cols=feat_cols, processor=processor) \
        .split_none() \
        .label_from_df(cols=label_cols,label_cls=CategoryList) \
        .databunch(bs=config.bs, pad_first=False, pad_idx=0)

    roberta_model = CustomRobertaModel(num_labels=config.num_labels)
    learn = Learner(data, roberta_model, metrics=[accuracy])
    learn.unfreeze()
    learn = learn.to_fp16()

    learn.model.roberta.train() # setting roberta to train as it is in eval mode by default
    learn.fit_one_cycle(config.epochs, max_lr=config.max_lr)

    out_file = 'models/large-roberta-fp16-{}.pkl'.format(bag)

    learn.export( file=out_file )
