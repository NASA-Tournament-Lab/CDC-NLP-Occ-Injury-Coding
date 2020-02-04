import argparse
import pandas as pd
import numpy as np
import os

from text import clean_text


from nltk.tokenize import RegexpTokenizer
import re

def parse_arguments(parser):
    parser.add_argument('--data_dir',type=str,  default='data')
    parser.add_argument('--test_file', type=str, default='data/test.csv')
    parser.add_argument('--train_file', type=str, default='data/train.csv')
    parser.add_argument('--phase',type=str,default='test')
    parser.add_argument('--output_dir',type=str,  default='data_tmp')
    args = parser.parse_args()
    return args

def preprocess(sentence):
    sentence=sentence.replace('{html}',"")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    return " ".join(tokens)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    
    if args.phase=='train':
        print('Loading ' + args.train_file)
        # Importing the raw data
        train = pd.read_csv(args.train_file,
                        usecols=['text', 'event'])
        
        num_train = train.shape[0]
        
        ids = np.array([''.join(['record', str(num)]) 
                    for num in list(range(num_train))])
        
        np.random.shuffle(ids)
        train['id'] = ids[0:num_train]
        
        print('cleaning text')
        train['text'] = train['text'].apply(lambda t: preprocess((t)))
        
        train.text = pd.Series(clean_text(train.text))
        
        print('clipping')
        train_lengths = np.array([len(doc.split()) for doc in train.text])
        clip_to = np.max(train_lengths)
        train.text = pd.Series([' '.join(doc.split()[:clip_to])
                                for doc in train.text])
        
        
        # Making a lookup dictionary for the event codes
        code_df = pd.read_csv(os.path.join(args.data_dir,'code_descriptions.csv'))
        codes = code_df.event.values
        print(codes)
        code_dict = dict(zip(codes, np.arange(len(codes))))
        print(code_dict)
        train.event = [code_dict[code] for code in train.event]
        

#         # Saving the code dict to disk
#         code_df = pd.DataFrame.from_dict(code_dict, orient='index')
#         code_df['event_code'] = code_df.index
#         code_df.columns = ['value', 'event_code']
#         code_df.to_csv(args.data_dir + 'code_dict.csv', index=False)

        # Rearranging the columns for BERT
        train['filler'] = np.repeat('a', train.shape[0])
        
        train = train[['id', 'event', 'filler', 'text']]
        
        # Shuffling the rows
        train = train.sample(frac=1)
        
        # Writing the regular splits to disk
        train.to_csv(os.path.join(args.output_dir ,'train.tsv'), sep='\t', 
                     index=False, header=False)
        
    elif args.phase=='test':
        print('Loading ' + args.test_file)
        test = pd.read_csv(args.test_file,
                           usecols=['text'])

        # Adding a random identifier for the BERT scripts

        num_test = test.shape[0]
        ids = np.array([''.join(['record', str(num)]) 
                        for num in list(range(num_test))])
        np.random.shuffle(ids)
        test['id'] = ids[0:num_test]

        print('cleaning text')
        test['text'] = test['text'].apply(lambda t: preprocess((t)))

        # Lowercasing and adding spaces around common abbreviations;
        # only fixes a few things


        test.text = pd.Series(clean_text(test.text))

        print('clipping')
        # Clipping the docs to the max length
        test_lengths = np.array([len(doc.split()) for doc in test.text])
        clip_to = np.max(test_lengths)
        
        test.text = pd.Series([' '.join(doc.split()[:clip_to])
                              for doc in test.text])


        test['filler'] = np.repeat('a', test.shape[0])

        test = test[['id', 'text']]

        test.to_csv(os.path.join(args.output_dir ,'test.tsv'), sep='\t', 
                    index=False, header=True)


    print("completed preprocessing")
