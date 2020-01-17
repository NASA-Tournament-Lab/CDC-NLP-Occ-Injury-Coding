import argparse
import re

import numpy as np
import pandas as pd

from tools.text import clean_text


def parse_arguments(parser):
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--clip_to', type=int, default=44)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    assert (args.output_dir)

    # Importing the raw data
    if args.train_file:
        train = pd.read_csv(args.train_file,
                            usecols=['text', 'event', 'age', 'sex'])
        num_train = train.shape[0]
        ids = np.array(
            [''.join(['record', str(num)]) for num in list(range(num_train))])
        np.random.shuffle(ids)
        train['id'] = ids

        codes = train.event.unique()
        code_dict = dict(zip(codes, np.arange(len(codes))))
        train.event = [code_dict[code] for code in train.event]

        train.text = pd.Series(clean_text(train.text))
        train.text = pd.Series(
            [' '.join(doc.split()[:args.clip_to]) for doc in train.text])

        # Saving the code dict to disk
        code_df = pd.DataFrame.from_dict(code_dict, orient='index')
        code_df['event_code'] = code_df.index
        code_df.columns = ['value', 'event_code']
        code_df.to_csv(args.output_dir + '/code_dict.csv', index=False)

        train['filler'] = np.repeat('a', train.shape[0])
        train = train[['id', 'event', 'filler', 'text', 'age', 'sex']]
        train = train.sample(frac=1)
        train.to_csv(args.output_dir + '/train.tsv',
                     sep='\t',
                     index=False,
                     header=False)

        # Remove some unnecessary beginning
        text = [re.sub('.* year old male', 'male', s) for s in train.text]
        text = [re.sub('.* year old female', 'female', s) for s in text]
        train['text'] = pd.Series(text)
        train['text'].to_csv(args.output_dir + '/train.txt',
                             index=False,
                             header=False)

    if args.test_file:
        test = pd.read_csv(args.test_file, usecols=['text', 'age', 'sex'])
        num_test = test.shape[0]

        # Adding a random identifier for the BERT scripts
        ids = np.array(
            [''.join(['record', str(num)]) for num in list(range(num_test))])
        np.random.shuffle(ids)
        test['id'] = ids
        test.text = pd.Series(clean_text(test.text))
        test.text = pd.Series(
            [' '.join(doc.split()[:args.clip_to]) for doc in test.text])
        test['filler'] = np.repeat('a', test.shape[0])
        test = test[['id', 'text', 'age', 'sex']]
        test.to_csv(args.output_dir + '/test.tsv',
                    sep='\t',
                    index=False,
                    header=True)
