import argparse

import sentencepiece as spm


def parse_arguments(parser):
    parser.add_argument('--train_file', type=str, default='train.csv')
    parser.add_argument('--model_prefix', type=str, default='cdc')
    parser.add_argument('--vocab_size', type=int, default=500)
    parser.add_argument('--model_type', type=str, default='unigram')
    parser.add_argument('--split_by_whitespace', type=str, default='true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    spm.SentencePieceTrainer.train(
        '--input={} --model_prefix={} --split_by_whitespace={} --vocab_size={} --model_type={}'
        .format(args.train_file, args.model_prefix, args.split_by_whitespace,
                args.vocab_size, args.model_type))
