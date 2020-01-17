import argparse
import logging
from pathlib import Path

import pandas as pd

from processing import preprocess_symspell

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def process_train_data(train_path: Path, output_path: Path):
    output_path.mkdir(exist_ok=True, parents=True)
    train = pd.read_csv(train_path).reset_index()
    train["text_data"] = train["text"].apply(preprocess_symspell)
    logging.info(f"Training data processed. N={len(train)}")
    train["dummy"] = "a"
    train[["index", "event", "dummy", "text_data"]].to_csv(
        output_path / "train.tsv", sep="\t", header=False, index=False
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "train_data_file",
        default=None,
        type=str,
        nargs=1,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "output_path",
        default=None,
        type=str,
        nargs=1,
        help="Output of the parsed training data",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_data_path = Path(args.train_data_file[0])
    output_path = Path(args.output_path[0])
    process_train_data(train_data_path, output_path)


if __name__ == "__main__":
    main()
