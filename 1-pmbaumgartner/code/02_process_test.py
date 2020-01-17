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


def process_train_data(input_path: Path, output_path: Path):
    output_path.mkdir(exist_ok=True, parents=True)

    test = pd.read_csv(input_path).reset_index()
    test["text_data"] = test["text"].apply(preprocess_symspell)
    logging.info(f"Test data processed. N={len(test)}")
    test["dummy"] = "a"
    test["event"] = "10"
    test[["index", "event", "dummy", "text_data"]].to_csv(
        output_path / "dev.tsv", sep="\t", header=False, index=False
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "test_data_file",
        default=None,
        type=str,
        nargs=1,
        help="The test data file to generate predictions on.",
    )
    parser.add_argument(
        "output_path",
        default=None,
        type=str,
        nargs=1,
        help="Output of the parsed test files data",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    test_data_file = Path(args.test_data_file[0])
    output_path = Path(args.output_path[0])
    process_train_data(test_data_file, output_path)


if __name__ == "__main__":
    main()
