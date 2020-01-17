import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def process_train_data(preds_path: Path, test_csv: Path, output_file: Path):
    df_paths = list(preds_path.glob("**/plogits.csv"))
    dfs = [pd.read_csv(p, index_col=0) for p in df_paths]
    concated = pd.concat(dfs, axis=1)
    logging.info(f"Concatenated logits from {len(dfs)} models.")
    pavg = pd.DataFrame()
    if len(dfs) > 1:
        for e in dfs[0].columns:
            pavg[e] = concated[e].mean(axis=1)
    else:
        pavg[e] = concated[e]
    ensemble_predictions = pavg.idxmax(axis=1)
    test = pd.read_csv(test_csv)
    test["event"] = ensemble_predictions.tolist()
    test_export = test[["text", "sex", "age", "event"]]
    logging.info(f"Test data processed. N={len(test)}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    test_export.to_csv(str(output_file), index=False)
    logging.info(f"Predictions exported to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "predictions_path",
        default=None,
        type=str,
        nargs=1,
        help="The directory of predictions from the model ensemble.",
    )
    parser.add_argument(
        "test_csv", default=None, type=str, nargs=1, help="The Test CSV."
    )
    parser.add_argument(
        "output_path",
        default=None,
        type=str,
        nargs=1,
        help="Output of the solution file. Should be a filename like `solution.csv`",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    predictions_path = Path(args.predictions_path[0])
    test_csv = Path(args.test_csv[0])
    output_path = Path(args.output_path[0])
    process_train_data(predictions_path, test_csv, output_path)


if __name__ == "__main__":
    main()
