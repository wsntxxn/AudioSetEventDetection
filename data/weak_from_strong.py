import argparse
from pathlib import Path
import pandas as pd


def main(args):
    strong_df = pd.read_csv(args.strong_label, sep="\t")
    weak_df = strong_df.groupby(["audio_id"])["event_label"].unique().apply(
        lambda x: ";".join(x)).to_frame().reset_index()
    weak_df["event_labels"] = weak_df["event_label"]
    weak_df.drop(["event_label"], axis=1, inplace=True)
    input_fname = Path(args.strong_label).stem + "_weak.csv"
    if args.weak_label is None:
        output_csv = Path(args.strong_label).with_name(input_fname)
    else:
        output_csv = args.weak_label
    print(f"writting to {output_csv}")
    weak_df.to_csv(output_csv, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("strong_label", type=str)
    parser.add_argument("--weak_label", type=str, required=False, default=None)
    args = parser.parse_args()
    main(args)
