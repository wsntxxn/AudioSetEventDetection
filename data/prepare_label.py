import argparse
from pathlib import Path
import pickle

import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer


def split_train_val(
        data_frame: pd.DataFrame,
        frac: float = 0.9,
        label_array=None,  # Only for stratified, computes necessary split
        stratified=True):
    if stratified:
        # Use statified sampling
        from skmultilearn.model_selection import iterative_train_test_split
        index_train, _, index_val, _ = iterative_train_test_split(
            data_frame.index.values.reshape(-1, 1), label_array, test_size=1. - frac)
        train_data = data_frame[data_frame.index.isin(index_train.squeeze())]
        val_data = data_frame[data_frame.index.isin(index_val.squeeze())]
    else:
        # Simply split train_test
        train_data = data_frame.sample(frac=frac, random_state=0)
        val_data = data_frame[~data_frame.index.isin(train_data.index)]
    return train_data, val_data


def main(args):
    label_dir = Path(args.label_dir)
    mid_to_label = {}
    with open(label_dir / "mid_to_display_name.tsv", "r") as reader:
        for line in reader.readlines():
            mid, label = line.strip().split("\t")
            mid_to_label[mid] = label
    
    train_data = []
    with open(label_dir / "train_strong.tsv", "r") as reader:
        lines = reader.readlines()
        for line in tqdm(lines[1:], ascii=True):
            segment_id, start, end, mid = line.strip().split()
            yid = segment_id[:11]
            train_data.append({
                "audio_id": "Y" + yid + ".wav",
                "onset": float(start),
                "offset": float(end),
                "event_label": mid_to_label[mid]
            })
    train_df = pd.DataFrame(train_data)
    train_df.to_csv("./train/label.csv", sep="\t", index=False)
    
    weak_df = train_df.groupby(["audio_id"])["event_label"].unique().apply(
        lambda x: ";".join(x)).to_frame().reset_index()
    label_array = weak_df["event_label"].str.split(';').values.tolist()
    label_encoder = MultiLabelBinarizer()
    label_encoder.fit(label_array)
    label_array = label_encoder.transform(label_array)
    val_number = args.val_number
    percent = 1.0 - val_number / weak_df.shape[0]
    stratified = not args.no_stratified
    train_subdf, val_subdf = split_train_val(weak_df, percent,
                                             label_array, stratified)
    print("train audio clips: ", train_subdf.shape[0])
    print("val audio clips: ", val_subdf.shape[0])
    train_df[train_df["audio_id"].isin(train_subdf["audio_id"].unique())].to_csv(
        "./train/label_train.csv", sep="\t", index=False)
    train_df[train_df["audio_id"].isin(val_subdf["audio_id"].unique())].to_csv(
        "./train/label_val.csv", sep="\t", index=False)
    pickle.dump(label_encoder, open("./train/label_encoder.pkl", "wb"))


    eval_data = []
    with open(label_dir / "eval_strong.tsv", "r") as reader:
        lines = reader.readlines()
        for line in tqdm(lines[1:], ascii=True):
            segment_id, start, end, mid = line.strip().split()
            yid = segment_id[:11]
            eval_data.append({
                "audio_id": "Y" + yid + ".wav",
                "onset": float(start),
                "offset": float(end),
                "event_label": mid_to_label[mid]
            })
    pd.DataFrame(eval_data).to_csv("./eval/label.csv", sep="\t", index=False)

    eval_framed_posneg_data = []
    with open(label_dir / "eval_strong_framed_posneg.tsv", "r") as reader:
        lines = reader.readlines()
        for line in tqdm(lines[1:], ascii=True):
            segment_id, start, end, mid, present = line.strip().split()
            yid = segment_id[:11]
            if present == "PRESENT":
                eval_framed_posneg_data.append({
                    "audio_id": "Y" + yid + ".wav",
                    "onset": float(start),
                    "offset": float(end),
                    "event_label": mid_to_label[mid]
                })
    pd.DataFrame(eval_framed_posneg_data).to_csv("./eval/label_framed_posneg.csv",
                                                 sep="\t",
                                                 index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label_dir", type=str)
    parser.add_argument("--val_number", type=int, default=5000)
    parser.add_argument("--no_stratified", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
