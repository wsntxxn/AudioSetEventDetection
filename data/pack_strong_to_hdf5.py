import argparse
import math
from pathlib import Path

import pandas as pd
import h5py
from tqdm import tqdm


def main(args):
    waveform_df = pd.read_csv(args.all_waveform, sep="\t")
    aid_to_h5 = dict(zip(waveform_df["audio_id"], waveform_df["hdf5_path"]))

    df = pd.read_csv(args.label_csv, sep="\t")
    
    all_aids = df["audio_id"].unique()
    num_files = len(all_aids)

    if num_files < args.part_size:
        csv_data = []
        output_h5 = args.output
        with h5py.File(output_h5, "w") as writer:
            for audio_id in tqdm(df["audio_id"].unique(), ascii=True):
                if audio_id not in aid_to_h5:
                    continue
                hdf5_path = aid_to_h5[audio_id]
                if args.softlink:
                    writer[audio_id] = h5py.ExternalLink(hdf5_path, audio_id)
                else:
                    with h5py.File(hdf5_path, "r") as reader:
                        data = reader[audio_id][()]
                    writer[audio_id] = data
                csv_data.append({
                    "audio_id": audio_id,
                    "hdf5_path": Path(output_h5).absolute().__str__()
                })
        output_csv = Path(output_h5).with_suffix(".csv")
        pd.DataFrame(csv_data).to_csv(output_csv, sep="\t", index=False)
    else:
        output = Path(args.output)
        if not output.exists():
            output.mkdir()
            (output / "hdf5").mkdir()
            (output / "csv").mkdir()
        for index in range(1, 1 + math.ceil(num_files / args.part_size)):
            csv_data = []
            start = (index - 1) * args.part_size
            end = index * args.part_size
            part_aids = all_aids[start: end]
            output_h5 = output / f"hdf5/{index}.h5"
            with h5py.File(output_h5, "w") as writer:
                for audio_id in tqdm(part_aids, ascii=True):
                    if audio_id not in aid_to_h5:
                        continue
                    hdf5_path = aid_to_h5[audio_id]
                    if args.softlink:
                        writer[audio_id] = h5py.ExternalLink(hdf5_path, audio_id)
                    else:
                        with h5py.File(hdf5_path, "r") as reader:
                            data = reader[audio_id][()]
                        writer[audio_id] = data
                    csv_data.append({
                        "audio_id": audio_id,
                        "hdf5_path": Path(output_h5).absolute().__str__()
                    })
            output_csv = output / f"csv/{index}.csv"
            pd.DataFrame(csv_data).to_csv(output_csv, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_csv", type=str, default="./train/label.csv")
    parser.add_argument("--output", type=str, default="./train/waveform.h5")
    parser.add_argument("--all_waveform", type=str,)
    parser.add_argument("--softlink", default=False, action="store_true")
    parser.add_argument("--part_size", type=int, default=50000)

    args = parser.parse_args()
    main(args)
