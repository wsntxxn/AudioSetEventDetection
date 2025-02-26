import os
from pathlib import Path, PosixPath
import argparse
import pickle

import numpy as np
import pandas as pd
import librosa
import h5py
import torch
from tqdm import tqdm
from psds_eval import PSDSEval

import utils
import models
import sed_utils


desed_label_to_asstrong = {
    "Speech": ["Speech", 'Babbling', 'Child speech, kid speaking', 'Conversation', 
               'Female speech, woman speaking', 'Male speech, man speaking', 
               'Narration, monologue', 'Speech synthesizer'],
    "Frying": ["Frying (food)", 'Chopping (food)', 'Microwave oven'],
    "Dishes": ["Dishes, pots, and pans"],
    "Running_water": ["Water", 'Ocean', 'Rain', 'Rain on surface', 'Raindrop', 'Steam',
                      'Waterfall', 'Waves, surf'],
    "Blender": ["Blender, food processor"],
    "Electric_shaver_toothbrush": ["Electric shaver, electric razor"],
    "Alarm_bell_ringing": ["Alarm", 'Alarm clock', 'Busy signal', 'Buzzer', 'Car alarm',
                           'Civil defense siren', 'Dial tone', 'Fire alarm', 'Foghorn',
                           'Ringtone', 'Siren', 'Smoke detector, smoke alarm', 'Telephone',
                           'Telephone bell ringing', 'Telephone dialing, DTMF', 'Whistle'],
    "Cat": ["Cat", 'Caterwaul', 'Hiss', 'Meow', 'Purr'],
    "Dog": ["Dog", 'Bark', 'Bow-wow', 'Growling', 'Howl', 'Yip'],
    "Vacuum_cleaner": ["Vacuum cleaner"],
}


class DesedEncoder:
    
    def __init__(self):
        self.classes_ = [
            "Speech",
            "Frying",
            "Dishes",
            "Running_water",
            "Blender",
            "Electric_shaver_toothbrush",
            "Alarm_bell_ringing",
            "Cat",
            "Dog",
            "Vacuum_cleaner"
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}


def load_audio(audio_path):
    if isinstance(audio_path, PosixPath):
        audio_path = audio_path.__str__()
    waveform, _ = librosa.core.load(audio_path, sr=32000)
    return waveform


def read_from_h5(key, key_to_h5, cache):
    hdf5_path = key_to_h5[key]
    if hdf5_path not in cache:
        cache[hdf5_path] = h5py.File(hdf5_path, "r")
    try:
        return cache[hdf5_path][key][()]
    except KeyError: # audiocaps compatibility
        key = "Y" + key + ".wav"
        return cache[hdf5_path][key][()]


class InferDataset(torch.utils.data.Dataset):

    def __init__(self, wav_df):
        super().__init__()
        if "file_name" in wav_df.columns:
            self.aid_to_fname = dict(zip(wav_df["audio_id"],
                                         wav_df["file_name"]))
        elif "hdf5_path" in wav_df.columns:
            self.aid_to_h5 = dict(zip(wav_df["audio_id"],
                                      wav_df["hdf5_path"]))
            self.h5_cache = {}
        self.aids = wav_df["audio_id"].unique()

    def __len__(self):
        return len(self.aids)

    def __getitem__(self, index):
        audio_id = self.aids[index]
        if hasattr(self, "aid_to_fname"):
            waveform = load_audio(self.aid_to_fname[audio_id])
        elif hasattr(self, "aid_to_h5"):
            waveform = read_from_h5(audio_id, self.aid_to_h5, self.h5_cache)
        return {"audio_id": audio_id, "waveform": waveform}


def evaluate(args):
    experiment_path = args.experiment_path
    device = torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')
    config = utils.load_config(os.path.join(experiment_path, "config.yaml"))
    checkpoints_dir = os.path.join(experiment_path, "checkpoints")

    wav_df = pd.read_csv(args.waveform, sep="\t")
    dataset = InferDataset(wav_df)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=4
    )

    model = getattr(models, config["model"]["type"])(**config["model"]["args"])
    checkpoint_path = os.path.join(checkpoints_dir, 'best.pth')
    print('Loading checkpoint {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, "cpu")
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    threshold = args.threshold
    if len(threshold) == 1:
        postprocessing_method = sed_utils.binarize
    elif len(threshold) == 2:
        postprocessing_method = sed_utils.double_threshold
    else:
        raise Exception(f"unknown threshold {threshold}")

    label_encoder_path = config["data"]["train"]["args"]["strong_label_encoder"]
    label_encoder = pickle.load(open(label_encoder_path, "rb"))

    desed_encoder = DesedEncoder()

    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, ascii=True):
            waveform = batch["waveform"].to(device).float()
            output = model(waveform)
            framewise_output = output["framewise_output"].cpu().numpy()
            probs = np.zeros((framewise_output.shape[0], framewise_output.shape[1], 10))
            for desed_label, as_labels in desed_label_to_asstrong.items():
                multi_hot = label_encoder.transform([as_labels])
                indices = np.where(multi_hot)[1]
                prob_class = framewise_output[..., indices].max(axis=-1)[0]
                probs[..., desed_encoder.class_to_idx[desed_label]] = prob_class

            thresholded_predictions = postprocessing_method(
                probs, *threshold)
            #### SED predictions
            labelled_predictions = sed_utils.decode_with_timestamps(
                desed_encoder, thresholded_predictions)
            for sample_idx in range(len(labelled_predictions)):
                audio_id = batch["audio_id"][sample_idx]
                prediction = labelled_predictions[sample_idx]
                for event_label, onset, offset in prediction:
                    predictions.append({
                        "filename": audio_id,
                        "event_label": event_label,
                        "onset": onset,
                        "offset": offset
                    })
    output_df = pd.DataFrame(predictions, columns=['filename', 'event_label',
                                                   'onset', 'offset'])
    output_df = sed_utils.predictions_to_time(output_df, args.time_resolution)
    
    if not Path(args.output_pred).parent.exists():
        Path(args.output_pred).parent.mkdir(parents=True)
    output_df.to_csv(args.output_pred, sep="\t", index=False, float_format="%.3f")

    groundtruth = pd.read_csv(args.label, sep="\t")
    metadata = pd.read_csv(args.metadata, sep="\t")
    psds_eval = PSDSEval(ground_truth=groundtruth, metadata=metadata)
    macro_f, class_f = psds_eval.compute_macro_f_score(output_df)
    
    if not Path(args.output_score).parent.exists():
        Path(args.output_score).parent.mkdir(parents=True)
    with open(args.output_score, "w") as writer:
        print(f"macro F-score: {macro_f*100:.2f}", file=writer)
        print(f"macro F-score: {macro_f*100:.2f}")
        for clsname, f in class_f.items():
            print(f"  {clsname}: {f*100:.2f}", file=writer)
            print(f"  {clsname}: {f*100:.2f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", "-exp", type=str, required=True)
    parser.add_argument("--waveform", "-w", type=str, required=True)
    parser.add_argument("--label", "-l", type=str, required=True)
    parser.add_argument("--output_pred", "-o_p", type=str, required=True)
    parser.add_argument("--output_score", "-o_s", type=str, required=True)
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.75, 0.25])
    parser.add_argument("--metadata", "-m", type=str, required=True)
    parser.add_argument("--time_resolution", "-t", type=float, default=0.01)
    
    args = parser.parse_args()
    evaluate(args)
