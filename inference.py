import argparse
import os
from pathlib import Path, PosixPath
import pickle

import h5py
import librosa
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

import models
import utils
import sed_utils


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


class InferenceDataset(torch.utils.data.Dataset):

    def __init__(self, wav_df) -> None:
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


def audio_tagging(args):
    print(args)
    # Arugments & parameters
    experiment_path = Path(args.experiment_path)
    audio_path = Path(args.input)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = utils.load_config(os.path.join(experiment_path, "config.yaml"))
    label_encoder_path = config["data"]["train"]["args"]["strong_label_encoder"]
    label_encoder = pickle.load(open(label_encoder_path, "rb"))

    model = getattr(models, config["model"]["type"])(**config["model"]["args"])
    
    checkpoint_path = experiment_path / "checkpoints/best.pth"
    checkpoint = torch.load(checkpoint_path, "cpu")
    model_dict = model.state_dict()
    if "melspec_extractor.spectrogram.window" not in checkpoint['model']:
        for k in ["melspec_extractor.spectrogram.window",
                  "melspec_extractor.mel_scale.fb"]:
            checkpoint["model"][k] = model_dict[k]
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    if audio_path.suffix == ".csv":
        wav_df = pd.read_csv(audio_path, sep="\t")
        dataset = InferenceDataset(wav_df)
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)
        with torch.no_grad(), h5py.File(args.output, "w") as hf:
            for batch in tqdm(dataloader, ascii=True):
                waveform = batch["waveform"].to(device).float()
                output = model(waveform)
                prob_batch = output['clipwise_output'].cpu().numpy()
                for i in range(len(waveform)):
                    aid = batch["audio_id"][i]
                    prob = prob_batch[i]
                    hf[aid] = prob
    else:
        # Load audio
        waveform = load_audio(audio_path)
        waveform = waveform[None, :]
        waveform = torch.as_tensor(waveform).to(device)

        # Forward
        with torch.no_grad():
            batch_output_dict = model(waveform)

        clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
        """(classes_num,)"""

        sorted_indexes = np.argsort(clipwise_output)[::-1]

        # Print audio tagging top probabilities
        for k in range(10):
            print('{}: {:.3f}'.format(label_encoder.classes_[sorted_indexes[k]], 
                clipwise_output[sorted_indexes[k]]))


def sound_event_detection(args):
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_path = Path(args.experiment_path)
    config = utils.load_config(experiment_path / "config.yaml")

    model = getattr(models, config["model"]["type"])(**config["model"]["args"])
    checkpoint = torch.load(experiment_path / "checkpoints/best.pth", "cpu")
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    label_encoder_path = config["data"]["train"]["args"]["strong_label_encoder"]
    label_encoder = pickle.load(open(label_encoder_path, "rb"))

    if not args.input.endswith(".csv"):
        wav_df = pd.DataFrame({
            "audio_id": [Path(args.input).name],
            "file_name": [Path(args.input).absolute().__str__()]
        })
    else:
        wav_df = pd.read_csv(args.input, sep="\t")

    dataset = InferenceDataset(wav_df)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=2)

    if args.output_format == "segment":
        predictions = []
        threshold = args.threshold
        if len(threshold) == 1:
            postprocessing_method = sed_utils.binarize
        elif len(threshold) == 2:
            postprocessing_method = sed_utils.double_threshold
        else:
            raise Exception(f"unknown threshold {threshold}")

        with torch.no_grad(), tqdm(total=len(dataloader), ascii=True) as pbar:
            for batch in dataloader:
                waveform = batch["waveform"].to(device).float()
                output = model(waveform)
                framewise_output = output["framewise_output"].cpu().numpy()
                segmentwise_output = output["segmentwise_output"].cpu().numpy()
                thresholded_predictions = postprocessing_method(
                    framewise_output, *threshold)
                #### SED predictions
                labelled_predictions = sed_utils.decode_with_timestamps(
                    label_encoder, thresholded_predictions)
                for sample_idx in range(len(labelled_predictions)):
                    audio_id = batch["audio_id"][sample_idx]
                    prediction = labelled_predictions[sample_idx]
                    for event_label, onset, offset in prediction:
                        predictions.append({
                            "audio_id": audio_id,
                            "event_label": event_label,
                            "onset": onset,
                            "offset": offset
                        })
                pbar.update()

        output_df = pd.DataFrame(predictions, columns=['audio_id', 'event_label',
                                                       'onset', 'offset'])
        output_df = sed_utils.predictions_to_time(output_df, args.time_resolution)
        output_df.to_csv(args.output, sep="\t", index=False, float_format="%.3f")

        if not args.input.endswith(".csv"):
            import matplotlib.pyplot as plt
            # Plot result
            stft = librosa.core.stft(y=waveform[0].cpu().numpy(), n_fft=1024,
                hop_length=320, window='hann', center=True)
            frames_num = stft.shape[-1]
            
            frames_per_second = int(1 / args.time_resolution)
            framewise_output = framewise_output[0]
            clipwise_output = output["clipwise_output"][0].cpu().numpy()
            sorted_indexes = np.argsort(clipwise_output)[::-1]

            top_k = 10  # Show top results
            top_result_mat = framewise_output[:, sorted_indexes[0 : top_k]]
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
            axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
            axs[0].set_ylabel('Frequency bins')
            axs[0].set_title('Log spectrogram')
            axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
            axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
            axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / frames_per_second))
            axs[1].yaxis.set_ticks(np.arange(0, top_k))
            axs[1].yaxis.set_ticklabels(np.array(label_encoder.classes_)[sorted_indexes[0 : top_k]])
            axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
            axs[1].set_xlabel('Seconds')
            axs[1].xaxis.set_ticks_position('bottom')
            
            fig_path = Path(args.output).with_suffix(".png")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            print('Save sound event detection visualization to {}'.format(fig_path))

            label_to_idx = {lbl: idx for idx, lbl in enumerate(label_encoder.classes_)}
            while True:
                lbl = input("Input the interested class: ")
                try:
                    idx = label_to_idx[lbl]
                except KeyError:
                    print(f"{lbl} not in labels")
                    continue
                plt.figure(figsize=(14, 5))
                prob = segmentwise_output[0, :, idx]
                np.save("prob.npy", prob)
                plt.plot(prob)
                plt.axhline(y=0.5, color='r', linestyle='--')
                duration = waveform.shape[1] / 32000
                xlabels = [f"{x:.2f}" for x in np.arange(0, duration, duration / 5)]
                plt.xticks(ticks=np.arange(0, len(prob), len(prob) / 5),
                           labels=xlabels,
                           fontsize=15)
                plt.title(lbl)
                plt.xlabel("Time / second", fontsize=12)
                plt.ylabel("Probability", fontsize=12)
                plt.ylim(-0.2, 1.2)
                plt.tight_layout()
                plt.savefig(fig_path.with_name("single_class.png"))
    elif args.output_format == "prob":
        with torch.no_grad(), tqdm(total=len(dataloader), ascii=True) as pbar, \
            h5py.File(args.output, "w") as writer:
            for batch in dataloader:
                waveform = batch["waveform"].to(device).float()
                output = model(waveform)
                segmentwise_output = output["segmentwise_output"].cpu().numpy()
                clipwise_output = output["clipwise_output"].cpu().numpy()
                for sample_idx in range(len(waveform)):
                    audio_id = batch["audio_id"][sample_idx]
                    writer[f"clip/{audio_id}"] = clipwise_output[sample_idx]
                    writer[f"segment/{audio_id}"] = segmentwise_output[sample_idx]
                pbar.update()

        print(f"Writing predicted probabilities to {args.output}")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_sed = subparsers.add_parser('sound_event_detection')
    parser_sed.add_argument("--input", type=str)
    parser_sed.add_argument("--experiment_path", type=str)
    parser_sed.add_argument("--output", type=str)
    parser_sed.add_argument("--threshold", type=float, nargs="+", default=[0.75, 0.25])
    parser_sed.add_argument("--time_resolution", type=float, default=0.01)
    parser_sed.add_argument("--output_format", type=str, default="segment",
                            choices=["prob", "segment"])

    parser_at = subparsers.add_parser('audio_tagging')
    parser_at.add_argument("--input", type=str)
    parser_at.add_argument('--experiment_path', type=str, required=True)
    parser_at.add_argument('--output', type=str, required=False)
    
    args = parser.parse_args()

    if args.mode == 'audio_tagging':
        audio_tagging(args)
    elif args.mode == 'sound_event_detection':
        sound_event_detection(args)
    else:
        raise Exception('Error argument!')
