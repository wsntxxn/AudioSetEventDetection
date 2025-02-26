import math
import warnings
import csv
import time
import pickle
import logging
from typing import Dict

import h5py
import scipy
import torch
import numpy as np
import pandas as pd
from tqdm.std import tqdm

import utils


def read_black_list(black_list_csv):
    """Read audio names from black list. 
    """
    with open(black_list_csv, 'r') as fr:
        reader = csv.reader(fr)
        lines = list(reader)

    black_list_names = ['Y{}.wav'.format(line[0]) for line in lines]
    return black_list_names


class AudioSetWeakDataset(object):
    def __init__(self, waveform: str, label: str, label_encoder):
        self.aid_to_h5 = utils.load_dict_from_csv(waveform,
            ["audio_id", "hdf5_path"])
        label_df = pd.read_csv(label, sep="\t")
        self.label_encoder = pickle.load(open(label_encoder, "rb"))
        label_array, _ = utils.encode_labels(label_df["event_labels"],
                                             self.label_encoder)
        self.aid_to_label = dict(zip(label_df["audio_id"], label_array))
        self.aids = set(self.aid_to_h5.keys())
        self.aid_to_label = {k: v for k, v in self.aid_to_label.items()
            if k in self.aids}
        self.dataset_cache = {}
    
    def __getitem__(self, audio_id):
        """Load waveform and target of an audio clip.
        
        Args:
          audio_id: str

        Returns: 
          data_dict: {
            'audio_id': str, 
            'waveform': (n_samples), 
            'target': (classes_num,)}
        """
        hdf5_path = self.aid_to_h5[audio_id]
        if not hdf5_path in self.dataset_cache:
            self.dataset_cache[hdf5_path] = h5py.File(hdf5_path, "r")
        waveform = self.dataset_cache[hdf5_path][audio_id][()]
        target = self.aid_to_label[audio_id]
        if scipy.sparse.issparse(target):
            target = target.toarray().squeeze(0)
        target = target.astype(np.float32)

        data_dict = {
            "audio_name": audio_id, "waveform": waveform, "target": target}
            
        return data_dict


class AudioSetStrongDataset(AudioSetWeakDataset):

    def __init__(self, waveform: str, weak_label: str, strong_label: str,
            weak_label_encoder: str, strong_label_encoder: str,
            time_resolution: float):
        super().__init__(waveform, weak_label, weak_label_encoder)
        self.strong_df = pd.read_csv(strong_label, sep="\t")
        self.aids = set(self.strong_df["audio_id"].values)
        self.aid_to_h5 = {k: v for k, v in self.aid_to_h5.items()
            if k in self.aids}
        self.aid_to_label = {k: v for k, v in self.aid_to_label.items()
            if k in self.aids}
        self.time_resolution = time_resolution
        self.target_length = int(10. / time_resolution)
        self.strong_label_encoder = pickle.load(open(strong_label_encoder, "rb"))
        self.label_to_idx = {label: idx for idx, label in enumerate(
            self.strong_label_encoder.classes_)}

    def __getitem__(self, audio_id):
        data_dict = super().__getitem__(audio_id)
        data_dict["weak_target"] = data_dict["target"]
        del data_dict["target"]
        data_df = self.strong_df[self.strong_df["audio_id"].isin([audio_id])]
        strong_target = np.zeros((self.target_length, len(self.label_to_idx)))
        for _, row in data_df.iterrows():
            onset = round(row["onset"] / self.time_resolution)
            offset = round(row["offset"] / self.time_resolution)
            event_label = row["event_label"]
            # event_label = event_label.replace(",", "").replace(" ", "_")
            if event_label not in self.label_to_idx:
                warnings.warn(f"{event_label} not in the training data")
                continue
            target_idx = self.label_to_idx[event_label]
            strong_target[onset: offset, target_idx] = 1
        data_dict["strong_target"] = strong_target
        return data_dict


class BaseSampler(object):
    def __init__(self, aid_to_label, batch_size, black_list_csv, random_seed):
        """Base class of train sampler.
        
        Args:
          aid_to_label: {
              <audio_id>: <target>
          }
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        # Black list
        if black_list_csv:
            self.black_list_names = read_black_list(black_list_csv)
        else:
            self.black_list_names = []

        logging.info('Black list samples: {}'.format(len(self.black_list_names)))

        # Load target
        load_time = time.time()

        self.audio_ids = []
        self.audios_num = len(aid_to_label)
        self.classes_num = next(iter(aid_to_label.values())).shape[-1]
        self.targets = np.zeros((self.audios_num, self.classes_num)).astype(np.float32)

        for idx, (audio_id, target) in enumerate(aid_to_label.items()):
            self.audio_ids.append(audio_id)
            if scipy.sparse.issparse(target):
                target = target.toarray().squeeze(0)
            self.targets[idx] = target

        logging.info('Training number: {}'.format(self.audios_num))
        logging.info('Load target time: {:.3f} s'.format(time.time() - load_time))


class TrainSampler(BaseSampler):
    def __init__(self, aid_to_label, batch_size, black_list_csv=None,
        random_seed=1234):
        """Balanced sampler. Generate batch meta for training.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        super().__init__(aid_to_label, batch_size, 
            black_list_csv, random_seed)
        
        self.indexes = np.arange(self.audios_num)
            
        # Shuffle indexes
        self.random_state.shuffle(self.indexes)
        
        self.pointer = 0

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_idx: e.g.: [
            <audio_id>: str,
            ...]
        """
        batch_size = self.batch_size

        while True:
            batch_aids = []
            i = 0
            while i < batch_size:
                index = self.indexes[self.pointer]
                self.pointer += 1

                # Shuffle indexes and reset pointer
                if self.pointer >= self.audios_num:
                    self.pointer = 0
                    self.random_state.shuffle(self.indexes)
                
                # If audio in black list then continue
                if self.audio_ids[index] in self.black_list_names:
                    continue
                else:
                    batch_aids.append(self.audio_ids[index])
                    i += 1

            yield batch_aids

    def state_dict(self):
        state = {
            'indexes': self.indexes,
            'pointer': self.pointer}
        return state
            
    def load_state_dict(self, state):
        self.indexes = state['indexes']
        self.pointer = state['pointer']


class BalancedTrainSampler(BaseSampler):
    def __init__(self, aid_to_label, batch_size, black_list_csv=None, 
        random_seed=1234):
        """Balanced sampler. Generate batch meta for training. Data are equally 
        sampled from different sound classes.
        
        Args:
          aid_to_label: dict
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        super().__init__(aid_to_label, batch_size, black_list_csv, random_seed)
        
        self.samples_num_per_class = np.sum(self.targets, axis=0)
        logging.info('samples_num_per_class: {}'.format(
            self.samples_num_per_class.astype(np.int32)))
        
        # Training indexes of all sound classes. E.g.: 
        # [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...]
        self.indexes_per_class = []
        
        for k in range(self.classes_num):
            self.indexes_per_class.append(
                np.where(self.targets[:, k] == 1)[0])
            
        # Shuffle indexes
        for k in range(self.classes_num):
            self.random_state.shuffle(self.indexes_per_class[k])
        
        self.queue = []
        self.pointers_of_classes = [0] * self.classes_num

    def expand_queue(self, queue):
        classes_set = np.arange(self.classes_num).tolist()
        self.random_state.shuffle(classes_set)
        queue += classes_set
        return queue

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_idx: e.g.: [
            <audio_id>: str,
            ...]
        """
        batch_size = self.batch_size

        while True:
            batch_aids = []
            i = 0
            while i < batch_size:
                if len(self.queue) == 0:
                    self.queue = self.expand_queue(self.queue)

                class_id = self.queue.pop(0)
                pointer = self.pointers_of_classes[class_id]
                self.pointers_of_classes[class_id] += 1
                index = self.indexes_per_class[class_id][pointer]
                
                # When finish one epoch of a sound class, then shuffle its indexes and reset pointer
                if self.pointers_of_classes[class_id] >= self.samples_num_per_class[class_id]:
                    self.pointers_of_classes[class_id] = 0
                    self.random_state.shuffle(self.indexes_per_class[class_id])

                # If audio in black list then continue
                if self.audio_ids[index] in self.black_list_names:
                    continue
                else:
                    batch_aids.append(self.audio_ids[index])
                    i += 1

            yield batch_aids

    def state_dict(self):
        state = {
            'indexes_per_class': self.indexes_per_class, 
            'queue': self.queue, 
            'pointers_of_classes': self.pointers_of_classes}
        return state
            
    def load_state_dict(self, state):
        self.indexes_per_class = state['indexes_per_class']
        self.queue = state['queue']
        self.pointers_of_classes = state['pointers_of_classes']


class AlternateTrainSampler(BaseSampler):
    def __init__(self, aid_to_label, batch_size, black_list_csv=None,
        random_seed=1234):
        """AlternateSampler is a combination of Sampler and Balanced Sampler. 
        AlternateSampler alternately sample data from Sampler and Blanced Sampler.
        
        Args:
          indexes_hdf5_path: string          
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        self.sampler1 = TrainSampler(aid_to_label, batch_size, 
            black_list_csv, random_seed)

        self.sampler2 = BalancedTrainSampler(aid_to_label, batch_size, 
            black_list_csv, random_seed)

        self.batch_size = batch_size
        self.count = 0

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_idx: e.g.: [
            <audio_id>: str,
            ...]
        """
        batch_size = self.batch_size

        while True:
            self.count += 1

            if self.count % 2 == 0:
                batch_aids = []
                i = 0
                while i < batch_size:
                    index = self.sampler1.indexes[self.sampler1.pointer]
                    self.sampler1.pointer += 1

                    # Shuffle indexes and reset pointer
                    if self.sampler1.pointer >= self.sampler1.audios_num:
                        self.sampler1.pointer = 0
                        self.sampler1.random_state.shuffle(self.sampler1.indexes)
                    
                    # If audio in black list then continue
                    if self.sampler1.audio_ids[index] in self.sampler1.black_list_names:
                        continue
                    else:
                        batch_aids.append(self.sampler1.audio_ids[index])
                        i += 1

            elif self.count % 2 == 1:
                batch_aids = []
                i = 0
                while i < batch_size:
                    if len(self.sampler2.queue) == 0:
                        self.sampler2.queue = self.sampler2.expand_queue(self.sampler2.queue)

                    class_id = self.sampler2.queue.pop(0)
                    pointer = self.sampler2.pointers_of_classes[class_id]
                    self.sampler2.pointers_of_classes[class_id] += 1
                    index = self.sampler2.indexes_per_class[class_id][pointer]
                    
                    # When finish one epoch of a sound class, then shuffle its indexes and reset pointer
                    if self.sampler2.pointers_of_classes[class_id] >= self.sampler2.samples_num_per_class[class_id]:
                        self.sampler2.pointers_of_classes[class_id] = 0
                        self.sampler2.random_state.shuffle(self.sampler2.indexes_per_class[class_id])

                    # If audio in black list then continue
                    if self.sampler2.audio_ids[index] in self.sampler2.black_list_names:
                        continue
                    else:
                        batch_aids.append(self.sampler2.audio_ids[index])
                        i += 1

            yield batch_aids

    def state_dict(self):
        state = {
            'sampler1': self.sampler1.state_dict(), 
            'sampler2': self.sampler2.state_dict()}
        return state

    def load_state_dict(self, state):
        self.sampler1.load_state_dict(state['sampler1'])
        self.sampler2.load_state_dict(state['sampler2'])


class EvaluateSampler(object):
    def __init__(self, aid_to_label, batch_size):
        """Evaluate sampler. Generate batch meta for evaluation.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
        """
        self.batch_size = batch_size

        self.audio_ids = []
        self.audios_num = len(aid_to_label)
        self.classes_num = next(iter(aid_to_label.values())).shape[-1]
        self.targets = np.zeros((self.audios_num, self.classes_num))

        for idx, (audio_id, target) in enumerate(aid_to_label.items()):
            self.audio_ids.append(audio_id)
            if scipy.sparse.issparse(target):
                target = target.toarray().squeeze(0)
            self.targets[idx] = target

    def __iter__(self):
        """Generate batch audio ids for evaluation. 
        
        Returns:
          batch_idx: e.g.: [
            <audio_id>: str,
            ...]
        """
        batch_size = self.batch_size
        pointer = 0

        while pointer < self.audios_num:
            batch_indexes = np.arange(pointer, 
                min(pointer + batch_size, self.audios_num))

            batch_aids = []

            for index in batch_indexes:
                batch_aids.append(self.audio_ids[index])

            pointer += batch_size
            yield batch_aids

    def __len__(self):
        return math.ceil(self.audios_num / self.batch_size)


def collate_fn(list_data_dict):
    """Collate data.
    Args:
      list_data_dict, e.g., [{'audio_name': str, 'feature': (n_frame, n_mel), ...}, 
                             {'audio_name': str, 'feature': (n_frame, n_mel), ...},
                             ...]
    Returns:
      np_data_dict, dict, e.g.,
          {'audio_name': (batch_size,), 'feature': (batch_size, n_frame, n_mel), ...}
    """
    np_data_dict = {}
    
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    return np_data_dict


def pad(data_list, batch_first=True, padding_value=0.):
    tensor_list = [torch.tensor(_) for _ in data_list]
    padded_seq = torch.nn.utils.rnn.pad_sequence(tensor_list,
                                                 batch_first=batch_first,
                                                 padding_value=padding_value)
    return padded_seq.numpy()


def pad_collate_fn(list_data_dict):
    np_data_dict = {}
    
    for key in list_data_dict[0].keys():
        if isinstance(list_data_dict[0][key], np.ndarray) and \
            list_data_dict[0][key].ndim > 1:
            np_data_dict[key] = pad([data_dict[key] for data_dict in list_data_dict])
        else:
            np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    return np_data_dict


if __name__ == "__main__":
    import argparse
    import random
    import torch

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    num_workers = args.num_workers
    batch_size = args.batch_size

    config = {
        # "waveform": "./data/train/waveform.csv",
        "waveform": "/mnt/lustre/sjtu/home/xnx98/work/AudioTagging/audioset_tagging/data/audioset/full_train/waveform.csv",
        "weak_label": "data/train/label_train_weak.csv",
        "strong_label": "./data/train/label.csv",
        "weak_label_encoder": "data/train/label_encoder.pkl",
        "strong_label_encoder": "./data/train/label_encoder.pkl",
        "time_resolution": 0.01
    }
    dset = AudioSetStrongDataset(**config)
    batch_sampler = BalancedTrainSampler(dset.aid_to_label, batch_size)
    dataloader = torch.utils.data.DataLoader(
        dset, batch_sampler=batch_sampler,
        collate_fn=collate_fn, num_workers=num_workers)

    total_iters = 100
    # total_iters = len(dataloader)
    # print(len(dataloader))

    start = time.time()
    iteration = 0
    with tqdm(total=total_iters, ascii=True) as pbar:
        for batch_data_dict in dataloader:
            iteration += 1
            pbar.set_postfix(iteration=iteration)
            pbar.update()
            # print("iteration: ", iteration, end="  ")
            # print("waveform: ", batch_data_dict["waveform"].shape, end="  ")
            # print("weak target: ", batch_data_dict["weak_target"].shape, end="  ")
            # print("strong target: ", batch_data_dict["strong_target"].shape)
            if iteration == total_iters:
                break
    end = time.time()
    print(f"{total_iters} iterations of loading uses {end - start} seconds")
        
