import os
import sys
import logging
import six
import datetime
import pickle
import collections

import yaml
from loguru import logger
import numpy as np
import pandas as pd
from scipy import stats 
import sklearn.preprocessing as pre
from sklearn import metrics


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
        
def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def get_sub_filepaths(folder):
    paths = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            paths.append(path)
    return paths


def get_logger(log_file):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if log_file:
        logger.add(log_file, enqueue=True, format=log_format)
    return logger


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging


def read_metadata(csv_path, classes_num, id_to_ix):
    """Read metadata of AudioSet from a csv file.

    Args:
      csv_path: str

    Returns:
      meta_dict: {'audio_name': (audios_num,), 'target': (audios_num, classes_num)}
    """

    with open(csv_path, 'r') as fr:
        lines = fr.readlines()
        lines = lines[3:]   # Remove heads

    audios_num = len(lines)
    targets = np.zeros((audios_num, classes_num), dtype=np.bool)
    audio_names = []
 
    for n, line in enumerate(lines):
        items = line.split(', ')
        """items: ['--4gqARaEJE', '0.000', '10.000', '"/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"\n']"""

        # audio_name = 'Y{}.wav'.format(items[0])   # Audios are started with an extra 'Y' when downloading
        audio_name = f'{items[0]}_{float(items[1]):.3f}_{float(items[2]):.3f}.wav'
        label_ids = items[3].split('"')[1].split(',')

        audio_names.append(audio_name)

        # Target
        for id in label_ids:
            ix = id_to_ix[id]
            targets[n, ix] = 1
    
    meta_dict = {'audio_name': np.array(audio_names), 'target': targets}
    return meta_dict


def float32_to_int16(x):
    # assert np.max(np.abs(x)) <= 1.2
    x = np.clip(x, -1, 1)
    return (x * 32767.).astype(np.int16)


def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)
    

def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
    else:
        return x[0 : audio_length]


def d_prime(auc):
    d_prime = stats.norm().ppf(auc) * np.sqrt(2.0)
    return d_prime


def lwlrap(target, output):
    sample_weight = np.sum(target > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    lwlrap = metrics.label_ranking_average_precision_score(
        target[nonzero_weight_sample_indices, :] > 0,
        output[nonzero_weight_sample_indices, :],
        sample_weight=sample_weight[nonzero_weight_sample_indices])
    return lwlrap


def encode_labels(labels, encoder=None, sep=";", sparse=True):
    """encode_labels

    Encodes labels

    :param labels: pd.Series representing the raw labels e.g., Speech, Water
    :param encoder (optional): Encoder already fitted 
    returns encoded labels (many hot) and the encoder
    """
    assert isinstance(labels, pd.Series), "Labels need to be series"
    instance = labels.iloc[0]
    if isinstance(instance, six.string_types):
        # In case of using non processed strings, e.g., Vaccum, Speech
        label_array = labels.str.split(sep).values.tolist()
    elif isinstance(instance, np.ndarray):
        # Encoder does not like to see numpy array
        label_array = [lab.tolist() for lab in labels]
    elif isinstance(instance, collections.Iterable):
        label_array = labels
    if not encoder:
        encoder = pre.MultiLabelBinarizer(sparse_output=sparse)
        encoder.fit(label_array)
    labels_encoded = encoder.transform(label_array)
    return labels_encoded, encoder


def merge_a_into_b(a, b):
    # merge dict a into dict b. values in a will overwrite b.
    for k, v in a.items():
        if isinstance(v, dict) and k in b:
            assert isinstance(
                b[k], dict
            ), "Cannot inherit key '{}' from base!".format(k)
            merge_a_into_b(v, b[k])
        else:
            b[k] = v


def load_config(config_file):
    with open(config_file, "r") as reader:
        config = yaml.load(reader, Loader=yaml.FullLoader)
    if "inherit_from" in config:
        base_config_file = config["inherit_from"]
        base_config_file = os.path.join(
            os.path.dirname(config_file), base_config_file
        )
        assert not os.path.samefile(config_file, base_config_file), \
            "inherit from itself"
        base_config = load_config(base_config_file)
        del config["inherit_from"]
        merge_a_into_b(config, base_config)
        return base_config
    return config


def load_dict_from_csv(filename, cols):
    df = pd.read_csv(filename, sep="\t")
    output = dict(zip(df[cols[0]], df[cols[1]]))
    return output


class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return np.array(mixup_lambdas)


class StatisticsContainer(object):
    def __init__(self, statistics_path):
        """Contain statistics of different training iterations.
        """
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pkl'.format(
            os.path.splitext(self.statistics_path)[0],
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.statistics_dict = {'val': []}

    def append(self, iteration, statistics, data_type):
        statistics['iteration'] = iteration
        self.statistics_dict[data_type].append(statistics)
        
    def dump(self, verbose=True):
        pickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        pickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        if verbose:
            logging.info('    Dump statistics to {}'.format(self.statistics_path))
            logging.info('    Dump statistics to {}'.format(self.backup_statistics_path))
        
    def load_state_dict(self, statistics_checkpoint):
        self.statistics_dict = pickle.load(open(statistics_checkpoint, 'rb'))

        # resume_statistics_dict = {'bal': [], 'test': []}
        
        # for key in self.statistics_dict.keys():
            # for statistics in self.statistics_dict[key]:
                # if statistics['iteration'] <= resume_iteration:
                    # resume_statistics_dict[key].append(statistics)
                
        # self.statistics_dict = resume_statistics_dict
