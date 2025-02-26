import os
import sys
# sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import argparse
import time
import random
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
 
import utils
import models
from pytorch_utils import (move_data_to_device, count_parameters, count_flops, 
    do_mixup)
import dataset
import losses



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate(dataloader, model, loss_fn, device):
    loss_history = []
    clipwise_outputs = []
    weak_targets = []
    model.eval()
    loss_fn.eval()
    with torch.no_grad():
        for batch_data_dict in tqdm(dataloader):
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key],
                                                           device)
            batch_output = model(batch_data_dict["waveform"])
            batch_output.update(batch_data_dict)
            loss = loss_fn(batch_output)
            loss_history.append(loss.item())
            clipwise_outputs.append(batch_output["clipwise_output"].cpu().numpy())
            weak_targets.append(batch_output["weak_target"].cpu().numpy())
    loss = np.mean(loss_history)
    clipwise_outputs = np.concatenate(clipwise_outputs)
    weak_targets = np.concatenate(weak_targets)
    average_precision = metrics.average_precision_score(
        weak_targets, clipwise_outputs, average=None
    )
    if np.isnan(average_precision).sum() > 0:
        mAP = average_precision[~np.isnan(average_precision)].mean()
    else:
        mAP = average_precision.mean()
    return {"loss": loss, "mAP": mAP}


def train(args):
    """Train AudioSet tagging model.

    Args:
      dataset_dir: str
    """
    
    seed = args.seed
    set_seed(seed)

    # Arugments & parameters
    config = utils.load_config(args.config_file)
    experiment_path = config["experiment_path"]
    mixup = config["mixup"]
    specaug = config["specaug"]
    iterations = config["iterations"]
    device = torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')

    loss_fn = getattr(losses, config["loss"]["type"])(**config["loss"]["args"])

    checkpoints_dir = os.path.join(experiment_path, "checkpoints")
    utils.create_folder(checkpoints_dir)
    
    logger = utils.get_logger(os.path.join(experiment_path, "train.log"))
    for line in yaml.dump(config, indent=4).split("\n"):
        logger.info(line)

    if "SLURM_JOB_ID" in os.environ:
        logger.info(f"Slurm job id: {os.environ['SLURM_JOB_ID']}")
        logger.info(f"Slurm node: {os.environ['SLURM_JOB_NODELIST']}")

    
    if 'cuda' in str(device):
        logger.info('Using GPU.')
        device = 'cuda'
    else:
        logger.info('Using CPU. Set --cuda flag to use GPU.')
        device = 'cpu'
    
    dataloaders = {}
    for split in ["train", "val"]:
        dset = getattr(dataset, config["data"][split]["type"])(
            **config["data"][split]["args"])
        sampler_config = config["data"][split]["batch_sampler"]
        batch_sampler = getattr(dataset, sampler_config["type"])(
            dset.aid_to_label, **sampler_config["args"])
        collate_fn = getattr(dataset, config["data"][split]["collate_fn"])
        dataloader = torch.utils.data.DataLoader(
            dset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            **config["data"]["dataloader_args"])
        dataloaders[split] = dataloader

    with open(os.path.join(experiment_path, "config.yaml"), "w") as writer:
        yaml.dump(config, writer, default_flow_style=False, indent=4)

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    model = getattr(models, config["model"]["type"])(**config["model"]["args"])

    params_num = count_parameters(model)
    # flops_num = count_flops(model, clip_samples)
    logger.info('Parameters num: {}'.format(params_num))
    # logging.info('Flops num: {:.3f} G'.format(flops_num / 1e9))

    if mixup:
        mixup_augmenter = utils.Mixup(mixup_alpha=1.)

    # Statistics
    statistics_path = os.path.join(experiment_path, "statistics.pkl")
    statistics_container = utils.StatisticsContainer(statistics_path)

    # Optimizer
    optimizer = getattr(optim, config["optimizer"]["type"])(
        model.parameters(), **config["optimizer"]["args"])
    lr_scheduler = getattr(optim.lr_scheduler, config["lr_scheduler"]["type"])(
        optimizer, **config["lr_scheduler"]["args"])

    train_bgn_time = time.time()

    if "resume_config" in config:
        checkpoint_path = config["resume_config"]["checkpoint"]
        logger.info('Loading checkpoint {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "model" in checkpoint:
            # audio tagging checkpoints
            model_dict = model.state_dict()
            state_dict = checkpoint["model"]
            pretrained_dict = {
                k: v for k, v in state_dict.items() if k in model_dict and
                    model_dict[k].shape == v.shape
            }
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            # clap checkpoints
            state_dict = model.state_dict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("audio_encoder."):
                    model_key = key.replace("audio_encoder.", "")
                    state_dict[model_key] = value
            model.load_state_dict(state_dict)

        if config["resume_config"]["finetune"]:
            iteration = 0
        else:
            train_loader.batch_sampler.load_state_dict(checkpoint['sampler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            statistics_container.load_state_dict(config["resume_config"][
                "statistics"])
            iteration = checkpoint["iteration"]
    else:
        iteration = 0

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)

    time1 = time.time()
    tb_writer = SummaryWriter(os.path.join(experiment_path, "run"))
    loss_history = []
    min_val_loss = np.inf
    not_improve_cnt = 0

    for batch_data_dict in train_loader:

        # Evaluate
        if iteration % config["eval_interval"] == 0 or iteration == 0:
            train_fin_time = time.time()

            if len(loss_history) > 0:
                train_loss = np.mean(loss_history)
                logger.info(f'Train loss: {train_loss:.3g}')
                train_loss = []

            val_statistics = validate(val_loader, model, loss_fn, device)

            lr = optimizer.param_groups[0]["lr"]

            logger.info('Validate loss: {:.3g}, mAP: {:.3g}, Lr: {:.2g}'.format(
                val_statistics['loss'], val_statistics["mAP"], lr)
            )
            tb_writer.add_scalar("loss/val", val_statistics['loss'],
                iteration // config["eval_interval"])

            statistics_container.append(iteration, val_statistics, data_type='val')
            statistics_container.dump(verbose=False)

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logger.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(iteration, train_time, validate_time))

            if val_statistics["loss"] < min_val_loss:
                min_val_loss = val_statistics["loss"]
                not_improve_cnt = 0
                checkpoint = {
                    "iteration": iteration,
                    "model": model.module.state_dict() if \
                        torch.cuda.device_count() > 1 else model.state_dict(),
                    "sampler": train_loader.batch_sampler.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict()
                }

                checkpoint_path = os.path.join(checkpoints_dir, 'best.pth')
                    
                torch.save(checkpoint, checkpoint_path)
                logger.info('Model saved to {}'.format(checkpoint_path))
            else:
                not_improve_cnt += 1

            logger.info('------------------------------------')

            lr_scheduler.step(val_statistics['loss'])

            if not_improve_cnt == config["early_stop"]:
                break

            train_bgn_time = time.time()

        # Save model
        if iteration % config["save_interval"] == 0:
            checkpoint = {
                "iteration": iteration,
                "model": model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                "sampler": train_loader.batch_sampler.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict()
            }

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logger.info('Model saved to {}'.format(checkpoint_path))

        # Mixup lambda
        if mixup:
            batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                batch_size=len(batch_data_dict['waveform']))

        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

        # Forward
        model.train()
        loss_fn.train()

        if mixup:
            batch_output_dict = model(
                batch_data_dict['waveform'],
                batch_data_dict['mixup_lambda'],
                specaug)
            batch_target_dict = {
                'weak_target': do_mixup(batch_data_dict['weak_target'],
                                        batch_data_dict['mixup_lambda']),
                'strong_target': do_mixup(batch_data_dict['strong_target'],
                                          batch_data_dict['mixup_lambda']),
            }
            batch_output_dict.update(batch_target_dict)
        else:
            batch_output_dict = model(batch_data_dict['waveform'], None, specaug)
            batch_output_dict.update(batch_data_dict)

        # Loss
        loss = loss_fn(batch_output_dict)

        # Backward
        loss.backward()
        # print(loss)
        tb_writer.add_scalar("loss/train", loss.item(), iteration)
        loss_history.append(loss.item())

        optimizer.step()
        optimizer.zero_grad()

        if iteration % config["print_interval"] == 0:
            sys.stdout.flush()
            print('--- Iteration: {}, train time: {:.3f} s / 100 iterations ---'\
                .format(iteration, time.time() - time1))
            time1 = time.time()

        # Stop learning
        if iteration == iterations:
            break

        iteration += 1


def evaluate(args):
    # Arugments & parameters

    experiment_path = args.experiment_path
    device = torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')
    config = utils.load_config(os.path.join(experiment_path, "config.yaml"))
    checkpoints_dir = os.path.join(experiment_path, "checkpoints")

    eval_config = utils.load_config(args.eval_config_file)

    dset = getattr(dataset, eval_config["data"]["type"])(
        **eval_config["data"]["args"])
    sampler_config = eval_config["data"]["batch_sampler"]
    batch_sampler = getattr(dataset, sampler_config["type"])(
        dset.aid_to_label, **sampler_config["args"])
    collate_fn = getattr(dataset, eval_config["data"]["collate_fn"])
    dataloader = torch.utils.data.DataLoader(
        dset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn, 
        **eval_config["data"]["dataloader_args"])

    # Model
    model = getattr(models, config["model"]["type"])(**config["model"]["args"])
    checkpoint_path = os.path.join(checkpoints_dir, 'best.pth')
    print('Loading checkpoint {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, "cpu")
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    segment_length = eval_config["data"]["args"]["time_resolution"]
    clipwise_outputs = []
    segmentwise_outputs = []
    weak_targets = []
    strong_targets = []
    with torch.no_grad():
        for batch in tqdm(dataloader, ascii=True):
            for key in batch.keys():
                batch[key] = move_data_to_device(batch[key], device)
            output = model(batch["waveform"], segment_length=segment_length)
            segmentwise_outputs.append(output["segmentwise_output"].cpu().numpy())
            strong_targets.append(batch["strong_target"].cpu().numpy())
            clipwise_outputs.append(output["clipwise_output"].cpu().numpy())
            weak_targets.append(batch["weak_target"].cpu().numpy())

    segmentwise_outputs = np.concatenate(segmentwise_outputs)
    classes_num = segmentwise_outputs.shape[-1]
    segmentwise_outputs = segmentwise_outputs.reshape(-1, classes_num)
    strong_targets = np.concatenate(strong_targets)
    strong_targets = strong_targets.reshape(-1, classes_num)
    clipwise_outputs = np.concatenate(clipwise_outputs)
    weak_targets = np.concatenate(weak_targets)

    average_precision = metrics.average_precision_score(
        weak_targets, clipwise_outputs, average=None
    )
    label_mask = ~np.isnan(average_precision)
    mAP = average_precision[label_mask].mean()
    auc = metrics.roc_auc_score(strong_targets[:, label_mask],
                                segmentwise_outputs[:, label_mask])
    d_prime = utils.d_prime(auc)
    lwlrap = utils.lwlrap(strong_targets[:, label_mask],
                          segmentwise_outputs[:, label_mask])

    with open(os.path.join(experiment_path, "eval_result.txt"), "w") as fp:
        print(f'Weak mAP: {mAP:.3f}', file=fp)
        print(f'Strong d prime: {d_prime:.3f}', file=fp)
        print(f'Strong lwlrap: {lwlrap:.3f}', file=fp)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--config_file', type=str, required=True)
    parser_train.add_argument('--seed', type=int, default=1)

    parser_evaluate = subparsers.add_parser('evaluate')
    parser_evaluate.add_argument('--experiment_path', type=str, required=True)
    parser_evaluate.add_argument('--eval_config_file', type=str, required=True)
    
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    else:
        raise Exception('Error argument!')
