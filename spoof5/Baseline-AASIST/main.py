"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import TestDataset, TrainDataset, genSpoof_list
from eval.calculate_metrics import (calculate_aDCF_tdcf_tEER,
                                    calculate_minDCF_EER_CLLR)
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm import tqdm


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)

    database_path = Path(config["database_path"])
    train_flac_dir = Path(config["train_flac_dir"])
    eval_flac_dir = Path(config["eval_flac_dir"])
    train_meta_file = Path(config["train_meta_file"])
    eval_meta_file = Path(config["eval_meta_file"])

    trn_database_path = database_path / train_flac_dir
    dev_database_path = database_path / eval_flac_dir

    trn_list_path = database_path / train_meta_file
    # Used to create file with scores to compute metrics:
    dev_trial_path = database_path / eval_meta_file

    # define model related paths
    model_tag = "{}_ep{}_bs{}".format(
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"],
        config["batch_size"],
    )
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)
    if args.model_path_train is not None:
        model.load_state_dict(torch.load(args.model_path_train, map_location=device))
        print("Model loaded : {}".format(args.model_path_train))

    # define dataloaders

    trn_loader, dev_loader = get_loader(
        database_path,
        train_flac_dir,
        eval_flac_dir,
        train_meta_file,
        eval_meta_file,
        args.seed,
        config,
        args
    )

    # evaluates pretrained model
    # NOTE: Currently it is evaluated on the development set instead of the evaluation set
    if args.eval:
        model.load_state_dict(torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")
        produce_evaluation_file(
            dev_loader, model, device, eval_score_path, dev_trial_path
        )

        eval_dcf, eval_eer, eval_cllr = calculate_minDCF_EER_CLLR(
            cm_scores_file=eval_score_path,
            output_file=model_tag / "loaded_model_result.txt",
        )
        print(
            "DONE. eval_eer: {:.3f}, eval_dcf:{:.5f} , eval_cllr:{:.5f}".format(
                eval_eer, eval_dcf, eval_cllr
            )
        )

        """
        # Need asv score file for Track 2
        asv_score_path = ""
        eval_adcf, eval_tdcf, eval_teer = calculate_aDCF_tdcf_tEER(
            cm_scores_file=eval_score_path,
            asv_scores_file= asv_score_path,
            output_file=model_tag/"loaded_model_Phase2_result.txt")
        print("DONE. eval_adcf: {:.3f}, eval_tdcf:{:.5f} , eval_teer:{:.5f}".format(eval_adcf, eval_tdcf, eval_teer))
        """
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 100.0
    best_dev_dcf = 1.0
    best_dev_cllr = 1.0
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(config["num_epochs"]):
        print("training epoch{:03d}".format(epoch))

        running_loss = train_epoch(
            trn_loader, model, optimizer, device, scheduler, config
        )

        produce_evaluation_file(
            dev_loader, model, device, metric_path / "dev_score.txt", dev_trial_path
        )
        dev_eer, dev_dcf, dev_cllr = calculate_minDCF_EER_CLLR(
            cm_scores_file=metric_path / "dev_score.txt",
            output_file=metric_path / "dev_DCF_EER_{}epo.txt".format(epoch),
            printout=False,
        )
        print(
            "DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_dcf:{:.5f} , dev_cllr:{:.5f}".format(
                running_loss, dev_eer, dev_dcf, dev_cllr
            )
        )
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_dcf", dev_dcf, epoch)
        writer.add_scalar("dev_cllr", dev_cllr, epoch)
        torch.save(
            model.state_dict(),
            model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer),
        )

        best_dev_dcf = min(dev_dcf, best_dev_dcf)
        best_dev_cllr = min(dev_cllr, best_dev_cllr)
        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer

            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        writer.add_scalar("best_dev_tdcf", best_dev_dcf, epoch)
        writer.add_scalar("best_dev_cllr", best_dev_cllr, epoch)


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print(f"no. model params:{(nb_params / 1000):.3f}k")

    return model


def get_loader(
    database_path: str,
    train_flac_dir,
    eval_flac_dir,
    train_meta_file,
    eval_meta_file,
    seed: int,
    config: dict,
    args: argparse.ArgumentParser
) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement"""

    trn_database_path = database_path / train_flac_dir
    dev_database_path = database_path / eval_flac_dir

    trn_list_path = database_path / train_meta_file
    dev_trial_path = database_path / eval_meta_file

    d_label_trn, file_train = genSpoof_list(
        dir_meta=trn_list_path, is_train=True, is_eval=False
    )
    print("no. training files:", len(file_train))

    train_set = TrainDataset(
        list_IDs=file_train, labels=d_label_trn, base_dir=trn_database_path, use_rawboost=args.use_rawboost, algo_rawboost=args.algo, args=args
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=gen,
    )

    _, file_dev = genSpoof_list(dir_meta=dev_trial_path, is_train=False, is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = TestDataset(list_IDs=file_dev[:2000], base_dir=dev_database_path)
    dev_loader = DataLoader(
        dev_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    return trn_loader, dev_loader


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str,
) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in tqdm(data_loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    # assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            # spk_id, utt_id, _, _, src, key = trl.strip().split(" ")
            # For 2021 eval file: LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval
            spk_id, utt_id, _, _, _, key, _, _ = trl.strip().split(" ")
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(spk_id, utt_id, sco, key))
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace,
):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in tqdm(trn_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument(
        "--config", dest="config", type=str, help="configuration file", required=True
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="random seed (default: 1234)"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit",
    )
    parser.add_argument(
        "--comment", type=str, default=None, help="comment to describe the saved model"
    )
    parser.add_argument(
        "--eval_model_weights",
        type=str,
        default=None,
        help="directory to the model weight file (can be also given in the config file)",
    )
    
    parser.add_argument(
        "--model_path_train",
        type=str,
        default=None,
        help="directory to the model weight file (can be also given in the config file)",
    )

    ##===================================================Rawboost data augmentation parameters======================================================================#

    parser.add_argument(
        "--use_rawboost",
        action="store_true",
        help="To use rawboost augemntation or not",
    )

    parser.add_argument(
        "--algo",
        type=int,
        default=4,
        help="Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]",
    )

    # LnL_convolutive_noise parameters
    parser.add_argument(
        "--nBands",
        type=int,
        default=5,
        help="number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]",
    )
    parser.add_argument(
        "--minF",
        type=int,
        default=20,
        help="minimum centre frequency [Hz] of notch filter.[default=20] ",
    )
    parser.add_argument(
        "--maxF",
        type=int,
        default=8000,
        help="maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]",
    )
    parser.add_argument(
        "--minBW",
        type=int,
        default=100,
        help="minimum width [Hz] of filter.[default=100] ",
    )
    parser.add_argument(
        "--maxBW",
        type=int,
        default=1000,
        help="maximum width [Hz] of filter.[default=1000] ",
    )
    parser.add_argument(
        "--minCoeff",
        type=int,
        default=10,
        help="minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]",
    )
    parser.add_argument(
        "--maxCoeff",
        type=int,
        default=100,
        help="maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]",
    )
    parser.add_argument(
        "--minG",
        type=int,
        default=0,
        help="minimum gain factor of linear component.[default=0]",
    )
    parser.add_argument(
        "--maxG",
        type=int,
        default=0,
        help="maximum gain factor of linear component.[default=0]",
    )
    parser.add_argument(
        "--minBiasLinNonLin",
        type=int,
        default=5,
        help=" minimum gain difference between linear and non-linear components.[default=5]",
    )
    parser.add_argument(
        "--maxBiasLinNonLin",
        type=int,
        default=20,
        help=" maximum gain difference between linear and non-linear components.[default=20]",
    )
    parser.add_argument(
        "--N_f",
        type=int,
        default=5,
        help="order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]",
    )

    # ISD_additive_noise parameters
    parser.add_argument(
        "--P",
        type=int,
        default=10,
        help="Maximum number of uniformly distributed samples in [%].[defaul=10]",
    )
    parser.add_argument(
        "--g_sd", type=int, default=2, help="gain parameters > 0. [default=2]"
    )

    # SSI_additive_noise parameters
    parser.add_argument(
        "--SNRmin",
        type=int,
        default=10,
        help="Minimum SNR value for coloured additive noise.[defaul=10]",
    )
    parser.add_argument(
        "--SNRmax",
        type=int,
        default=40,
        help="Maximum SNR value for coloured additive noise.[defaul=40]",
    )

    ##===================================================Rawboost data augmentation ======================================================================#

    main(parser.parse_args())
