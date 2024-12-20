import argparse
import os
import sys

import numpy as np
import torch
import yaml
from core_scripts.startup_config import set_random_seed
from data_utils_SSL import (Dataset_ASVspoof2019_train,
                            Dataset_ASVspoof2021_eval, genSpoof_list)
from tensorboardX import SummaryWriter
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Model, ModelWithSEMAA

__author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"


def evaluate_accuracy(dev_loader, model, device, total):
    val_loss = 0.0
    num_total = 0.0
    batch_loss = 100
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in tqdm(
        dev_loader, desc=f"Eval Epoch, eval-loss: {val_loss}", total=total
    ):

        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)

        batch_loss = criterion(batch_out, batch_y)
        val_loss += batch_loss.item() * batch_size

    val_loss /= num_total

    return val_loss


def produce_evaluation_file(dataset, model, device, save_path, total, batch_size):
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    num_correct = 0.0
    num_total = 0.0
    model.eval()

    fname_list = []
    key_list = []
    score_list = []

    for batch_x, utt_id in tqdm(data_loader, desc="Evaluation", total=total):
        fname_list = []
        score_list = []
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)

        batch_out = model(batch_x)

        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

        with open(save_path, "a+") as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write("{} {}\n".format(f, cm))
        fh.close()
    print("Scores saved to {}".format(save_path))


def train_epoch(train_loader, model, lr, optim, device, total, model_save_path):
    running_loss = 0
    num_total = 0.0
    batch_loss = 100

    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for i, (batch_x, batch_y) in tqdm(
        enumerate(train_loader),
        desc=f"Train epoch, Train-Loss: {running_loss}",
        total=total,
    ):

        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)

        batch_loss = criterion(batch_out, batch_y)

        running_loss += batch_loss.item() * batch_size

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # if i % 8000 == 0 and i > 0:
        #     torch.save(
        #         model.state_dict(),
        #         os.path.join(f"{model_save_path}", f"ep_{epoch}_steps_{i}_ckpt.pth"),
        #     )

    running_loss /= num_total

    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof2021 baseline system")

    parser.add_argument("--use_semaa", action="store_true", default=False)

    # Dataset
    parser.add_argument(
        "--train_year",
        type=str,
        default="2019",
    )

    parser.add_argument(
        "--database_path",
        type=str,
        default="/data/a.varlamov/asvspoof/",
        help="Change this to user's full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.",
    )
    """
    % database_path/
    %   |- LA
    %      |- ASVspoof2021_LA_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac

 
 
    """

    parser.add_argument(
        "--protocols_path",
        type=str,
        default="/data/a.varlamov/asvspoof/",
        help="Change with path to user's LA database protocols directory address",
    )
    """
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
  
    """

    ### SSL-related:
    parser.add_argument("--ssl_backbone", default="wav2vec")
    parser.add_argument("--freeze_ssl", action="store_true", default=False)
    parser.add_argument(
        "--ssl_behaviour",
        default="last-layer",
        help="Only for DPHubert. Can also be 'weighted-sum' .",
    )

    # =============

    # Hyperparameters
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--loss", type=str, default="weighted_CCE")
    # model
    parser.add_argument(
        "--seed", type=int, default=1234, help="random seed (default: 1234)"
    )

    parser.add_argument("--model_path", type=str, default=None, help="Model checkpoint")
    parser.add_argument(
        "--comment", type=str, default=None, help="Comment to describe the saved model"
    )
    # Auxiliary arguments
    parser.add_argument(
        "--track", type=str, default="LA", choices=["LA", "PA", "DF"], help="LA/PA/DF"
    )
    parser.add_argument(
        "--eval_output",
        type=str,
        default=None,
        help="Path to save the evaluation result",
    )
    parser.add_argument("--eval", action="store_true", default=False, help="eval mode")
    parser.add_argument(
        "--is_eval", action="store_true", default=False, help="eval database"
    )
    parser.add_argument("--eval_part", type=int, default=0)
    # backend options
    parser.add_argument(
        "--cudnn-deterministic-toggle",
        action="store_false",
        default=True,
        help="use cudnn-deterministic? (default true)",
    )

    parser.add_argument(
        "--cudnn-benchmark-toggle",
        action="store_true",
        default=False,
        help="use cudnn-benchmark? (default false)",
    )

    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument(
        "--algo",
        type=int,
        default=5,
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

    if not os.path.exists("models"):
        os.mkdir("models")
    args = parser.parse_args()

    # make experiment reproducible
    set_random_seed(args.seed, args)

    track = args.track

    assert track in ["LA", "PA", "DF"], "Invalid track given"

    # database
    prefix = "ASVspoof_{}".format(track)
    prefix_2019 = "ASVspoof2019.{}".format(track)
    prefix_2021 = "ASVspoof2021.{}".format(track)

    # define model saving path
    model_tag = "model_{}_{}_{}_{}_{}".format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr
    )
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_save_path = os.path.join("models", model_tag)

    # set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # GPU device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    if not args.use_semaa:
        model = Model(
            args,
            device,
            ssl_backbone=args.ssl_backbone,
            freeze_ssl=args.freeze_ssl,
            ssl_behaviour=args.ssl_behaviour,
        )
    else:
        print("using model with SEMAA")
        model = ModelWithSEMAA(
            args,
            device,
            ssl_backbone=args.ssl_backbone,
            freeze_ssl=args.freeze_ssl,
            ssl_behaviour=args.ssl_behaviour,
        )
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = model.to(device)
    print("nb_params:", nb_params)

    # set Adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Model loaded : {}".format(args.model_path))

    # evaluation
    eval_protocol = (
        args.protocols_path + "ASVspoof2021_LA_eval/keys/LA/CM/trial_metadata.txt"
    )
    if args.eval:
        file_eval = genSpoof_list(
            dir_meta=eval_protocol,
            is_train=False,
            is_eval=True,
        )
        print("no. of eval trials", len(file_eval))
        eval_path = args.database_path + f"ASVspoof2021_{args.track}_eval/"
        eval_set = Dataset_ASVspoof2021_eval(
            list_IDs=file_eval,
            base_dir=eval_path,
        )
        produce_evaluation_file(
            eval_set,
            model,
            device,
            args.eval_output,
            total=len(file_eval) // args.batch_size + 1,
            batch_size=args.batch_size,
        )
        sys.exit(0)

    # define train dataloader
    if args.train_year == "2019":
        train_protocol = (
            args.protocols_path
            + "2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
        )
    elif args.train_year == "2021":
        train_protocol = (
            args.protocols_path + "ASVspoof2021_LA_eval/keys/LA/CM/trial_metadata.txt"
        )

    elif args.train_year == "2021_DF":
        train_protocol = (
            args.protocols_path + "ASVspoof2021_LA_eval/keys/DF/CM/trial_metadata.txt"
        )

    elif args.train_year == "2021_DF_BIG":
        train_protocol = args.protocols_path + "ASVspoof2021_DF_eval/all_data_meta.csv"

    d_label_trn, file_train = genSpoof_list(
        dir_meta=train_protocol,
        is_train=True,
        is_eval=False,
    )

    print("no. of training trials", len(file_train))
    if args.train_year == "2019":
        train_path = args.database_path + "/2019_LA/ASVspoof2019_LA_train/"
    elif args.train_year == "2021":
        train_path = args.database_path + f"ASVspoof2021_{args.track}_eval/"
    elif args.train_year == "2021_DF" or args.train_year == "2021_DF_BIG":
        train_path = args.database_path + f"ASVspoof2021_DF_eval/"

    train_set = Dataset_ASVspoof2019_train(
        args,
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=train_path,
        algo=args.algo,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )

    del train_set, d_label_trn

    # define validation dataloader
    dev_protocol = (
        args.protocols_path
        + "2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
    )
    d_label_dev, file_dev = genSpoof_list(
        dir_meta=dev_protocol,
        is_train=False,
        is_eval=False,
    )

    print("no. of validation trials", len(file_dev))

    val_path = args.database_path + "/2019_LA/ASVspoof2019_LA_dev/"
    dev_set = Dataset_ASVspoof2019_train(
        args,
        list_IDs=file_dev,
        labels=d_label_dev,
        base_dir=val_path,
        algo=args.algo,
    )
    dev_loader = DataLoader(
        dev_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )
    del dev_set, d_label_dev

    # Training and validation
    num_epochs = args.num_epochs
    # writer = SummaryWriter("logs/{}".format(model_tag))

    for epoch in tqdm(range(num_epochs), desc="Epoch Number:..."):

        running_loss = train_epoch(
            train_loader,
            model,
            args.lr,
            optimizer,
            device,
            total=len(file_train) // args.batch_size + 1,
            model_save_path=model_save_path,
        )
        val_loss = evaluate_accuracy(
            dev_loader, model, device, total=len(file_dev) // args.batch_size + 1
        )
        # writer.add_scalar("val_loss", val_loss, epoch)
        # writer.add_scalar("loss", running_loss, epoch)
        print("\n{} - {} - {} ".format(epoch, running_loss, val_loss))
        torch.save(
            model.state_dict(),
            os.path.join(f"{model_save_path}", f"ep_{epoch}_ckpt.pth"),
        )
