import argparse

import torch
from dataset import EvalDataset, get_data_for_evaldataset, get_dataloaders
# from model.models import get_model
from metrics import produce_submit_file
from utils import load_checkpoint, pad

from rawformer_asv_spoof.config import ExpConfig, SysConfig
# =========== Rawformer =====
from rawformer_asv_spoof.models.rawformer import Rawformer_SE


def get_rawformer_model(ckpt_path, device):
    """
    Initializes Rawformer Architecture from the given checkpoint
    """
    sys_config, exp_config = config.SysConfig(), config.ExpConfig()
    state_dict = torch.load(ckpt_path, weights_only=True, map_location=device)
    new_state_dict = {
        key.replace("module.", "").replace("se_fc", "se_module.fc"): value
        for key, value in state_dict.items()
    }

    model = Rawformer_SE(
        device=device,
        sample_rate=exp_config.sample_rate,
        transformer_hidden=exp_config.transformer_hidden,
    )
    model = model.load_state_dict(new_state_dict)

    return model


def main(args):
    eval_ids = get_data_for_evaldataset(args.eval_path_wav)

    eval_dataset = EvalDataset(eval_ids, args.eval_path_wav, pad)
    eval_dataset = {"eval": eval_dataset}
    dataloader = get_dataloaders(eval_dataset, config)

    # model = get_model(config).to(args.device)
    model = get_rawformer_model(args.ckpt_path, args.device)

    produce_submit_file(dataloader["eval"], model, args.device, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--architecture",
        default="Rawformer",
        help="Rawformer / SEMAA / wav2vec / dp_hubert",
    )

    parser.add_argument("--ckpt_path", default=None)

    parser.add_argument("--device", default="cpu")

    parser.add_argument("--need_sigmoid", action="store_true", default=False)

    parser.add_argument(
        "--eval_path_wav", default="/data/a.varlamov/safespeak/wavs", type=str
    )

    parser.add_argument("--output_file", type=str, default="submit.csv")

    args = parser.parse_args()
    config = load_checkpoint(args.config)
    main(args, config)
