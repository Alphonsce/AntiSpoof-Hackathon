import argparse
import torch
from subm_utils.submission_utils import pad

from subm_utils.submission_dataset import get_data_for_evaldataset, EvalDataset, get_dataloaders
# from model.models import get_model
from subm_utils.submission_metrics import produce_submit_file

from model import Model, ModelWithSEMAA

def get_ssl_aasist_model(ckpt_path, ssl_backbone, ssl_behaviour, device, use_semaa):
    '''
    Initializes Rawformer Architecture from the given checkpoint
    '''
    if not use_semaa:
        model = Model(
            args=None,  # not used
            device=device,
            ssl_backbone=ssl_backbone,
            freeze_ssl=False,
            ssl_behaviour=ssl_behaviour
        )
    else:
        model = ModelWithSEMAA(
            args=None,  # not used
            device=device,
            ssl_backbone=ssl_backbone,
            freeze_ssl=False,
            ssl_behaviour=ssl_behaviour
        )
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    return model

def main(args):
    eval_ids = get_data_for_evaldataset(args.eval_path_wav)

    eval_dataset = EvalDataset(eval_ids, args.eval_path_wav, pad)
    eval_dataset = {"eval": eval_dataset}
    dataloader = get_dataloaders(eval_dataset, args)["eval"]

    model = get_ssl_aasist_model(args.ckpt_path, args.ssl_backbone, args.ssl_behaviour, args.device, args.use_semaa).to(args.device)

    produce_submit_file(dataloader, model, args.device, args.output_file, need_sigmoid=args.need_sigmoid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_semaa", action="store_true", default=False)
    parser.add_argument("--architecture", default="SEMAA", help="Rawformer / SEMAA / wav2vec / dp_hubert")

    parser.add_argument("--ckpt_path", default=None)
    
    parser.add_argument("--ssl_backbone", default="wav2vec")
    parser.add_argument("--ssl_behaviour", default="last-layer")
    
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--need_sigmoid", action="store_true", default=False)

    parser.add_argument(
        "--eval_path_wav", default="/data/a.varlamov/safespeak/wavs", type=str
    )

    parser.add_argument("--output_file", type=str, default="submit.csv")
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    main(args)
