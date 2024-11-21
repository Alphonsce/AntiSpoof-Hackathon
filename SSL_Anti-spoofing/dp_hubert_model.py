import torch
import torch.nn as nn

from DPHuBERT.wav2vec2.model import wav2vec2_model


class DPHubertModel(nn.Module):
    def __init__(self, device, behaviour="last-layer", freeze=False):
        """
        Args:
            device: obvious...
            behaviour: last-layer / weighted-sum
            freeze: to freeze weights of the pre-train or not
                for weighted-sum freezing will not let weights of sum train
        """
        super(DPHubertModel, self).__init__()

        ckpt_path = "./DPHuBERT/checkpoints/DPHuBERT-sp0.75.pth"
        ckpt = torch.load(ckpt_path)
        self.model = wav2vec2_model(**ckpt["config"]).to(device)
        self.device = device
        self.out_dim = 768
        self.n_layers = 12
        self.behaviour = behaviour

        if behaviour == "weighted-sum":
            self.sum_weights = (
                nn.parameter.Parameter(torch.tensor([0.0] * 9 + [0.5, 0.5, 0.5]))
                .reshape(self.n_layers, 1, 1, 1)
                .to(device)
            )

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def extract_feat(self, input_data):

        # put the model to GPU if it not there
        if (
            next(self.model.parameters()).device != input_data.device
            or next(self.model.parameters()).dtype != input_data.dtype
        ):
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        if True:
            # input should be in shape (batch, length)
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data

            # [batch, length, dim]
            if self.behaviour == "last-layer":
                emb = self.model.extract_features(input_tmp)[0][
                    -1
                ]  # getting features from the last layer of transformer
            elif self.behaviour == "weighted-sum":
                all_layers_out = self.model.extract_features(input_tmp)[0][1:]
                all_layers_out = torch.stack(all_layers_out)
                emb = (all_layers_out * self.sum_weights).sum(dim=0)
                return emb
        return emb
