from time import perf_counter

import torch
from einops import rearrange
from torch import nn

from downstream.imputation.GRIN.layers.gril import BiGRIL

# from ...utils.parser_utils import str_to_bool


class GRINet(nn.Module):
    def __init__(
        self,
        adj,
        d_in,
        d_hidden,
        d_ff,
        ff_dropout,
        n_layers=1,
        kernel_size=2,
        decoder_order=1,
        global_att=False,
        d_u=0,
        d_emb=0,
        layer_norm=False,
        merge="mlp",
        impute_only_holes=True,
    ):
        super(GRINet, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_u = int(d_u) if d_u is not None else 0
        self.d_emb = int(d_emb) if d_emb is not None else 0
        self.register_buffer("adj", torch.tensor(adj).float())
        self.impute_only_holes = impute_only_holes

        self.bigrill = BiGRIL(
            input_size=self.d_in,
            ff_size=d_ff,
            ff_dropout=ff_dropout,
            hidden_size=self.d_hidden,
            embedding_size=self.d_emb,
            n_nodes=self.adj.shape[0],
            n_layers=n_layers,
            kernel_size=kernel_size,
            decoder_order=decoder_order,
            global_att=global_att,
            u_size=self.d_u,
            layer_norm=layer_norm,
            merge=merge,
        )

    def forward(self, x, mask=None, u=None, **kwargs):
        total_imputation_time = 0.0
        x = x.unsqueeze(0)
        # x: [batches, steps, nodes, channels] -> [batches, channels, nodes, steps]
        x = rearrange(x, "b s n c -> b c n s")
        if mask is not None:
            mask = mask.unsqueeze(0)
            mask = rearrange(mask, "b s n c -> b c n s")

        if u is not None:
            u = rearrange(u, "b s n c -> b c n s")

        # imputation: [batches, channels, nodes, steps] prediction: [4, batches, channels, nodes, steps]
        imputation_start = perf_counter()
        imputation, prediction = self.bigrill(
            x, self.adj, mask=mask, u=u, cached_support=self.training
        )
        imputation_end = perf_counter()
        total_imputation_time = imputation_end - imputation_start
        # In evaluation stage impute only missing values
        if self.impute_only_holes and not self.training:
            imputation = torch.where(mask, x, imputation)
        # out: [batches, channels, nodes, steps] -> [batches, steps, nodes, channels]
        imputation = torch.transpose(imputation, -3, -1)
        prediction = torch.transpose(prediction, -3, -1)

        imputation = imputation.squeeze(0)

        return imputation, total_imputation_time

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--d-hidden", type=int, default=64)
        parser.add_argument("--d-ff", type=int, default=64)
        parser.add_argument("--ff-dropout", type=int, default=0.0)
        parser.add_argument("--n-layers", type=int, default=1)
        parser.add_argument("--kernel-size", type=int, default=2)
        parser.add_argument("--decoder-order", type=int, default=1)
        parser.add_argument("--d-u", type=int, default=0)
        parser.add_argument("--d-emb", type=int, default=8)
        parser.add_argument(
            "--layer-norm", type=bool, nargs="?", const=True, default=False
        )
        parser.add_argument(
            "--global-att", type=bool, nargs="?", const=True, default=False
        )
        parser.add_argument("--merge", type=str, default="mlp")
        parser.add_argument(
            "--impute-only-holes", type=bool, nargs="?", const=True, default=True
        )
        return parser
