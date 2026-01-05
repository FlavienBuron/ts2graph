from time import perf_counter

import torch
from einops import rearrange
from torch import nn

from downstream.imputation.models.GRIN.layers.gril import BiGRIL

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
        self.register_buffer("adj", adj.detach().clone().float())
        self.impute_only_holes = impute_only_holes
        # print(f"DEBUG: {self.impute_only_holes=}")

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

    def forward(
        self, x, mask=None, u=None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        dump_parameters(self)
        # print(f"    {x.min()=} {x.max()=} {x.mean()=} {x.sum()=} {x.std()=}")
        total_imputation_time = 0.0
        # x = x.unsqueeze(0)
        # x: [batches, steps, nodes, channels] -> [batches, channels, nodes, steps]
        # print(f"DEBUG GRIN 1.: {x.min()=} {x.max()=} {x.mean()=} {x.sum()=} {x.std()=}")
        x = rearrange(x, "b s n c -> b c n s")
        # print(f"DEBUG GRIN 2.: {x.min()=} {x.max()=} {x.mean()=} {x.sum()=} {x.std()=}")
        if mask is not None:
            # mask = mask.unsqueeze(0)
            mask = rearrange(mask, "b s n c -> b c n s")
        else:
            mask = torch.zeros_like(x)

        if u is not None:
            u = rearrange(u, "b s n c -> b c n s")

        # imputation: [batches, channels, nodes, steps] prediction: [4, batches, channels, nodes, steps]
        imputation_start = perf_counter()
        imputation, prediction = self.bigrill(
            x, self.adj, mask=mask, u=u, cached_support=self.training
        )
        print(
            f"DEBUG GRIN 1.: {imputation.min()=} {imputation.max()=} {imputation.mean()=} {imputation.sum()=} {imputation.std()=}"
        )
        imputation_end = perf_counter()
        total_imputation_time = imputation_end - imputation_start
        # In evaluation stage impute only missing values
        if self.impute_only_holes and not self.training:
            # print(f"DEBUG: forward {mask.float().mean()=} {mask.float().sum()=}")
            # print(
            #     f"DEBUG forward: {imputation.min()=} {imputation.max()=} {imputation.mean()=} {imputation.std()=} {imputation.sum()=}"
            # )
            # masked_imp = torch.where(mask, torch.tensor(float("nan")), imputation)
            imputation = torch.where(mask, x, imputation)
            # masked_x = torch.where(mask, x, torch.tensor(float("nan")))
            # print("Masked imputation:", masked_imp[0, :15, :15, 0])
            # print("Masked target:", masked_x[0, :15, :15, 0])
            # print("Actual imputation:", imputation[0, :15, :15, 0])
            # print(
            #     f"DEBUG forward: {imputation.min()=} {imputation.max()=} {imputation.mean()=} {imputation.std()=} {imputation.sum()=}"
            # )
        # out: [batches, channels, nodes, steps] -> [batches, steps, nodes, channels]
        imputation = torch.transpose(imputation, -3, -1)
        prediction = torch.transpose(prediction, -3, -1)

        # imputation = imputation.squeeze(0)

        return imputation, prediction, total_imputation_time

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


def dump_parameters(model):
    print("\n=== PARAMETERS ===")
    for name, p in model.named_parameters():
        if p is None:
            print(f"{name}: None")
        else:
            print(
                f"{name}: "
                f"shape={tuple(p.shape)} "
                f"mean={p.mean().item():.4g} "
                f"std={p.std().item():.4g} "
                f"max|x|={p.abs().max().item():.4g}"
            )
