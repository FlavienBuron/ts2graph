from torch.nn.functional import l1_loss

# from downstream.imputation.metrics.core.base import mae
from downstream.imputation.metrics.core.masked_loss import MaskedLoss


class MaskedMAELoss(MaskedLoss):
    def _elementwise(self, prediction, target):
        return l1_loss(prediction, target, reduction="none")
