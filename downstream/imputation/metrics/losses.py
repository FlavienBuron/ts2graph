from downstream.imputation.metrics.core.base import mae
from downstream.imputation.metrics.core.masked_loss import MaskedLoss


class MaskedMAELoss(MaskedLoss):
    def _elementwise(self, prediction, target):
        return mae(prediction, target, reduction="none")
