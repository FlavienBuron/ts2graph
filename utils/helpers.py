import numpy as np
import pandas as pd
import torch
from scipy.signal.windows import gaussian


def aggregate_predictions(predictions):
    targets, imputations, masks = [], [], []

    for pred in predictions:
        if pred["target"] is not None:
            targets.append(pred["target"])
        imputations.append(pred["imputation"])
        if pred["mask"] is not None:
            masks.append(pred["mask"])

    target = torch.cat(targets, dim=0) if targets else None
    imputation = torch.cat(imputations, dim=0)
    mask = torch.cat(masks, dim=0) if masks else None

    return target, imputation, mask


def prediction_dataframe(y, index, columns=None, aggregate_by="mean"):
    """Aggregate batched predictions in a single DataFrame.

    @param (list or np.ndarray) y: the list of predictions.
    @param (list or np.ndarray) index: the list of time indexes coupled with the predictions.
    @param (list or pd.Index) columns: the columns of the returned DataFrame.
    @param (str or list) aggregate_by: how to aggregate the predictions in case there are more than one for a step.
    - `mean`: take the mean of the predictions
    - `central`: take the prediction at the central position, assuming that the predictions are ordered chronologically
    - `smooth_central`: average the predictions weighted by a gaussian signal with std=1
    - `last`: take the last prediction
    @return: pd.DataFrame df: the evaluation mask for the DataFrame
    """
    dfs = [
        pd.DataFrame(data=data.reshape(data.shape[:2]), index=idx, columns=columns)
        for data, idx in zip(y, index)
    ]
    df = pd.concat(dfs)
    preds_by_step = df.groupby(df.index)
    # aggregate according passed methods
    aggr_methods = ensure_list(aggregate_by)
    dfs = []
    for aggr_by in aggr_methods:
        if aggr_by == "mean":
            dfs.append(preds_by_step.mean())
        elif aggr_by == "central":
            dfs.append(preds_by_step.aggregate(lambda x: x[int(len(x) // 2)]))
        elif aggr_by == "smooth_central":
            dfs.append(
                preds_by_step.aggregate(
                    lambda x: np.average(x, weights=gaussian(len(x), 1))
                )
            )
        elif aggr_by == "last":
            dfs.append(
                preds_by_step.aggregate(lambda x: x[0])
            )  # first imputation has missing value in last position
        else:
            raise ValueError(
                "aggregate_by can only be one of %s"
                % ["mean", "centralsmooth_central", "last"]
            )
    if isinstance(aggregate_by, str):
        return dfs[0]
    return dfs


def ensure_list(obj):
    if isinstance(obj, (list, tuple)):
        return list(obj)
    else:
        return [obj]


def debug_mask_relationship(mask, eval_mask, name=""):
    # ensure boolean
    mask = mask.bool()
    eval_mask = eval_mask.bool()

    total = mask.numel()

    overlap = (mask & eval_mask).sum()
    mask_only = (mask & ~eval_mask).sum()
    eval_only = (~mask & eval_mask).sum()
    neither = (~mask & ~eval_mask).sum()

    print(f"\n=== MASK DEBUG {name} ===")
    print(f"total elements         : {total}")
    print(f"mask true              : {mask.sum().item()} ({mask.float().mean():.4f})")
    print(
        f"eval_mask true         : {eval_mask.sum().item()} ({eval_mask.float().mean():.4f})"
    )
    print(f"overlap (mask & eval)  : {overlap.item()}")
    print(f"mask only              : {mask_only.item()}")
    print(f"eval_mask only         : {eval_only.item()}")
    print(f"neither                : {neither.item()}")

    # logical relations
    print("mask == eval_mask      :", torch.equal(mask, eval_mask))
    print("eval ⊆ mask            :", eval_only.item() == 0)
    print("mask ⊆ eval            :", mask_only.item() == 0)
