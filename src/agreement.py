"""
Agreement between a model mask and a manual mask: Dice, HD95 and MASD.

The values follow MONAI's DiceMetric, HausdorffDistanceMetric(percentile=95)
and SurfaceDistanceMetric with their default settings, written with numpy and
scipy so the app needs neither torch nor MONAI:

- HD95 is the larger of the two one-way 95th percentiles of the distances
  between the mask edges (model -> manual and manual -> model).
- MASD is one-way: the mean distance from each edge pixel of the model mask to
  the nearest edge pixel of the manual mask.

A mask's edge is the mask with its erosion taken away.
"""

import math

import numpy as np
from scipy import ndimage


def _edges(mask):
    return mask ^ ndimage.binary_erosion(mask)


def _edge_distances(from_edges, to_edges, sampling):
    """Distance from each pixel of from_edges to the nearest pixel of to_edges."""
    return ndimage.distance_transform_edt(~to_edges, sampling=sampling)[from_edges]


def _distances(pred, manual, sampling):
    pred_edges = _edges(pred)
    manual_edges = _edges(manual)
    pred_to_manual = _edge_distances(pred_edges, manual_edges, sampling)
    manual_to_pred = _edge_distances(manual_edges, pred_edges, sampling)
    hd95 = max(
        float(np.percentile(pred_to_manual, 95)),
        float(np.percentile(manual_to_pred, 95)),
    )
    return hd95, float(pred_to_manual.mean())


def agreement(pred, manual, px_per_mm=None):
    """Dice, HD95 and MASD between a model mask and a manual mask of one image.

    px_per_mm is (pixels per mm across, pixels per mm down), or None. The
    distances in mm are given only when both are set; otherwise they are nan.
    Every value is nan when either mask is empty, as MONAI gives no value then.
    """
    pred = np.asarray(pred).astype(bool)
    manual = np.asarray(manual).astype(bool)
    nan = float("nan")
    result = {"dice": nan, "hd95_px": nan, "masd_px": nan, "hd95_mm": nan, "masd_mm": nan}
    if not pred.any() or not manual.any():
        return result
    overlap = np.logical_and(pred, manual).sum()
    result["dice"] = float(2.0 * overlap / (pred.sum() + manual.sum()))
    result["hd95_px"], result["masd_px"] = _distances(pred, manual, None)
    if px_per_mm is not None:
        across, down = px_per_mm
        if across and down:
            result["hd95_mm"], result["masd_mm"] = _distances(
                pred, manual, (1.0 / down, 1.0 / across)
            )
    return result


def is_number(value):
    return value is not None and not math.isnan(value)
