from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .mesh_ops import normalize_minmax_np, normalize_sum_np


def _pearson_corr(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_std = float(first.std())
    second_std = float(second.std())
    if first_std == 0.0 or second_std == 0.0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _spearman_corr(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = pd.Series(first).rank(method="average").to_numpy(dtype=np.float64)
    second_rank = pd.Series(second).rank(method="average").to_numpy(dtype=np.float64)
    return _pearson_corr(first_rank, second_rank)


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    numerator = float(np.dot(first, second))
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _nss(saliency_map: np.ndarray, fixation_mask: np.ndarray) -> float:
    saliency_map = np.asarray(saliency_map, dtype=np.float64)
    fixation_mask = np.asarray(fixation_mask, dtype=bool)
    if fixation_mask.sum() == 0:
        return 0.0
    std = float(saliency_map.std())
    if std == 0.0:
        return 0.0
    z_map = (saliency_map - saliency_map.mean()) / std
    return float(z_map[fixation_mask].mean())


def _auc_judd(saliency_map: np.ndarray, fixation_mask: np.ndarray) -> float:
    saliency_map = normalize_minmax_np(saliency_map).reshape(-1)
    fixation_mask = np.asarray(fixation_mask, dtype=bool).reshape(-1)
    fixation_count = int(fixation_mask.sum())
    non_fixation_count = int((~fixation_mask).sum())
    if fixation_count == 0 or non_fixation_count == 0:
        return 0.5

    thresholds = np.sort(np.unique(saliency_map[fixation_mask]))[::-1]
    true_positive = [0.0]
    false_positive = [0.0]

    for threshold in thresholds:
        above = saliency_map >= threshold
        tp = float(np.logical_and(above, fixation_mask).sum()) / fixation_count
        fp = float(np.logical_and(above, ~fixation_mask).sum()) / non_fixation_count
        true_positive.append(tp)
        false_positive.append(fp)

    true_positive.append(1.0)
    false_positive.append(1.0)
    return float(np.trapezoid(np.asarray(true_positive), np.asarray(false_positive)))


def _top_percent_label(percentile: float) -> str:
    top_percent = 100.0 - percentile
    if math.isclose(top_percent, round(top_percent)):
        return str(int(round(top_percent)))
    return str(top_percent).replace(".", "p")


def compute_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    proxy_fixation_percentiles: tuple[float, ...] = (90.0, 95.0, 99.0),
) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    ground_truth = np.asarray(ground_truth, dtype=np.float64).reshape(-1)

    if prediction.shape != ground_truth.shape:
        raise ValueError(f"Shape mismatch: prediction {prediction.shape}, ground truth {ground_truth.shape}")

    prediction_prob = normalize_sum_np(np.clip(prediction, a_min=0.0, a_max=None))
    ground_truth_prob = normalize_sum_np(np.clip(ground_truth, a_min=0.0, a_max=None))

    prediction_unit = normalize_minmax_np(prediction)
    ground_truth_unit = normalize_minmax_np(ground_truth)

    epsilon = 1e-12
    prediction_prob_safe = prediction_prob + epsilon
    ground_truth_prob_safe = ground_truth_prob + epsilon

    metrics = {
        "CC": _pearson_corr(prediction, ground_truth),
        "SIM": float(np.minimum(prediction_prob, ground_truth_prob).sum()),
        "KLD": float(np.sum(ground_truth_prob_safe * np.log(ground_truth_prob_safe / prediction_prob_safe))),
        "MSE": float(np.mean((prediction_unit - ground_truth_unit) ** 2)),
        "SE_MSE": float(np.mean((prediction_unit - ground_truth_unit) ** 2)),
        "MAE": float(np.mean(np.abs(prediction_unit - ground_truth_unit))),
        "Spearman": _spearman_corr(prediction, ground_truth),
        "Cosine": _cosine_similarity(prediction, ground_truth),
        "PredictionSum": float(prediction.sum()),
        "GroundTruthSum": float(ground_truth.sum()),
    }

    for percentile in proxy_fixation_percentiles:
        threshold = float(np.quantile(ground_truth_unit, percentile / 100.0))
        fixation_mask = ground_truth_unit >= threshold
        label = _top_percent_label(percentile)
        metrics[f"NSS_gt_top_{label}pct_proxy"] = _nss(prediction_unit, fixation_mask)
        metrics[f"AUC_Judd_gt_top_{label}pct_proxy"] = _auc_judd(prediction_unit, fixation_mask)
        metrics[f"GTMaskCount_top_{label}pct_proxy"] = float(fixation_mask.sum())

    return metrics
