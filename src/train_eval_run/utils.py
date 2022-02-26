import math
import os
import random
import logging
import subprocess
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.neighbors import NearestNeighbors

def init_logger(dir_path: str):
    os.makedirs(dir_path)
    path = os.path.join(dir_path, 'run.log')
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        filename=path,
                        filemode='w',
                        level=logging.INFO)


def init_tensorboard_writer(path: str):
    """
        returns a tensorboard-'writer', will save the summary into the relative 'path'
    """

    tb_writer = SummaryWriter(path)
    return tb_writer

def get_curr_time_for_filename():
    now = datetime.now()
    # dd_mm_YY_H_M
    now_string = now.strftime("%d_%m_%Y_%H_%M")
    return now_string

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


def compute_metrics_classification(label_targets, label_preds, name_suffix=''):
    assert len(label_preds) == len(label_targets)
    results = dict()

    results[f"accuracy{name_suffix}"] = accuracy_score(label_targets, label_preds)
    results[f"macro_precision{name_suffix}"], results[f"macro_recall{name_suffix}"], results[
        f"macro_f1{name_suffix}"], _ = precision_recall_fscore_support(
        label_targets, label_preds, average="macro")
    results[f"micro_precision{name_suffix}"], results[f"micro_recall{name_suffix}"], results[
        f"micro_f1{name_suffix}"], _ = precision_recall_fscore_support(
        label_targets, label_preds, average="micro")
    results[f"weighted_precision{name_suffix}"], results[f"weighted_recall{name_suffix}"], results[
        f"weighted_f1{name_suffix}"], _ = precision_recall_fscore_support(
        label_targets, label_preds, average="weighted")

    return results


def compute_metrics_regression_vad(vad_targets, vad_preds, emo_lbl_idx_to_vad):
    """ emo_lbl_idx_to_vad - list of the emotions' vad values (ordered by the emotions' labels order) """
    assert len(vad_preds) == len(vad_targets)
    results = dict()

    results["R_squared_score"] = r2_score(vad_targets, vad_preds)
    results["mean_squared_error"] = mean_squared_error(vad_targets, vad_preds)

    # [COMMENTED-OUT] Too noisy.
    # for i, vad_letter in enumerate("vad"):
    #     results[f"R_squared_score_{vad_letter}"] = r2_score(vad_targets[:, i], vad_preds[:, i])
    #     results[f"mean_squared_error_{vad_letter}"] = mean_squared_error(vad_targets[:, i], vad_preds[:, i])

    # Add the classification metrics by mapping back to labels
    label_targets = compute_labels_from_regression(vad_targets, 'euclidean', emo_lbl_idx_to_vad)

    # metrics - the metrics used for the mapping
    # key - metric name, value - metric function or metric str identifier as described in sklearn.metrics.DistanceMetric
    metrics = {
        'euclidean': 'euclidean',
        'manhattan': 'manhattan',
        # 'chebyshev': 'chebyshev',
        # 'minkowski': 'minkowski',
        # 'wminkowski': 'wminkowski',
        # 'seuclidean': 'seuclidean',
        # 'mahalanobis': 'mahalanobis'
        # TODO - we can add more metrics
        # 'my_euclidean': lambda p1, p2: np.sqrt(np.sum((p1 - p2)**2))  # custom metric example, same as 'euclidean'
    }

    for metric_name, metric in metrics.items():
        label_preds = compute_labels_from_regression(vad_preds, metric, emo_lbl_idx_to_vad)
        results.update(compute_metrics_classification(label_targets, label_preds, f'_{metric_name}'))

    # Try VA metric
    va_preds = vad_preds[:, :2]
    emo_lbl_idx_to_va = np.array(emo_lbl_idx_to_vad)[:, :2]
    for metric_name, metric in metrics.items():
        label_preds = nearest_neighbor(va_preds, emo_lbl_idx_to_va, metric)
        results.update(
            compute_metrics_classification(label_targets, label_preds, f'_va_{metric_name}'))

    return results


def compute_labels_from_regression(vads, metric, emo_lbl_idx_to_vad):
    """
        vads - list of vad points
        emo_lbl_idx_to_vad - list of the emotions' vad values (ordered by the emotions' labels order)
        Maps the VADs to emotions by NearestNeighbors with the specified metric
        metric - metric function or metric str identifier as described in sklearn.metrics.DistanceMetric
        returns labels idx list
    """

    labels = nearest_neighbor(vads, emo_lbl_idx_to_vad, metric)
    return labels

def nearest_neighbor(points, possible_points, metric):
    """
    metric - metric function or metric str identifier as described in sklearn.metrics.DistanceMetric
    returns a np.ndarray where cell i =
    the index of the point from possible_points that is closest (by metric) to points[i]
    """
    neigh = NearestNeighbors(n_neighbors=1, metric=metric)
    neigh.fit(possible_points)
    return np.array(neigh.kneighbors(points, return_distance=False))[:, 0]


SIGMOID_FUNC = lambda x: 1 / (1 + np.exp(-x.detach().cpu().numpy()))
IDENTITY_FUNC = lambda x: x


def min_dist_pair(vector) -> int:
    """
     Returns the minimum non-zero distance between a pair of values in a 1-D array.
    """
    min_dist = math.inf
    for i in range(len(vector)):
        for j in range(i + 1, len(vector)):
            if (vector[i] - vector[j]) != 0:
                min_dist = min(min_dist, abs(vector[i] - vector[j]))

    return min_dist