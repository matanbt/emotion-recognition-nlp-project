import os
import random
import logging
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


def compute_metrics_classification(label_targets, label_preds):
    assert len(label_preds) == len(label_targets)
    results = dict()

    results["accuracy"] = accuracy_score(label_targets, label_preds)
    results["macro_precision"], results["macro_recall"], results[
        "macro_f1"], _ = precision_recall_fscore_support(
        label_targets, label_preds, average="macro")
    results["micro_precision"], results["micro_recall"], results[
        "micro_f1"], _ = precision_recall_fscore_support(
        label_targets, label_preds, average="micro")
    results["weighted_precision"], results["weighted_recall"], results[
        "weighted_f1"], _ = precision_recall_fscore_support(
        label_targets, label_preds, average="weighted")

    return results


def compute_metrics_regression_vad(vad_targets, vad_preds):
    assert len(vad_preds) == len(vad_targets)
    results = dict()

    results["R_squared_score"] = r2_score(vad_targets, vad_preds)
    results["mean_squared_error"] = mean_squared_error(vad_targets, vad_preds)

    for i, vad_letter in enumerate("vad"):
        results[f"R_squared_score_{vad_letter}"] = r2_score(vad_targets[:, i], vad_preds[:, i])
        results[f"mean_squared_error_{vad_letter}"] = mean_squared_error(vad_targets[:, i], vad_preds[:, i])

    # Add the classification metrics by basic mapping back to labels
    label_targets, label_preds = compute_labels_from_regression(vad_targets, vad_preds)
    results.update(compute_metrics_classification(label_targets, label_preds))

    return results


def compute_labels_from_regression(vad_targets, vad_preds):
    """
        maps the predicted, target VADs to emotions and calculates the labels prediction
        returns label_targets, label_preds
    """
    label_targets = nearest_point_search(vad_targets, NRC_IDX_TO_VAD)
    label_preds = nearest_point_search(vad_preds, NRC_IDX_TO_VAD)
    return label_targets, label_preds

def nearest_point_search(points, possible_points):
    """
    returns a np.ndarray where cell i = the index of the point from possible_points that is closest to points[i]
    """
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(possible_points)
    return np.array(neigh.kneighbors(points, return_distance=False))[:, 0]


SIGMOID_FUNC = lambda x: 1 / (1 + np.exp(-x.detach().cpu().numpy()))
IDENTITY_FUNC = lambda x: x

NRC_VAD_TO_IDX = {(0.969, 0.583, 0.726): 0,
                  (0.929, 0.837, 0.803): 1,
                  (0.167, 0.865, 0.657): 2,
                  (0.167, 0.718, 0.342): 3,
                  (0.854, 0.46, 0.889): 4,
                  (0.635, 0.469, 0.5): 5,
                  (0.255, 0.667, 0.277): 6,
                  (0.75, 0.755, 0.463): 7,
                  (0.896, 0.692, 0.647): 8,
                  (0.115, 0.49, 0.336): 9,
                  (0.085, 0.551, 0.367): 10,
                  (0.052, 0.775, 0.317): 11,
                  (0.143, 0.685, 0.226): 12,
                  (0.896, 0.684, 0.731): 13,
                  (0.073, 0.84, 0.293): 14,
                  (0.885, 0.441, 0.61): 15,
                  (0.07, 0.64, 0.474): 16,
                  (0.98, 0.824, 0.794): 17,
                  (1.0, 0.519, 0.673): 18,
                  (0.163, 0.915, 0.241): 19,
                  (0.949, 0.565, 0.814): 22,
                  (0.729, 0.634, 0.848): 21,
                  (0.554, 0.51, 0.836): 22,
                  (0.844, 0.278, 0.481): 23,
                  (0.103, 0.673, 0.377): 24,
                  (0.052, 0.288, 0.164): 25,
                  (0.875, 0.875, 0.562): 26,
                  (0.469, 0.184, 0.357): 27}

NRC_IDX_TO_VAD = [(0.969, 0.583, 0.726),
                  (0.929, 0.837, 0.803),
                  (0.167, 0.865, 0.657),
                  (0.167, 0.718, 0.342),
                  (0.854, 0.46, 0.889),
                  (0.635, 0.469, 0.5),
                  (0.255, 0.667, 0.277),
                  (0.75, 0.755, 0.463),
                  (0.896, 0.692, 0.647),
                  (0.115, 0.49, 0.336),
                  (0.085, 0.551, 0.367),
                  (0.052, 0.775, 0.317),
                  (0.143, 0.685, 0.226),
                  (0.896, 0.684, 0.731),
                  (0.073, 0.84, 0.293),
                  (0.885, 0.441, 0.61),
                  (0.07, 0.64, 0.474),
                  (0.98, 0.824, 0.794),
                  (1.0, 0.519, 0.673),
                  (0.163, 0.915, 0.241),
                  (0.949, 0.565, 0.814),
                  (0.729, 0.634, 0.848),
                  (0.554, 0.51, 0.836),
                  (0.844, 0.278, 0.481),
                  (0.103, 0.673, 0.377),
                  (0.052, 0.288, 0.164),
                  (0.875, 0.875, 0.562),
                  (0.469, 0.184, 0.357)]