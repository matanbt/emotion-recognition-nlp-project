import os
import random
import logging
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import r2_score, mean_squared_error

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
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


def compute_metrics_classification(labels, preds):
    assert len(preds) == len(labels)
    results = dict()

    results["accuracy"] = accuracy_score(labels, preds)
    results["macro_precision"], results["macro_recall"], results[
        "macro_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="macro")
    results["micro_precision"], results["micro_recall"], results[
        "micro_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="micro")
    results["weighted_precision"], results["weighted_recall"], results[
        "weighted_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="weighted")

    return results


def compute_metrics_regression_vad(vads_target, vads_pred):
    assert len(vads_pred) == len(vads_target)
    results = dict()

    results["R_squared_score"] = r2_score(vads_target, vads_pred)
    results["mean_squared_error"] = mean_squared_error(vads_target, vads_pred)

    for i, vad_letter in enumerate("vad"):
        results[f"R_squared_score_{vad_letter}"] = r2_score(vads_target[:, i], vads_pred[:, i])
        results[f"mean_squared_error_{vad_letter}"] = mean_squared_error(vads_target[:, i], vads_pred[:, i])

    return results


def vad_to_emotion_idx(vad):
    vad_to_idx = {(0.969, 0.583, 0.726): 0,
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
    return vad_to_idx[vad]


SIGMOID_FUNC = lambda x: 1 / (1 + np.exp(-x.detach().cpu().numpy()))
IDENTITY_FUNC = lambda x: x
