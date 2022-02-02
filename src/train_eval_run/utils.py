import os
import random
import logging

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


def compute_metrics_regression(vads, preds):
    # TODO
    assert len(preds) == len(vads)
    results = dict()

    results["R_squared_score"] = r2_score(vads, preds)
    results["mean_squared_error"] = mean_squared_error(vads, preds)

    return results


