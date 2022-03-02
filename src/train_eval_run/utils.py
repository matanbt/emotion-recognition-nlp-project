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

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


logger = logging.getLogger(__name__)

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


def compute_metrics_regression_vad(vad_targets, vad_preds, emo_lbl_idx_to_vad,
                                   labels_targets, args):
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

    # calculate our classic variation of KNN with different metrics:
    for metric_name, metric in metrics.items():
        label_preds = compute_labels_from_regression(vad_preds, metric, emo_lbl_idx_to_vad)
        results.update(compute_metrics_classification(label_targets, label_preds, f'_{metric_name}'))

    # train classifier on training_vads --> use it to map dev-set predictions to labels
    results.update(special_classifiers_metrics(vad_preds, labels_targets, args))

    # Try VA metric [commented out]
    # va_preds = vad_preds[:, :2]
    # emo_lbl_idx_to_va = np.array(emo_lbl_idx_to_vad)[:, :2]
    # for metric_name, metric in metrics.items():
    #     label_preds = nearest_neighbor(va_preds, emo_lbl_idx_to_va, metric)
    #     results.update(
    #         compute_metrics_classification(label_targets, label_preds, f'_va_{metric_name}'))

    return results

def special_classifiers_metrics(eval_preds, eval_labels, args):
    results = {}

    # load training model results
    arr = np.loadtxt(os.path.join(args.summary_path, "trained_vad.csv"))
    train_vads, train_labels = arr[:, 1:], arr[:, 0].astype(int)

    clfs_dict = {
        'svm': SVC(kernel="sigmoid"),
        '1NN': KNeighborsClassifier(1),
        'tree': DecisionTreeClassifier(criterion='entropy', random_state=0),
        'forest': RandomForestClassifier(max_depth=2, random_state=0),
        'ADABoost': AdaBoostClassifier(),
        'MLP': MLPClassifier(alpha=1, max_iter=1000),
        'XGBoost': XGBClassifier(use_label_encoder=False)
    }

    for clf_name, clf in clfs_dict.items():
        logger.info(f"Fitting classifier: {clf}")
        clf.fit(train_vads, train_labels)
        logger.info(f"Classifier: {clf.__class__.__name__} | "
                    f"Training Accuracy: {clf.score(train_vads, train_labels)}")
        logger.info(f"Classifier: {clf.__class__.__name__} | "
                    f"Dev Accuracy: {clf.score(eval_preds, eval_labels)}")

        eval_labels_preds = clf.predict(eval_preds)

        results.update(compute_metrics_classification(eval_labels, eval_labels_preds,
                                                      name_suffix=f"_{clf.__class__.__name__}"))

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


