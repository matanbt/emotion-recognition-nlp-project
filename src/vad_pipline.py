import torch

import os

import json
import numpy as np

from attrdict import AttrDict
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from torch.utils.data import SequentialSampler, DataLoader

from transformers import (
    BertConfig, BertTokenizer,
)

from src.models.model_regression import BertForMultiDimensionRegression
from src.model_args import model_choices

import pandas as pd

from tqdm import tqdm


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

import plotly.express as px


from src.train_eval_run import utils
from src.train_eval_run.evaluate import only_eval
from src.train_eval_run.utils import compute_metrics_classification


PATH_CACHED_MODEL_DIR = "./paper-stuff/cahced-models/MAE_SCALED_checkpoint-20000"
PATH_OF_SAVED_MODEL = f"{PATH_CACHED_MODEL_DIR}/pytorch_model.bin"
TRAIN_VAD_CSV = f"{PATH_CACHED_MODEL_DIR}/trained_vad.csv"
CONFIG_NAME = "exp_clfs/mae_5_vad_scaled.json"
#
# PATH_OF_SAVED_MODEL = "./MAE_MEAN_checkpoint-22700/pytorch_model.bin"
# CONFIG_NAME = "exp3/new_baseline.json"

# BOILER PLATE FOR LOADING -------------------------------------------------------------
def load_model():
    config_path = os.path.join("config", CONFIG_NAME)
    with open(config_path) as f:
        args = AttrDict(json.load(f))

    model_args = model_choices.get(args.model_args)

    label_list = model_args.data_processor_class.get_labels_list(args)

    # Initiate all needed model's component
    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        finetuning_task=args.task,
        id2label={str(i): label for i, label in enumerate(label_list)},
        label2id={label: i for i, label in enumerate(label_list)}
    )

    tokenizer = BertTokenizer.from_pretrained(  # TODO - why do we use a custom tokenizer??
        args.tokenizer_name_or_path,
    )

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    # model.to(args.device)

    # Process Data
    build_dataset = False
    processor = model_args.data_processor_class(args, tokenizer, args.max_seq_len,
                                                vad_mapper_name=model_args.vad_mapper_name,
                                                build_dataset=build_dataset)

    train_dataset = None
    dev_dataset = None
    test_dataset = None

    if build_dataset:
        processor.perform_full_preprocess()

        # Load dataset
        train_dataset = processor.get_dataset_by_mode('train')
        dev_dataset = processor.get_dataset_by_mode('dev')
        test_dataset = processor.get_dataset_by_mode('test')

    model = BertForMultiDimensionRegression(config=config,
                                            **vars(model_args))

    model.load_state_dict(torch.load(PATH_OF_SAVED_MODEL, map_location=torch.device(args.device)))
    model.to(args.device)

    return {
        'model': model,
        'model_args': model_args,
        'data_processor': processor,
        'train_dataset': train_dataset,
        'dev_dataset': dev_dataset,
    }
# END OF BOILER PLATE FOR LOADING -------------------------------------------------------------


def pipline_example(model_dict, text):
    model = model_dict['model']
    data_processor = model_dict['data_processor']
    emo_lbl_idx_to_vad = data_processor.get_emotions_vads_lst()
    # Process example
    example = {"text": text}
    example = data_processor.process_example(example)

    # Run regression model
    model.eval()
    with torch.no_grad():
        outputs = model(**example)
        # loss, vad_pred = outputs[:2]
        vad_pred = outputs[0].numpy()
    print(f"Regression: loss={None}, predicted-VAD={vad_pred}")

    # Run classifiers:

    # 1NN
    metric = 'manhattan'  # manhattan # TODO try: 'euclidean'
    neigh = NearestNeighbors(n_neighbors=5, metric=metric)
    neigh.fit(emo_lbl_idx_to_vad)
    distances, labels = neigh.kneighbors(vad_pred, return_distance=True)
    distances = 1 - (distances / 2 ** 0.5)  # scaler to probability
    # (1 / distances) / (1 / distances).sum() # another scaler
    labels = [model.config.id2label[str(label)] for label in labels[0]]
    print(f"`{text}` - TOP 5: \n  "
          f">> vad_pred: {vad_pred} \n  "
          f">> labels - {labels} \n  "
          f">> probs: {distances}  ")
    
    # SVM
    svm_model = SVC(C=10, gamma=3.5, probability=True)

    # load training model results
    arr = np.loadtxt(TRAIN_VAD_CSV)
    train_vads, train_labels = arr[:, 1:], arr[:, 0].astype(int)
    svm_model.fit(train_vads, train_labels)

    print(f"  >> SVM predict: {model.config.id2label[str(svm_model.predict(vad_pred).item())]}")
    probs = svm_model.predict_proba(vad_pred)[0]
    idx_probs = probs.argsort()[::-1]
    probs = probs[idx_probs]  # sorted probs
    svm_rating = [model.config.id2label[str(idx)] for idx in idx_probs]
    print(f"SVM Rating: \n  "
          f">> svm_rating: {svm_rating} \n  "
          f">> probs - {probs}")


if __name__ == '__main__':
    model_dict = load_model()
    example_something = "Something in the way she moves, attracts me like no other lover."
    example_cant_wait = "I can't wait to see you again!"
    pipline_example(model_dict, example_something)