# %%
# ------------ loading model for evaluation ------------
# TODO (not mandatory) - maybe more easy - 5. Save and load entire model -
#  https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html
import torch

import os

import json
import numpy as np

from attrdict import AttrDict
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from torch.utils.data import SequentialSampler, DataLoader

from transformers import (
    BertConfig, BertTokenizer,
)

from src.models.model_regression import BertForMultiDimensionRegression
from src.model_args import model_choices

import pandas as pd

from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


from src.train_eval_run.utils import compute_metrics_classification


PATH_OF_SAVED_MODEL = "./MAE_MEAN_checkpoint-22700/pytorch_model.bin"
CONFIG_NAME = "exp3/new_baseline.json"

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
    processor = model_args.data_processor_class(args, tokenizer, args.max_seq_len,
                                                vad_mapper_name=model_args.vad_mapper_name)
    processor.perform_full_preprocess()

    # Load dataset
    train_dataset = processor.get_dataset_by_mode('train')
    dev_dataset = processor.get_dataset_by_mode('dev')
    test_dataset = processor.get_dataset_by_mode('test')

    # %%

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

# %%
import matplotlib.pyplot as plt

def save_training_to_csv(csv_f_name="data/trained_vad.csv"):
    model_dict = load_model()
    model = model_dict['model']
    train_dataset = model_dict['train_dataset']

    # by forward passes
    eval_sampler = SequentialSampler(train_dataset)
    eval_dataloader = DataLoader(train_dataset, sampler=eval_sampler, batch_size=1)
    arr = np.zeros((len(train_dataset),4))
    i = 0

    for batch in tqdm(eval_dataloader, desc="Saving Forward passes to CSV"):
        model.eval()

        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]

            arr[i] = [train_dataset['labels'][i][0]] + list(logits.squeeze(dim=0))

        i += 1

    np.savetxt(csv_f_name, arr, delimiter=",")

def vad_mapping_to_pandas(fname="data/train_vad_mapping_2.csv"):
    df = pd.read_csv(fname, usecols=['vad','labels'])
    df['vad'] = df['vad'].apply(lambda lst: json.loads(lst))

    return df

def df_to_numpy_arrays(df):
    arr = df.to_numpy()
    vads = np.vstack(arr[:, 1]).astype(float)
    labels = arr[:, 0].astype(int)
    return vads, labels

def train_clf(clf_name, X_train, y_train, X_test, y_test):
    clfs_dict = {
        'svm': SVC(),
        '1NN': KNeighborsClassifier(1),
        'tree': DecisionTreeClassifier(criterion='entropy', random_state=0),
        'forest': RandomForestClassifier(max_depth=2, random_state=0),
        'ADABoost': AdaBoostClassifier(),
        'MLP':  MLPClassifier(alpha=1, max_iter=1000),
        'XGBoost': XGBClassifier()
    }
    clf = clfs_dict[clf_name]
    clf.fit(X_train, y_train)
    print(f"Classifier: {clf_name} | Got score: {clf.score(X_test, y_test)}")

    return clf

def run_clf(NUM_OF_EXAMPLES=2000):
    model_dict = load_model()
    original_vad_mapping = model_dict['data_processor'].get_emotions_vads_lst()
    rev_original_vad_mapping = model_dict['data_processor'].get_vads_to_emotions_dict()

    # load training
    arr = np.loadtxt("data/trained_vad.csv", delimiter=",")
    train_vads, train_labels = arr[:, 1:], arr[:, 0].astype(int)
    train_targets = np.zeros((len(train_labels) ,3))
    for i, label in enumerate(train_labels):
        train_targets[i] = original_vad_mapping[label]

    print(f"overall MAE of training: {(abs(train_targets - train_vads)).mean()}")

    clf_names = [
        'svm',
        '1NN',
        'tree',
        'forest',
        'ADABoost',
        'MLP',
        'XGBoost',
    ]

    # eval_loss, labels, targets, preds = only_eval(model_dict['dev_dataset'],
    #                                               model_dict['model'], model_dict['model_args'],
    #                                               1)
    labels = np.loadtxt("data/eval_labels.csv").astype(int)
    targets = np.loadtxt("data/eval_targets.csv").astype(float)
    preds = np.loadtxt("data/eval_preds.csv").astype(float)
    label_targets = labels

    for clf_name in clf_names:
        clf = train_clf(clf_name, train_vads, train_labels,
                                  preds, label_targets)

        label_preds = clf.predict(preds)

        results = compute_metrics_classification(label_targets, label_preds, name_suffix=f"_{clf_name}")
        print(results)

run_clf()





