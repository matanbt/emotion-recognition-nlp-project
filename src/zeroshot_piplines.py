import torch

import os

import json
import numpy as np

from attrdict import AttrDict
from catboost.datasets import imdb
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import SequentialSampler, DataLoader

from transformers import (
    BertConfig, BertTokenizer,
)

from datasets import load_dataset
import datasets
from xgboost import XGBClassifier

from src.models.model_regression import BertForMultiDimensionRegression
from src.model_args import model_choices

import pandas as pd
import plotly.express as px


from tqdm import tqdm


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


from src.train_eval_run import utils
from src.train_eval_run.evaluate import only_eval
from src.train_eval_run.utils import compute_metrics_classification

# @Shir Change here paths to suit you here
PATH_CACHED_MODEL_DIR = "./paper-stuff/cahced-models/MAE_SCALED_checkpoint-20000"
PATH_OF_SAVED_MODEL = f"{PATH_CACHED_MODEL_DIR}/pytorch_model.bin"
CONFIG_NAME = "exp_clfs/mae_5_vad_scaled.json"



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
    # processor.perform_full_preprocess()

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


def pipline_imdb():
    model_dict = load_model()
    model = model_dict['model']
    data_processor = model_dict['data_processor']
    emo_lbl_idx_to_vad = data_processor.get_emotions_vads_lst()

    # load imb
    dataset = load_dataset("imdb")
    eval_imdb = dataset['test'] #.shuffle()  #.select(list(range(dataset_size)))  # can shuffle and select
    eval_imdb = data_processor.process_new_dataset(eval_imdb, 200)  # TODO change max_len
    eval_imdb.set_format(type='torch',
                         columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
    dl_eval_imdb = torch.utils.data.DataLoader(eval_imdb, batch_size=1)

    # Run regression model
    predicted_vads = torch.zeros((len(eval_imdb), 3))

    model.eval()
    with torch.no_grad():
        dl_eval_imdb = tqdm(dl_eval_imdb, desc="Evalutation on IMDB")
        for i, example in enumerate(dl_eval_imdb):
            outputs = model(**example)
            predicted_vads[i] = outputs[0]
            # print(f">> VAD = {predicted_vads[i]} || [real] Label = {example['label']}")

    # Save IMDB vads (test)
    np.savetxt(os.path.join(PATH_CACHED_MODEL_DIR, "imdb_test_predicted_vads.csv"), predicted_vads)

    # Calculate some metrics
    thresholds = np.arange(0, 1, 0.05)
    accuracy_by_threshold = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        acc = ((predicted_vads[:, 0] >= threshold).to(int) == eval_imdb['label']).sum() / len(eval_imdb)
        print(f">> threshold = {threshold} || accuracy={acc}")
        accuracy_by_threshold[i] = acc

    px.bar(thresholds, accuracy_by_threshold).show()  # 0.69 basic


def logistic_reg_imdb():
    pred_vads_train = np.loadtxt(os.path.join(PATH_CACHED_MODEL_DIR, "imdb_train_predicted_vads.csv"))
    pred_vads_test = np.loadtxt(os.path.join(PATH_CACHED_MODEL_DIR, "imdb_test_predicted_vads.csv"))

    dataset = load_dataset("imdb")

    log_regr = LogisticRegression() # penalty ='elasticnet'
    log_regr = log_regr.fit(pred_vads_train, dataset['train']['label'])

    print(f">> Accuracy on training: {log_regr.score(pred_vads_train, dataset['train']['label'])}")
    print(f">> Accuracy on test: {log_regr.score(pred_vads_test, dataset['test']['label'])}")


def pipline_emoint():
    model_dict = load_model()
    model = model_dict['model']
    data_processor = model_dict['data_processor']
    emo_lbl_idx_to_vad = data_processor.get_emotions_vads_lst()

    # Load DS
    dataset = datasets.load_dataset('csv', data_files='data/emoint/emoint.tsv',
                                    sep='\t', header=None)
    dataset['train'] = dataset['train'].rename_columns({'0': "id", '1':"text", '2':'label', '3':'volume'})
    eval_emoint = dataset['train']  # .shuffle()

    # Preprocess helpers
    label2id = {
        'anger': 0,
        'fear': 1,
        'joy': 2,
        'sadness': 3,
    }
    def _hf_label_mapper(example):
        example['label'] = label2id[example['label']]
        return example

    # Preprocess dataset
    eval_emoint = eval_emoint.map(_hf_label_mapper)
    eval_emoint = data_processor.process_new_dataset(eval_emoint, 50)
    eval_emoint.set_format(type='torch',
                           columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
    dl_eval_emotint = torch.utils.data.DataLoader(eval_emoint, batch_size=1)

    # Predict VAD with our model
    predicted_vads = torch.zeros((len(eval_emoint), 3))

    model.eval()
    with torch.no_grad():
        dl_eval_emotint = tqdm(dl_eval_emotint, desc="Evalutation on EmoInt")
        for i, example in enumerate(dl_eval_emotint):
            outputs = model(**example)
            predicted_vads[i] = outputs[0]

            # print(f">> VAD = {predicted_vads[i]} || [real] Label = {example['label']}")

    # Save IMDB vads (test)
    np.savetxt(os.path.join(PATH_CACHED_MODEL_DIR, "emoint_train_predicted_vads.csv"), predicted_vads)
    np.savetxt(os.path.join(PATH_CACHED_MODEL_DIR, "emoint_train_labels.csv"), np.array(eval_emoint['label']))
    print("")


def svm_emoint():
    id2label = {
        'anger': 0,
        'fear': 1,
        'joy': 2,
        'sadness': 3,
    }
    predicted_vads = np.loadtxt(os.path.join(PATH_CACHED_MODEL_DIR, "emoint_train_predicted_vads.csv"))
    labels = np.loadtxt(os.path.join(PATH_CACHED_MODEL_DIR, "emoint_train_labels.csv"))
    labels = np.expand_dims(labels, axis=-1).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(predicted_vads, labels, test_size=0.2)

    model = svm.SVC()
    model.fit(X_train, y_train)

    print(f">> [SVM] Accuracy on training: {model.score(X_train, y_train)}")
    print(f">> [SVM] Accuracy on test: {model.score(X_test, y_test)}")

    model = XGBClassifier(num_class=4)
    model.fit(X_train, y_train)

    print(f">> [XGBoost] Accuracy on training: {model.score(X_train, y_train)}")
    print(f">> [XGBoost] Accuracy on test: {model.score(X_test, y_test)}")


def pipline_isaer():
    dataset = datasets.load_dataset('csv', data_files='data/isear/isear.csv')

    print("")


def emotion_pipeline():
    model_dict = load_model()
    model = model_dict['model']
    data_processor = model_dict['data_processor']
    emo_lbl_idx_to_vad = data_processor.get_emotions_vads_lst()

    # load imb
    train_size = 3000
    dataset = load_dataset("emotion")
    train_dataset = dataset['train'] # .select(list(range(train_size)))  #.shuffle()
    test_dataset = dataset['test']
    train_dataset = data_processor.process_new_dataset(train_dataset, 100)
    test_dataset = data_processor.process_new_dataset(test_dataset, 100)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
    dl_train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    dl_test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    # Run regression model
    train_predicted_vads = torch.zeros((len(train_dataset), 3))
    test_predicted_vads = torch.zeros((len(test_dataset), 3))

    model.eval()
    with torch.no_grad():
        dl_train_dataset = tqdm(dl_train_dataset, desc="Training on Emotion")
        for i, example in enumerate(dl_train_dataset):
            outputs = model(**example)
            train_predicted_vads[i] = outputs[0]

        dl_test_dataset = tqdm(dl_test_dataset, desc="Testing on Emotion")
        for i, example in enumerate(dl_test_dataset):
            outputs = model(**example)
            test_predicted_vads[i] = outputs[0]

    # Save Emotion vads
    np.savetxt(os.path.join(PATH_CACHED_MODEL_DIR, "emotion_train_predicted_vads.csv"), train_predicted_vads)
    np.savetxt(os.path.join(PATH_CACHED_MODEL_DIR, "emotion_test_predicted_vads.csv"), test_predicted_vads)

    # svm from VAD
    # TODO make sure the following line is OK type wise for mode.fit
    X_train, X_test, y_train, y_test = train_predicted_vads, test_predicted_vads, train_dataset['label'], test_dataset['label']

    model = svm.SVC()
    model.fit(X_train, y_train)

    print(f">> [SVM] Accuracy on training: {model.score(X_train, y_train)}")
    print(f">> [SVM] Accuracy on test: {model.score(X_test, y_test)}")

    model = XGBClassifier(num_class=4)
    model.fit(X_train, y_train)


if __name__ == '__main__':
    print("---- Testing IMDB with valence threshold -----")
    pipline_imdb()
    print("---- Testing Emotion with SVM -----")
    emotion_pipeline() #57%
