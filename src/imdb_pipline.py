import torch

import os

import json
import numpy as np

from attrdict import AttrDict
from catboost.datasets import imdb
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from torch.utils.data import SequentialSampler, DataLoader

from transformers import (
    BertConfig, BertTokenizer,
)

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

PATH_CACHED_MODEL_DIR = "./paper-stuff/cahced-models/MAE_SCALED_checkpoint-20000"
PATH_OF_SAVED_MODEL = f"{PATH_CACHED_MODEL_DIR}/pytorch_model.bin"
CONFIG_NAME = "exp_clfs/mae_5_vad_scaled.json"

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
    dataset_size = 5000 # how much to select from the dataset
    from datasets import load_dataset
    dataset = load_dataset("imdb")
    eval_imdb = dataset['test']  #.shuffle().select(list(range(dataset_size)))  # can shuffle and select
    eval_imdb = data_processor.process_new_dataset(dataset_size, 200)  # TODO change max_len
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

    px.bar(thresholds, accuracy_by_threshold).show()

if __name__ == '__main__':
    pipline_imdb()