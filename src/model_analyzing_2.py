# %%
# ------------ loading model for evaluation ------------
# TODO (not mandatory) - maybe more easy - 5. Save and load entire model -
#  https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html
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


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.train_eval_run.evaluate import only_eval
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

# def save_training_to_csv(csv_f_name="data/trained_vad.csv"):
#     model_dict = load_model()
#     model = model_dict['model']
#     train_dataset = model_dict['train_dataset']
#
#     # by forward passes
#     eval_sampler = SequentialSampler(train_dataset)
#     eval_dataloader = DataLoader(train_dataset, sampler=eval_sampler, batch_size=1)
#     arr = np.zeros((len(train_dataset),4))
#     i = 0
#
#     for batch in tqdm(eval_dataloader, desc="Saving Forward passes to CSV"):
#         model.eval()
#
#         with torch.no_grad():
#             outputs = model(**batch)
#             loss, logits = outputs[:2]
#
#             arr[i] = [train_dataset['labels'][i][0]] + list(logits.squeeze(dim=0))
#
#         i += 1
#
#     np.savetxt(csv_f_name, arr)
#
# def vad_mapping_to_pandas(fname="data/train_vad_mapping_2.csv"):
#     df = pd.read_csv(fname, usecols=['vad','labels'])
#     df['vad'] = df['vad'].apply(lambda lst: json.loads(lst))
#
#     return df
#
# def df_to_numpy_arrays(df):
#     arr = df.to_numpy()
#     vads = np.vstack(arr[:, 1]).astype(float)
#     labels = arr[:, 0].astype(int)
#     return vads, labels

# ---------------- Classifiers Helpers ------------------------------
def train_clf(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    print(f"Classifier: {clf.__class__.__name__} | Training score: {clf.score(X_train, y_train)}")
    print(f"Classifier: {clf.__class__.__name__} | Dev score: {clf.score(X_test, y_test)}")

    return clf

def print_metrics(split, label_targets, label_preds):
    acc_score = accuracy_score(label_targets, label_preds)
    _, _, macro_f1_score, _ = precision_recall_fscore_support(label_targets,
                                                              label_preds,
                                                              average="macro")

    print(f"---> Split: {split} || Accuracy: {acc_score} || macro_f1: {macro_f1_score}")

    return accuracy_score, macro_f1_score


# TODO tune C as well
def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (11,) :
             An array that contains the accuracy of the
             resulting model on the VALIDATION set.
    """
    print(" [RBF ACCURACY PER (gamma, c)] ")
    gamma_labels = np.arange(-3, 2, 0.5)
    c_labels = np.arange(-3, 4, 0.5)
    gamma_lst, c_lst = 10 ** gamma_labels, 10 ** c_labels
    df_dev_f1 = pd.DataFrame(0, index=gamma_labels, columns=c_labels, dtype=float)
    df_dev_acc = pd.DataFrame(0, index=gamma_labels, columns=c_labels, dtype=float)
    df_train_f1 = pd.DataFrame(0, index=gamma_labels, columns=c_labels, dtype=float)
    df_train_acc = pd.DataFrame(0, index=gamma_labels, columns=c_labels, dtype=float)

    # tune C & GAMMA together
    for i, gamma in enumerate(gamma_lst):
        for j, c in enumerate(c_lst):
            clf = SVC(kernel='rbf', C=c, gamma=gamma)
            print(f" ----------- Gamma = {gamma} || C = {c} -----------")
            clf.fit(X_train, y_train)
            train_preds = clf.predict(X_train)
            val_preds = clf.predict(X_val)
            df_train_acc.iloc[i, j], df_train_f1.iloc[i, j] = print_metrics("train", train_preds, y_train)
            df_dev_acc.iloc[i, j], df_dev_f1.iloc[i,j] = print_metrics("dev", val_preds, y_val)

    print("DONE checking parameters for SVM")
    return df_train_acc, df_train_f1, df_dev_acc, df_dev_f1
    # show train VS eval
    # plt.clf()
    # plt.ylabel("Score")
    # plt.xlabel("log_10 of Gamma")
    # plt.title("Score as function of Gamma")
    # plt.plot(gamma_labels, scores_train, color='blue', label='Training', marker='o')
    # plt.plot(gamma_labels, scores_val, color='red', label='Validation',  marker='o')
    # plt.legend()
    # plt.show()
    #
    # return scores_val


# ---------------- END Classifiers Helpers ------------------------------



# constants:
EVAL_LABELS_CSV = "eval_labels.csv"
EVAL_TARGETS_CSV = "eval_targets.csv"
EVAL_PREDS_CSV = "eval_preds.csv"
TRAIN_VAD_CSV = "trained_vad.csv"

def run_clf(summary_path, cached_eval=True):
    # ---------------- LOADING MODEL & DATA FROM CACHE ------------------------------


    # load training
    arr = np.loadtxt(os.path.join(summary_path, TRAIN_VAD_CSV))
    train_vads, train_labels = arr[:, 1:], arr[:, 0].astype(int)

    # >> to calculate original VADs we need the VAD mapping that was used, by LOADING THE MODEL
    # model_dict = load_model()
    # original_vad_mapping = model_dict['data_processor'].get_emotions_vads_lst()
    # train_targets = np.zeros((len(train_labels) ,3))  # original VADs
    # for i, label in enumerate(train_labels):
    #     train_targets[i] = original_vad_mapping[label]
    #
    # print(f"overall MAE of training: {(abs(train_targets - train_vads)).mean()}")

    # loading eval
    if not cached_eval:

        # load the model
        model_dict = load_model()

        eval_loss, eval_labels, eval_targets, eval_preds = \
            only_eval(model_dict['dev_dataset'],
                      model_dict['model'], model_dict['model_args'],
                      1)

        # optionally - cache the eval data (SAVES ALOT OF TIME for next time):
        np.savetxt(os.path.join(summary_path, EVAL_LABELS_CSV), eval_labels)
        np.savetxt(os.path.join(summary_path, EVAL_TARGETS_CSV), eval_targets)
        np.savetxt(os.path.join(summary_path, EVAL_PREDS_CSV), eval_preds)
        print(f"overall MAE of eval: {(abs(eval_targets - eval_preds)).mean()}")

    else:
        eval_labels = np.loadtxt(os.path.join(summary_path, EVAL_LABELS_CSV)).astype(int)
        # eval_targets = np.loadtxt(os.path.join(summary_path, EVAL_TARGETS_CSV)).astype(float)
        eval_preds = np.loadtxt(os.path.join(summary_path, EVAL_PREDS_CSV)).astype(float)

    # ---------------- END LOADING MODEL & DATA FROM CACHE ------------------------------


    # ---------------- CLFs logic - play with stuff here:  ------------------------------
    res = rbf_accuracy_per_gamma(train_vads, train_labels, eval_preds, eval_labels)
    print(res)
    df_train_acc, df_train_f1, df_dev_acc, df_dev_f1 = res
    df_train_f1.to_pickle(os.path.join(summary_path, "df_train_f1.csv"))
    df_train_acc.to_pickle(os.path.join(summary_path, "df_train_acc.csv"))
    df_dev_f1.to_pickle(os.path.join(summary_path, "df_dev_f1.csv"))
    df_dev_acc.to_pickle(os.path.join(summary_path, "df_dev_acc.csv"))

    clfs_dict = {
        'svm': SVC(C=60, gamma=1),
        # '1NN': KNeighborsClassifier(1),
        # 'tree': DecisionTreeClassifier(criterion='entropy', random_state=0),
        # 'forest': RandomForestClassifier(max_depth=2, random_state=0),
        # 'ADABoost': AdaBoostClassifier(),
        # 'MLP': MLPClassifier(alpha=1, max_iter=1000),
        # 'XGBoost': XGBClassifier(use_label_encoder=False)
    }

    # TODO - add catboost (as another CLF)
    # # catboost CLF: --------------------------------
    # train_data = [["summer", 1924, 44],
    #               ["summer", 1932, 37],
    #               ["winter", 1980, 37],
    #               ["summer", 2012, 204]]
    #
    # eval_data = [["winter", 1996, 197],
    #              ["winter", 1968, 37],
    #              ["summer", 2002, 77],
    #              ["summer", 1948, 59]]
    #
    # cat_features = [0]
    #
    # train_label = [i for i in range(28)]
    #
    # train_dataset = Pool(data=train_vads,
    #                      label=train_labels,
    #                      cat_features=cat_features)
    #
    # eval_dataset = Pool(data=eval_data,
    #                     label=train_label,
    #                     cat_features=cat_features)
    #
    # # Initialize CatBoostClassifier
    # model = CatBoostClassifier(iterations=10,
    #                            learning_rate=1,
    #                            depth=2,
    #                            loss_function='MultiClass')
    # # Fit model
    # model.fit(train_dataset)

    # end catboost CLF: --------------------------------


    for clf in clfs_dict.values():
        clf = train_clf(clf, train_vads, train_labels,
                             eval_preds, eval_labels)

        eval_labels_preds = clf.predict(eval_preds)

        results = compute_metrics_classification(eval_labels, eval_labels_preds,
                                                 name_suffix=f"_{clf.__class__.__name__}")
        print(results)


# ---------------- RUNNING THE CLFs ----------------
MAE_5_PATH = "results\EXP_WITH_CLASSIFIERS\MAE_5_summary_regression_model_04_03_2022_15_11"
MSE_5_PATH = os.path.join("results", "EXP_WITH_CLASSIFIERS", "MSE_5_summary_regression_model_04_03_2022_12_06")
MAE_5_SCALED_3_PATH = "results\EXP_WITH_CLASSIFIERS\MAE_5_VAD_SCALED_3_summary_regression_model_04_03_2022_18_20"

run_clf(MAE_5_SCALED_3_PATH, cached_eval=True)





