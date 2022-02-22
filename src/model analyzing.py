# %%
# ------------ loading model for evaluation ------------
# TODO (not mandatory) - maybe more easy - 5. Save and load entire model -
#  https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html
import torch

import os

import json

from attrdict import AttrDict

from transformers import (
    BertConfig, BertTokenizer,
)

from src.models.model_regression import BertForMultiDimensionRegression
from src.model_args import model_choices

PATH_OF_SAVED_MODEL = "C:/Users/shir7/Desktop/Semester A year 3 2022/NLP/project/results/MAE_LR_5_summary_regression_model_21_02_2022_09_43/checkpoint-22700/pytorch_model.bin"
CONFIG_NAME = "regression.json"

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


# %%
import matplotlib.pyplot as plt

def show_distribution(points, axis_num, label_name, color="g"):
    """
    for axis=axis_num, plot the distribution of the points
    """
    # Normalize
    kwargs = dict(alpha=0.5, bins=500, density=True, stacked=True)

    # Plot
    plt.hist(points[:, axis_num], **kwargs, color=color, label=label_name)
    plt.gca().set(title=f'Probability Histogram', ylabel='Probability')
    plt.xlim(0, 1)
    plt.ylim(0, 50)
    plt.legend()


# %%
from train_eval_run.evaluate import only_eval
NUM_OF_EXAMPLES = 2000

eval_loss, targets, preds = only_eval(dev_dataset.select(range(NUM_OF_EXAMPLES)), model, model_args, 1)

# %%



from train_eval_run.evaluate import only_eval
NUM_OF_EXAMPLES = 2000

tr_eval_loss, tr_targets, tr_preds = only_eval(train_dataset.select(range(NUM_OF_EXAMPLES)), model, model_args, 1)


#%%
plt.clf()
show_distribution(preds, 0, "V preds", "g")
show_distribution(targets, 0, "V targets", "r")
plt.show()

#%%
plt.clf()
show_distribution(preds, 1, "A preds", "g")
show_distribution(targets, 1, "A targets", "r")
plt.show()

#%%
plt.clf()
show_distribution(preds, 2, "D preds", "g")
show_distribution(targets, 2, "D targets", "r")
plt.show()

#%%
print("unique target v: ", len(set(targets[:, 0])))
print("unique target a: ", len(set(targets[:, 1])))
print("unique target d: ", len(set(targets[:, 2])))

print("unique pred v: ", len(set(preds[:, 0])))
print("unique pred a: ", len(set(preds[:, 1])))
print("unique pred d: ", len(set(preds[:, 2])))

# %%
import numpy as np

emo = [(0.969, 0.583, 0.726),
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

emo = np.array(emo)


# %%

print(len(set(emo[:, 0])))  # 25
print(len(set(emo[:, 1])))  # 28
print(len(set(emo[:, 2])))  # 28

# %%
# find min distance between emotions in each dim of the vad

def print_min_distance(dim, dim_name):
    # prints the min- not zero distance between vals in dimention = dim

    distances = np.abs(emo[:, dim, np.newaxis] - emo[np.newaxis, :, dim])
    # set 0 distances to infinity
    distances[distances == 0] = 2
    print(f"{dim_name} min distance = {np.min(distances)}")


print_min_distance(0, 'V')
print_min_distance(1, 'A')
print_min_distance(2, 'D')