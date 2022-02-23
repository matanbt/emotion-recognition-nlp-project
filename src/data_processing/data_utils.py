import os

import pandas as pd
from datasets import DatasetDict


def add_tokenization_features(dataset: DatasetDict, tokenizer, max_length) -> DatasetDict:
    """
        dataset: full hugging-face dataset (must have 'text' column)
        tokenizer: hugging-face tokenizer
        max_length: the fixed length the tokens-vectors will have

        returns new dataset, with tokenization features, based on the 'text' column
    """

    def _hf_batch_mapper__tokenize(examples):
        return tokenizer(examples["text"], max_length=max_length,
                         padding='max_length', truncation=True)

    return dataset.map(_hf_batch_mapper__tokenize, batched=True)  # map() is not in-place


# GoEmotions Utils
go_emotions_cached_labels_list = None

def get_ge_labels_list(args=None):
    """
        returns a list with GoEmotions labels
    """
    # Get cached labels-list if exists
    global go_emotions_cached_labels_list
    if go_emotions_cached_labels_list is not None:
        return go_emotions_cached_labels_list

    assert args is not None, "Initial call for get_labels_list() must be with parameter 'args'"

    # Else - read labels-list
    labels = []
    with open(os.path.join(args.data_dir, args.label_file), "r", encoding="utf-8") as f:
        for line in f:
            labels.append(line.rstrip())
    go_emotions_cached_labels_list = labels.copy()
    return labels


def get_nrc_vad_mapping(nrc_vad_mapping_path, labels_names_list):
    df_nrc = pd.read_csv(nrc_vad_mapping_path, sep="\t", names=['word', 'v', 'a', 'd'])
    df_nrc = df_nrc[df_nrc["word"].apply(lambda word: word in labels_names_list)]

    assert len(df_nrc) == 28

    df_nrc.set_index('word', inplace=True)

    # Sort emotions by labels_names_list order
    df_nrc = df_nrc.reindex(labels_names_list)

    return df_nrc
