import os
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

def get_labels_list(args=None):
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