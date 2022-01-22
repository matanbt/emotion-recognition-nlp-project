import os

import datasets
import pandas as pd
import torch
from datasets.dataset_dict import DatasetDict

# ---------------------------------------------------------------------
class VADMapper:
    """
        Class for controlling the VAD mappings
    """

    # TODO
    # an instance will be provided to GoEmotionsProcessor.__init__().
    # this class will implement: map_go_emotions_labels() for the use of GoEmotionsProcessor

    def __init__(self):
        pass

    def map_go_emotions_labels(self, labels):
        pass


# ---------------------------------------------------------------------
class BaseProcessor:
    """
        each dataset-processor should extend this class.
    """

    def perform_full_preprocess(self):
        """
            triggers the full preprocessing of the dataset.
            MUST be overridden by the processor class.
        """
        print("WARNING: method perform_full_preprocess() was NOT overridden by dataset-processor.")

    def get_dataset_by_mode(self, mode):
        """
            will return the 'mode' part of the dataset, ready to be used by Torch.
            MUST be overridden by the processor class.
        """
        print("WARNING: method get_dataset_by_mode() was NOT overridden by dataset-processor.")


# ---------------------------------------------------------------------
go_emotions_cached_labels_list = None

# TODO add logs for data processing
class GoEmotionsProcessor(BaseProcessor):
    """
    fetcher and preprocessor for the GoEmotion dataset,
    - This class wraps GoEmotion *processed* dataset.
    - based on hugging-face dataset object.
    """

    def __init__(self,
                 args,
                 tokenizer,
                 max_length: int,
                 vad_mapper: VADMapper = None):
        """
            args - full config
            tokenizer - hugging-face pretrained tokenizer instance
            max_length - the max_length we want our sentences to be
            vad_mapper - instance of VADMapper, with method map_go_emotions_labels().
                         if VADMapper is not initiated (=None), no VAD will be presented.
        """
        self.args = args
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vad_mapper = vad_mapper
        self.with_vad = vad_mapper is not None

        # Fetches labels list
        self.labels_list = self.fetch_labels_list(self.args)

        # Fetch raw dataset, with columns: ['id', 'text', 'labels']
        # 'self.raw_dataset' remains *untouched* in this class.
        self.raw_dataset: DatasetDict = self._get_raw_dataset()

        # Will hold the processed data (data with encodings, special labels, etc)
        self.processed_dataset: DatasetDict = self.raw_dataset

    # --- Data to Features ---

    def _add_token_features(self):
        """ Add to 'processed_dataset' the columns: ['attention_mask', 'input_ids', 'token_type_ids'] """
        self.processed_dataset = self._add_tokenization_mapping(self.processed_dataset,
                                                                self.tokenizer, self.max_length)

    def _add_one_hot_labels(self):
        """ Add to 'processed_dataset' the columns: ['one_hot_labels']
            based on the indices in column 'labels' """

        # converts labels to one-hot
        def one_hot_label_batch(examples):
            examples['one_hot_labels'] = []
            for emotions_indices in examples['labels']:  # TODO (?) can be done without a loop with numpy vectorization
                # Note: each element in the one_hot_vector must be 'float', for Torch loss calculation
                one_hot_label = [float(i in emotions_indices) for i in
                                 range(len(self.labels_list))]
                examples['one_hot_labels'].append(one_hot_label)
            return examples

        self.processed_dataset = self.processed_dataset.map(one_hot_label_batch, batched=True)

    def _add_vad_features(self):
        """ adds VAD mappings to processed dataset """
        self.processed_dataset = self._add_vad_mapping(self.processed_dataset)

    def _cast_to_tensors(self):
        """
        Prepares dataset for Torch integration, by casting all the features to tensors *in* args.device
        """
        features_to_tensor = ['input_ids', 'attention_mask', 'token_type_ids', 'one_hot_labels']
        if self.with_vad: features_to_tensor += ['vad']

        self.processed_dataset.set_format(type='torch',
                                          columns=features_to_tensor,
                                          device=self.args.device)


    # --- Data PreProcessing ---

    def perform_full_preprocess(self):
        """ Performs full preprocessing, preparing dataset for training with Torch"""
        # TODO - consider caching + flagging to prevent this function from being called twice
        self._add_one_hot_labels()
        self._add_token_features()
        if self.with_vad:
            self._add_vad_features()
        self._cast_to_tensors()

    # --- Dataset Getters --

    def get_hf_dataset(self) -> DatasetDict:
        return self.processed_dataset

    def get_pandas(self):
        """ casts the processed dataset to 3 pandas data-frames - train, dev, test """
        train, dev, test = self.processed_dataset["train"].to_pandas(), \
                           self.processed_dataset["validation"].to_pandas(), \
                           self.processed_dataset["test"].to_pandas()
        return train, dev, test

    def get_dataset_by_mode(self, mode) -> datasets.arrow_dataset.Dataset:
        """
            returns the data part by chosen 'mode', could be from ['train', 'dev', 'test'].
            the datasets returned can be used by Torch
        """
        assert mode in ['train', 'dev', 'test']
        mode2split = {'train': 'train', 'dev': 'validation', 'test': 'test'}
        return self.processed_dataset[mode2split[mode]]

    # --- Utils static-methods: for data processing (TODO move to data-utils?) ---

    @staticmethod
    def fetch_labels_list(args):
        global go_emotions_cached_labels_list
        if go_emotions_cached_labels_list:
            return go_emotions_cached_labels_list

        labels = []
        with open(os.path.join(args.data_dir, args.label_file), "r", encoding="utf-8") as f:
            for line in f:
                labels.append(line.rstrip())
        return labels

    @staticmethod
    def _get_raw_dataset() -> DatasetDict:
        """ returns the raw go-emotions dataset, as an hugging-face's dataset """
        go_emotions = datasets.load_dataset("go_emotions", "simplified")
        # TODO this can also be done, if we want, by fetching offline from files
        #  (i.e. defining _get_raw_dataset_offline())
        return go_emotions

    @staticmethod
    def _add_tokenization_mapping(dataset: DatasetDict, tokenizer, max_length) -> DatasetDict:
        """ returns new dataset, with tokenization features, based on the 'text' column """

        def tokenize_batch(examples):
            return tokenizer(examples["text"], max_length=max_length,
                             padding='max_length', truncation=True)

        return dataset.map(tokenize_batch, batched=True)  # map() is not in-place

    @staticmethod
    def _add_vad_mapping(dataset: DatasetDict) -> DatasetDict:
        """ returns new dataset, with 'vad' column, based on 'labels' """

        def vad_mapping_batch(examples):
            examples['vad'] = []
            for emotions_indices in examples['labels']:  # TODO (?) can be done without a loop with numpy vectorization
                # Todo - call VADMapper API with `example['labels']`
                examples['vad'].append([1, 2, 3])
            return examples

        return dataset.map(vad_mapping_batch, batched=True)

# ---------------------------------------------------------------------
# TODO provide support for processing datasets such as fb-valence-arousal, emobank
class EmoBankProcessor(BaseProcessor):
    pass

# ---------------------------------------------------------------------
class FacebookVAProcessor(BaseProcessor):
    pass
