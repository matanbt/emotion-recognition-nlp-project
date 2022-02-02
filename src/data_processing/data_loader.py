import datasets
from datasets import Dataset

from . import data_utils
from datasets.dataset_dict import DatasetDict
import logging

from enum import Enum

import pandas as pd

NRC_VAD_LEXICON_PATH = "../data/mapping-emotions-to-vad/nrc-vad/NRC-VAD-Lexicon.txt"


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------


class VADMapperName(Enum):
    # Note: each VADMapperName should be handled in VADMapper.__init__
    NRC = "Mapping based on NRC lexicon"


class VADMapper:
    """
        Class for controlling the VAD mappings
    """

    def __init__(self, vad_mapper_name: VADMapperName, labels_names_list):
        """"
        labels_names_list - each label in it's corresponding index
        """
        self.label_idx_to_vad_mapping = None

        if vad_mapper_name is VADMapperName.NRC:
            df_nrc = pd.read_csv(NRC_VAD_LEXICON_PATH, sep="\t",
                                 names=['word', 'v', 'a', 'd'])
            df_nrc = df_nrc[df_nrc["word"].apply(lambda word: word in labels_names_list)]

            assert len(df_nrc) == 28

            df_nrc.set_index('word', inplace=True)

            # Sort emotions by labels_names_list order
            df_nrc = df_nrc.reindex(labels_names_list)

            self.label_idx_to_vad_mapping = df_nrc.values.tolist()

        else:
            logger.info("ERROR - Unexpected VADMapperName, please handle {} VADMapperName case in VADMapper.__init__")
            raise Exception\
                ("ERROR - Unexpected VADMapperName, please handle {} VADMapperName case in VADMapper.__init__")

    def map_go_emotions_labels(self, label_index):
        return self.label_idx_to_vad_mapping[label_index]


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
                 vad_mapper_name: VADMapperName = None):
        """
            args - full config
            tokenizer - hugging-face pretrained tokenizer instance
            max_length - the max_length we want our sentences to be
            vad_mapper_name - instance of VADMapperName,
                         if vad_mapper_name is not initiated (=None), no VAD will be presented.
        """
        self.args = args
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_vad = vad_mapper_name is not None

        # Fetches labels list
        self.labels_list = data_utils.get_labels_list(self.args)

        # Create vad_mapper
        self.vad_mapper = None if not self.with_vad else VADMapper(vad_mapper_name, self.labels_list)

        # Fetch raw dataset, with columns: ['id', 'text', 'labels']
        # 'self.raw_dataset' remains *untouched* in this class.
        self.raw_dataset: DatasetDict = self._fetch_raw_dataset()

        # Delete multi-labels from datasets
        train, validation, test = self.get_pandas(self.raw_dataset)
        train, validation, test = self.remove_multi_label_examples(train, validation, test)

        self.raw_dataset = datasets.DatasetDict({"train": Dataset.from_pandas(train),
                                                 "validation": Dataset.from_pandas(validation),
                                                 "test": Dataset.from_pandas(test)})

        # Will hold the processed data (data with encodings, special labels, etc)
        self.processed_dataset: DatasetDict = self.raw_dataset

    # --- Data PreProcessing ---

    def perform_full_preprocess(self):
        """ Performs full preprocessing, preparing dataset for training with Torch"""
        # TODO - consider caching + flagging to prevent this function from being called twice

        # Adds 'one_hot_labels' columns
        self.processed_dataset = \
            self.processed_dataset.map(self._hf_batch_mapper__one_hot_label, batched=True)

        # Adds 'attention_mask', 'input_ids', 'token_type_ids' columns
        self.processed_dataset = \
            data_utils.add_tokenization_features(self.processed_dataset,
                                                 self.tokenizer,
                                                 self.max_length)

        if self.with_vad:
            # Adds 'vad' column
            self.processed_dataset = \
                self.processed_dataset.map(self._hf_batch_mapper__vad_mapping,
                                                                batched=True)

        # sets the relevant columns as tensors
        self._cast_to_tensors()



    # --- Dataset Getters --

    def get_hf_dataset(self) -> DatasetDict:
        return self.processed_dataset

    @staticmethod
    def get_pandas(dataset):
        """ casts the dataset to 3 pandas data-frames - train, dev, test """
        train, dev, test = dataset["train"].to_pandas(), \
                           dataset["validation"].to_pandas(), \
                           dataset["test"].to_pandas()
        return train, dev, test

    @staticmethod
    def remove_multi_label_examples(train, dev, test):
        """ train, dev, test - 3 pandas data-frames """
        train_labels_count = train['labels'].apply(lambda labels: len(labels))
        dev_labels_count = dev['labels'].apply(lambda labels: len(labels))
        test_labels_count = test['labels'].apply(lambda labels: len(labels))

        train_one_label = train[train_labels_count == 1]
        dev_one_label = dev[dev_labels_count == 1]
        test_one_label = test[test_labels_count == 1]

        return train_one_label, dev_one_label, test_one_label


    def get_dataset_by_mode(self, mode) -> datasets.arrow_dataset.Dataset:
        """
            returns the data part by chosen 'mode', could be from ['train', 'dev', 'test'].
            the datasets returned can be used by Torch
        """
        assert mode in ['train', 'dev', 'test']
        mode2split = {'train': 'train', 'dev': 'validation', 'test': 'test'}
        return self.processed_dataset[mode2split[mode]]

    # --- Utils Methods: for data processing ---

    @staticmethod
    def _fetch_raw_dataset(is_offline=False) -> DatasetDict:
        """ returns the raw go-emotions dataset, as a hugging-face's dataset """
        if not is_offline:
            go_emotions = datasets.load_dataset("go_emotions", "simplified")
        else:
            # TODO this can also be done, if we want, by fetching offline from files
            go_emotions = None
        return go_emotions

    def _cast_to_tensors(self):
        """
        Prepares dataset for Torch integration, by casting all the features to tensors *in* args.device
        """
        features_to_tensor = ['input_ids', 'attention_mask', 'token_type_ids', 'one_hot_labels']
        if self.with_vad: features_to_tensor += ['vad']

        self.processed_dataset.set_format(type='torch',
                                          columns=features_to_tensor,
                                          device=self.args.device)


    # --- Hugging-face mappers --- (to be used in "dataset.map()" invocations)


    def _hf_batch_mapper__vad_mapping(self, examples):
        examples['vad'] = []
        for label_idx in examples['labels']:
            examples['vad'].append(self.vad_mapper.map_go_emotions_labels(label_idx[0]))
            # label_idx - list of labels corresponding to the given input
            # for now we assume we have one label per example
        return examples


    @staticmethod
    def _hf_batch_mapper__one_hot_label(examples):
        labels_list = data_utils.get_labels_list()
        examples['one_hot_labels'] = []
        for emotions_indices in examples['labels']:  # TODO (?) can be done without a loop with numpy vectorization
            # Note: each element in the one_hot_vector must be 'float', for Torch loss calculation
            one_hot_label = [float(i in emotions_indices) for i in
                             range(len(labels_list))]
            examples['one_hot_labels'].append(one_hot_label)
        return examples



# ---------------------------------------------------------------------
# TODO provide support for processing datasets such as fb-valence-arousal, emobank
class EmoBankProcessor(BaseProcessor):
    pass

# ---------------------------------------------------------------------
class FacebookVAProcessor(BaseProcessor):
    pass
