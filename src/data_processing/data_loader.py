import datasets
from datasets import Dataset

from . import data_utils
from datasets.dataset_dict import DatasetDict
import logging
import numpy as np

from enum import Enum

import pandas as pd

NRC_VAD_LEXICON_PATH = "../data/mapping-emotions-to-vad/nrc-vad/NRC-VAD-Lexicon.txt"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------

class VADMapperName(Enum):
    # Note: each VADMapperName should be handled in VADMapper.__init__
    NRC = "Mapping based on NRC lexicon"
    SCALED_NRC_1 = "Mapping by NRC, scaled with QuantileTransformer(n_quantiles=28): uniform dist all the way.."
    SCALED_NRC_2 = "Mapping by NRC, scaled with QuantileTransformer(n_quantiles=15)"
    SCALED_NRC_3 = "Mapping by NRC, scaled with QuantileTransformer(n_quantiles=21)"
    SCALED_NRC_4 = "Mapping by NRC, scaled with QuantileTransformer(n_quantiles=18)"
    SCALED_NRC_5 = "Mapping by NRC, scaled with QuantileTransformer(n_quantiles=10)"
    SCALED_NRC_6 = "Mapping by NRC, scaled with QuantileTransformer(n_quantiles=5)"

    # baseline for VAD mappings, to prove that a good mapping helps.
    NAIVE = "Mapping naively, by defining evenly spaced VADs"


class VADMapper:
    """
        Class for controlling the VAD mappings
    """

    def __init__(self, args, vad_mapper_name: VADMapperName, labels_names_list):
        """"
        args - contains the vad mappings paths
        vad_mapper_name - the dataset the mapper shall be set to, an instance of VADMapperName
        labels_names_list - each label in its corresponding index
        """
        self.label_idx_to_vad_mapping = None

        if vad_mapper_name is VADMapperName.NRC:
            logger.info("VADMapper using NRC for VAD mappings.")
            df_nrc = data_utils.get_nrc_vad_mapping(args.nrc_vad_mapping_path, labels_names_list)
            self.label_idx_to_vad_mapping = df_nrc.values.tolist()

        elif vad_mapper_name in (VADMapperName.SCALED_NRC_1, VADMapperName.SCALED_NRC_2,
                                 VADMapperName.SCALED_NRC_3, VADMapperName.SCALED_NRC_4,
                                 VADMapperName.SCALED_NRC_5, VADMapperName.SCALED_NRC_6):
            from sklearn.preprocessing import QuantileTransformer
            n_quantiles = 28  # default (SCALED_NRC_1)
            if vad_mapper_name is VADMapperName.SCALED_NRC_2:
                n_quantiles = 15
            elif vad_mapper_name is VADMapperName.SCALED_NRC_3:
                n_quantiles = 21
            elif vad_mapper_name is VADMapperName.SCALED_NRC_4:
                n_quantiles = 18
            elif vad_mapper_name is VADMapperName.SCALED_NRC_5:
                n_quantiles = 10
            elif vad_mapper_name is VADMapperName.SCALED_NRC_6:
                n_quantiles = 5

            logger.info("VADMapper using *scaled* NRC for VAD mappings, "
                        f"with n_quantiles={n_quantiles}")
            df_nrc = data_utils.get_nrc_vad_mapping(args.nrc_vad_mapping_path, labels_names_list)
            self.label_idx_to_vad_mapping = df_nrc.values.tolist()
            scaler = QuantileTransformer(n_quantiles=n_quantiles,
                                         output_distribution='uniform')
            self.label_idx_to_vad_mapping = scaler.fit_transform(self.label_idx_to_vad_mapping)

        elif vad_mapper_name is VADMapperName.NAIVE:
            _even_space = 1 / (len(labels_names_list) - 1)
            _vad_dim = 3
            self.label_idx_to_vad_mapping = np.zeros((len(labels_names_list), _vad_dim))

            # evenly spreads the VADs in the unit-cube
            for i in range(_vad_dim):
                self.label_idx_to_vad_mapping[:, i] = np.arange(0, 1 + _even_space, _even_space)
            self.label_idx_to_vad_mapping = self.label_idx_to_vad_mapping.tolist()

        else:
            raise Exception(f"ERROR - Unexpected VADMapperName, please handle {vad_mapper_name} "
                             f"VADMapperName case in VADMapper.__init__()")

    def map_go_emotions_labels(self, label_index):
        return self.label_idx_to_vad_mapping[label_index]

    def get_emotions_vads_lst(self):
        """ returns the list of the emotions' vad values (ordered by the emotions' labels order) """
        return list(self.label_idx_to_vad_mapping)


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
        logger.warning("WARNING: method perform_full_preprocess() was NOT overridden by dataset-processor.")

    def get_dataset_by_mode(self, mode):
        """
            will return the 'mode' part of the dataset, ready to be used by Torch.
            MUST be overridden by the processor class.
        """
        logger.warning("WARNING: method get_dataset_by_mode() was NOT overridden by dataset-processor.")

    @staticmethod
    def get_labels_list(args):
        """
            will return the list of labels (most of the times emotions) of the data.
        """
        logger.warning("WARNING: method get_labels_list() was NOT overridden by dataset-processor.")


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
                 remove_multi_lables=True, # TODO this is a temporary arg that should be removed! multi-labels should be dealt with !
                 add_external_training_fb=False,  # TODO This is another experimental parameter
                 add_external_training_eb=False,  # TODO This is another experimental parameter
                 vad_mapper_name: VADMapperName = None):
        """
            args - full config
            tokenizer - hugging-face pretrained tokenizer instance
            max_length - the max_length we want our sentences to be
            vad_mapper_name - instance of VADMapperName,
                         if vad_mapper_name is not initiated (=None), no VAD will be presented.
        """
        logger.info("GoEmotionsProcessor: Fetching data and initiating the data-processor with: \n"
                    f"args, tokenizer={tokenizer}, max_length={max_length}, vad_mapper_name={vad_mapper_name}")

        self.args = args
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_vad = vad_mapper_name is not None
        self.with_noise = args.with_noise
        self.noise_param = None if not self.with_noise else args.noise_param
        self.add_external_training_fb = add_external_training_fb
        self.add_external_training_eb = add_external_training_eb

        # Fetches labels list
        self.labels_list = GoEmotionsProcessor.get_labels_list(args)

        # Create vad_mapper
        self.vad_mapper = None if not self.with_vad else VADMapper(args, vad_mapper_name, self.labels_list)
        # Create min_distance_for_each_dim (minimum-not zero distance for each dim of the vad)
        self.min_distance_for_each_dim = None if not self.with_vad else self.calc_min_distance_for_each_vad_dim()

        # Fetch raw dataset, with columns: ['id', 'text', 'labels']
        # 'self.raw_dataset' remains *untouched* in this class.
        self.raw_dataset: DatasetDict = self._fetch_raw_dataset()

        if remove_multi_lables:
            self.raw_dataset = self.raw_dataset.filter(self._hf_filterer__remove_multi_label)

        # Will hold the processed data (data with encodings, special labels, etc)
        self.processed_dataset: DatasetDict = self.raw_dataset

    # --- Data PreProcessing ---

    def perform_full_preprocess(self):
        """ Performs full preprocessing, preparing dataset for training with Torch"""
        # TODO - consider caching + flagging to prevent this function from being called twice
        logger.info("GoEmotionsProcessor: Performs full preprocessing of the data")

        # Adds 'one_hot_labels' columns
        self.processed_dataset = \
            self.processed_dataset.map(self._hf_batch_mapper__one_hot_label, batched=True)

        if self.with_vad:
            # Adds 'vad' column
            self.processed_dataset = \
                self.processed_dataset.map(self._hf_batch_mapper__vad_mapping, batched=True)

        if self.add_external_training_fb:
            # adds more VAD example to training :) (Facebook bank)
            external_ds = datasets.load_dataset('csv', data_files={
                                'train': 'data/fb-va/dataset-fb-valence-arousal-anon.csv'})
            external_ds = external_ds.map(self._hf_mapper__add_vad_to_fb_va)
            self.processed_dataset['train'] = datasets.concatenate_datasets([self.processed_dataset['train'],
                                                                             external_ds['train']])

        if self.add_external_training_eb:
            # adds more VAD example to training :) (EmoBank)
            AMOUNT_OF_EXAMPLES_TO_ADD = 5000
            external_ds = datasets.load_dataset('csv', data_files={
                                'train': 'data/emobank/emobank.csv'})
            external_ds = external_ds.filter(lambda example: example['split'] == 'train')
            external_ds = external_ds['train'].select(range(AMOUNT_OF_EXAMPLES_TO_ADD))
            external_ds = external_ds.map(self._hf_mapper__add_vad_to_eb_vad)
            self.processed_dataset['train'] = datasets.concatenate_datasets([self.processed_dataset['train'],
                                                                             external_ds])

        # Adds 'attention_mask', 'input_ids', 'token_type_ids' columns
        self.processed_dataset = \
            data_utils.add_tokenization_features(self.processed_dataset,
                                                 self.tokenizer,
                                                 self.max_length)
        # sets the relevant columns as tensors
        self._cast_to_tensors()

    def calc_min_distance_for_each_vad_dim(self):

        # helper function
        def get_min_non_zero_distance(np_arr, dim_idx):
            """ returns the minimum-not zero distance between values from np_arr in dimension = dim_idx"""
            distances = np.abs(np_arr[:, dim_idx, np.newaxis] - np_arr[np.newaxis, :, dim_idx])
            # set 0 distances to infinity
            distances[distances == 0] = 2
            return np.min(distances)

        emotions_vads_np_arr = np.array(self.get_emotions_vads_lst())
        return [get_min_non_zero_distance(emotions_vads_np_arr, dim) for dim in range(emotions_vads_np_arr.shape[1])]

    # --- Dataset Getters --

    def get_hf_dataset(self) -> DatasetDict:
        return self.processed_dataset

    @staticmethod
    def get_pandas(dataset: DatasetDict):
        """ casts the dataset to 3 pandas data-frames - train, dev, test """
        train, dev, test = dataset["train"].to_pandas(), \
                           dataset["validation"].to_pandas(), \
                           dataset["test"].to_pandas()
        return train, dev, test

    def get_dataset_by_mode(self, mode) -> datasets.arrow_dataset.Dataset:
        """
            returns the data part by chosen 'mode', could be from ['train', 'dev', 'test'].
            the datasets returned can be used by Torch
        """
        assert mode in ['train', 'dev', 'test']
        mode2split = {'train': 'train', 'dev': 'validation', 'test': 'test'}
        return self.processed_dataset[mode2split[mode]]

    @staticmethod
    def get_labels_list(args=None):
        return data_utils.get_ge_labels_list(args)

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

    def get_emotions_vads_lst(self):
        """ returns the list of the emotions' vad values (ordered by the emotions' labels order)
        can be used only when self.with_vad is True """
        assert self.with_vad is True
        return self.vad_mapper.get_emotions_vads_lst()

    # --- Hugging-face mappers --- (to be used in "dataset.map()" invocations)

    def _hf_batch_mapper__vad_mapping(self, examples):
        examples['vad'] = []
        for label_idx in examples['labels']:
            # label_idx - list of labels corresponding to the given input
            _vad = self.vad_mapper.map_go_emotions_labels(label_idx[0])
            if self.with_noise:
                _vad = _vad + np.random.normal(loc=[0.0, 0.0, 0.0],
                                               scale=np.array(self.min_distance_for_each_dim) * self.noise_param)
            examples['vad'].append(_vad)
            # TODO overcome this assumption: for now we assume we have one label per example
        return examples

    @staticmethod
    def _hf_batch_mapper__one_hot_label(examples):
        labels_list = GoEmotionsProcessor.get_labels_list()
        examples['one_hot_labels'] = []
        for emotions_indices in examples['labels']:  # TODO (?) can be done without a loop with numpy vectorization
            # Note: each element in the one_hot_vector must be 'float', for Torch loss calculation
            one_hot_label = [float(i in emotions_indices) for i in
                             range(len(labels_list))]
            examples['one_hot_labels'].append(one_hot_label)
        return examples

    @staticmethod
    def _hf_filterer__remove_multi_label(example):
        return len(example['labels']) == 1

    @staticmethod
    def _hf_mapper__add_vad_to_fb_va(example):
        result = {}
        result['text'] = example['Anonymized Message']
        result['one_hot_labels'] = [0.0] * 28  # dummy labeling

        _reducer = lambda a, b: (a + b) / 2
        _scaler = lambda a: (a - 1) / 8

        result['vad'] = []
        result['vad'].append(_scaler(_reducer(example['Valence1'], example['Valence2'])))
        result['vad'].append(_scaler(_reducer(example['Arousal1'], example['Arousal2'])))
        result['vad'].append(0.5)  # dominance is neutral

        return result

    @staticmethod
    def _hf_mapper__add_vad_to_eb_vad(example):
        result = {}
        result['text'] = example['text']
        result['one_hot_labels'] = [0.0] * 28  # dummy labeling

        _scaler = lambda a: (a - 1) / 4

        result['vad'] = []
        result['vad'].append(_scaler(example['V']))
        result['vad'].append(_scaler(example['A']))
        result['vad'].append(_scaler(example['D']))

        return result


# ---------------------------------------------------------------------
# TODO provide support for processing datasets such as fb-valence-arousal, emobank
class EmoBankProcessor(BaseProcessor):
    pass


# ---------------------------------------------------------------------
class FacebookVAProcessor(BaseProcessor):
    pass
