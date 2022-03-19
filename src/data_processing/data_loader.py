import datasets
from datasets import Dataset

from . import data_utils
from datasets.dataset_dict import DatasetDict
import logging
import numpy as np

from enum import Enum

import pandas as pd

NRC_VAD_LEXICON_PATH = "../data/mapping-emotions-to-vad/nrc-vad/NRC-VAD-Lexicon.txt"
RANDOM_UNIFORM_28_POINTS_VAD_SPACE = [(0.9654290681605461, 0.7810771796285593, 0.4768636002935215), (0.7373966731895499, 0.7122276276979275, 0.4947215068494365), (0.6841013792263198, 0.8809669493091409, 0.6610419253316695), (0.3521918512858728, 0.39930532523910756, 0.2911827841339677), (0.7650014821503568, 0.47233924377871983, 0.17455953282087977), (0.7690961591751533, 0.9242248607972423, 0.23398509695659053), (0.48410824561164845, 0.8794289429243328, 0.4167677853971644), (0.662290910696874, 0.9437455295744083, 0.6882889095840883), (0.5433498644720054, 0.17299684899029055, 0.9818494961462796), (0.17022376506820014, 0.5873928155135671, 0.22513044256346615), (0.6010017469161022, 0.18571961395133862, 0.15045425243564214), (0.38610203560749823, 0.577605318981289, 0.21899549487503667), (0.500139293912145, 0.04853269659565196, 0.4724691915942453), (0.6828254609604009, 0.9678248132734697, 0.4286237722214987), (0.5683420059179503, 0.9192562372931478, 0.6285780332014991), (0.7289428520747615, 0.4638312438164799, 0.8945462777249052), (0.6150695866704562, 0.5952760543752944, 0.007797992713616364), (0.4937254068527138, 0.25393122026052384, 0.29903009827380855), (0.1324345322630357, 0.03782365342178218, 0.9200629040032965), (0.1550001386592068, 0.04840202567938734, 0.4588001997625426), (0.8983050791286917, 0.7543552186427225, 0.495002014737028), (0.9708656320285785, 0.8925874272286406, 0.49663278956786594), (0.6410405327567263, 0.9476706417519792, 0.5246646641888845), (0.7735122314859783, 0.20746667085165338, 0.7362661616860379), (0.027141697328987524, 0.6048989822639763, 0.07308764456686856), (0.0021745931270553687, 0.9352963584306079, 0.03344494256464425), (0.07135336798138592, 0.3433426684186681, 0.1809256274226031), (0.10299168124185809, 0.35790667900678297, 0.3615576586816426)]
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
    NRC1000 = "Mapping by NRC, multiplied by 1000"

    # baseline for VAD mappings, to prove that a good mapping helps.
    NAIVE = "Mapping naively, by defining evenly spaced VADs"
    RANDOM_UNIFORM = "Mapping by random points generated from uniform distribution and scaled with QuantileTransformer(n_quantiles=28)"


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
        from sklearn.preprocessing import QuantileTransformer
        self.label_idx_to_vad_mapping = None

        if vad_mapper_name is VADMapperName.NRC:
            logger.info("VADMapper using NRC for VAD mappings.")
            df_nrc = data_utils.get_nrc_vad_mapping(args.nrc_vad_mapping_path, labels_names_list)
            self.label_idx_to_vad_mapping = df_nrc.values.tolist()

        elif vad_mapper_name is VADMapperName.NRC1000:
            logger.info("VADMapper using NRC for VAD mappings, multiplied by 1000.")
            df_nrc = data_utils.get_nrc_vad_mapping(args.nrc_vad_mapping_path,
                                                    labels_names_list) * 1000
            self.label_idx_to_vad_mapping = df_nrc.values.tolist()

        elif vad_mapper_name in (VADMapperName.SCALED_NRC_1, VADMapperName.SCALED_NRC_2,
                                 VADMapperName.SCALED_NRC_3, VADMapperName.SCALED_NRC_4,
                                 VADMapperName.SCALED_NRC_5, VADMapperName.SCALED_NRC_6):

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

        elif vad_mapper_name is VADMapperName.RANDOM_UNIFORM:

            scaler = QuantileTransformer(n_quantiles=28,
                                         output_distribution='uniform')
            self.label_idx_to_vad_mapping = scaler.fit_transform(RANDOM_UNIFORM_28_POINTS_VAD_SPACE)
            print(self.label_idx_to_vad_mapping.tolist())


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

    def process_example(self, text):
        return data_utils.add_tokenization_features_example(text, self.tokenizer, self.max_length)

    def process_new_dataset(self, dataset: DatasetDict, max_length):
        return data_utils.add_tokenization_features(dataset, self.tokenizer, max_length)

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
