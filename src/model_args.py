from dataclasses import dataclass

from train_eval_run.utils import compute_metrics_classification, SIGMOID_FUNC, compute_metrics_regression, IDENTITY_FUNC
from data_processing.data_loader import VADMapperName, GoEmotionsProcessor
from models.model_baseline import BertForMultiLabelClassification
from models.model_regression import BertForMultiDimensionRegression


@dataclass
class ModelArgs:
    """
    An instance of this class will contain the model *specific* arguments.
    Basically wraps a `model` with everything it needs to be trained / evaluated.

    model_name - an informative name for the model
    model_class - class from models (e.g.: model_class.model_baseline.BertForMultiLabelClassification)
    data_processor_class - class from data_loader.py (e.g. data_loader.GoEmotionsProcessor)
    compute_metrics - function that computes the relevant metrics (e.g.: utils.compute_metrics_classification)
    func_on_model_output - name of column contains the target for each input (e.g.: constants.SIGMOID_FUNC)
    target_name - function to apply on the output of the model (e.g.: "one_hot_labels")

    Optional (defaults to None):
    vad_mapper_name - data_loader.VADMapperName enum
    num_dim - relevant only for regression models, number of dimensions the model need to estimate (e.g.: in VAD =3)
    hidden_layers_count - amount of NN layers in the "output-layer" of the model (default to `1`)
    threshold - the threshold for classification problems, used in evaluation.
    """
    model_name: str

    # Classes:
    model_class: type
    data_processor_class: type

    # Functions:
    compute_metrics: callable
    func_on_model_output: callable

    # Data metadata:
    target_name: str

    # Optional (regression oriented):
    vad_mapper_name: VADMapperName = None
    num_dim: int = None
    hidden_layers_count: int = 1

    # Optional (classification oriented)
    threshold: float = None


def get_model_args_from_json(filename: str) -> ModelArgs:
    # TODO can add a feature to create a ModelArgs instance from a json definition.
    pass


# -------------------------------------- PreSet Model Args  --------------------------------------

classic_multi_label_model_conf = ModelArgs("classic_multi_label",
                                           BertForMultiLabelClassification,
                                           GoEmotionsProcessor,
                                           compute_metrics_classification,
                                           SIGMOID_FUNC,
                                           "one_hot_labels",
                                           threshold=0.3)

# ---------------------------

classic_vad_regression_model_conf = ModelArgs("classic_multi_label",
                                              BertForMultiDimensionRegression,
                                              GoEmotionsProcessor,
                                              compute_metrics_regression,
                                              IDENTITY_FUNC,
                                              "vad",
                                              vad_mapper_name=VADMapperName.NRC,
                                              num_dim=3,
                                              hidden_layers_count=3)
# ---------------------------
# We can add more ModelArgs instances here...
# ---------------------------------------------------------------------

model_choices = {
    'basic': classic_multi_label_model_conf,
    'regression': classic_vad_regression_model_conf,
    # we can add more model choices here... ('our-model-name': the-ModelArgs-instance)
}
