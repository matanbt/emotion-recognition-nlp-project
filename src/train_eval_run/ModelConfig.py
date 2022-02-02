from .utils import (
    init_tensorboard_writer
)

import os

class ModelConfig:

    def __init__(self, model_name, model_class, compute_metrics, target_name, func_on_model_output, args, vad_mapper_name, num_dim=None):
        """"
        model_class - class from models (e.g.: model_class.model_baseline.BertForMultiLabelClassification)
        compute_metrics - function that computes the relevant metrics (e.g.: utils.compute_metrics_classification)
        target_name - function to apply on the output of the model (e.g.: "one_hot_labels")
        func_on_model_output - name of column contains the target for each input (e.g.: constants.SIGMOID_FUNC)
        vad_mapper_name - data_loader.VADMapperName enum
        num_dim - relevant only for regression models, number of dimensions the model need to estimate (e.g.: in VAD =3)
        """
        self.model_class = model_class
        self.compute_metrics = compute_metrics
        self.target_name = target_name
        self.func_on_model_output = func_on_model_output
        self.tb_writer = init_tensorboard_writer(os.path.join(args.output_dir, f"tb_summary_for_{model_name}_model"))
        self.vad_mapper_name = vad_mapper_name
        self.num_dim = num_dim
