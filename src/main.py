import argparse

from constants import *
from src.train_eval_run.ModelConfig import ModelConfig

from train_eval_run import run_model

# Models classes
from models.model_baseline import BertForMultiLabelClassification

# Data processor classes
from data_processing.data_loader import GoEmotionsProcessor

from train_eval_run.utils import (
    compute_metrics_classification,
    init_logger,
)

import json
import os
import logging

from attrdict import AttrDict

logger = logging.getLogger(__name__)
init_logger()

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--config", type=str, required=True,
                            help="json config file to be used (e.g. `original.json`)")
    cli_args = cli_parser.parse_args()

    data_processor_class: type = GoEmotionsProcessor

    logger.info("***** Starting main() *****")

    # --- Initializations ---
    # Read from config file and make args
    config_filename = "{}".format(cli_args.config)
    with open(os.path.join("config", config_filename)) as f:
        args = AttrDict(json.load(f))  # Note: don't add additional attributes to args
    logger.info("Training/evaluation parameters {}".format(args))

    # args.output_dir = ... # TODO add time-stamp to path

    # TODO this could change
    model_config = ModelConfig("classic_multi_label", BertForMultiLabelClassification, compute_metrics_classification,
                               "one_hot_labels", SIGMOID_FUNC, args)

    run_model.run(args, data_processor_class, model_config)
