import argparse

from train_eval_run import run_model

# Models classes
from models.model_baseline import BertForMultiLabelClassification

# Data processor classes
from data_processing.data_loader import GoEmotionsProcessor


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--config", type=str, required=True,
                            help="json config file to be used (e.g. `original.json`)")
    cli_args = cli_parser.parse_args()

    model_class: type = BertForMultiLabelClassification  # TODO this model could change
    data_processor_class: type = GoEmotionsProcessor
    run_model.run(cli_args, data_processor_class, model_class)