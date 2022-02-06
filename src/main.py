import argparse
import json
import os
import logging

from attrdict import AttrDict

from train_eval_run import run_model
from train_eval_run.utils import init_logger, init_tensorboard_writer

from model_args import model_choices
from train_eval_run.utils import get_curr_time_for_filename

logger = logging.getLogger(__name__)


def main():
    init_logger()

    # --- Parsing CLI Arguments ---
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--config", type=str, required=True,
                            help="json config file to be used (e.g. `baseline.json`)")
    cli_args = cli_parser.parse_args()

    logger.info("***** Starting main() *****")

    # --- Initializations ---
    # Read from config file and make args  # TODO: divide the config files to groups
    config_path = os.path.join("config", cli_args.config)
    with open(config_path) as f:
        args = AttrDict(json.load(f))
    logger.info(f"Configuration file: {config_path}")
    logger.info("Training/evaluation parameters {}".format(args))

    # Choose model args
    assert args.model_args in model_choices, \
        f"Got unexpected model_args choice: `{args.model_args}`, should be from {set(model_choices.keys())}"
    model_args = model_choices.get(args.model_args)
    model_args.override_with_args(args.get('model_args_override'))
    logger.info(f"Model choice: {args.model_args}")
    logger.info(f"Model arguments: {model_args}")

    # Initiate Tensorboard
    tb_writer_path = os.path.join(args.output_dir, f"tb_summary_for_{model_args.model_name}_model_"
                                                   f"{get_curr_time_for_filename}")
    tb_writer = init_tensorboard_writer(tb_writer_path)

    # --- Run ---
    run_model.run(args, model_args, tb_writer)

    # --- Finalization ---
    tb_writer.close()


if __name__ == '__main__':
    main()
