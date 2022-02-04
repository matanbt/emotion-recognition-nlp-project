import argparse
import json
import os
import logging

from attrdict import AttrDict

from train_eval_run import run_model
from train_eval_run.utils import init_logger, init_tensorboard_writer

from model_args import classic_multi_label_model_conf

logger = logging.getLogger(__name__)


def main():
    init_logger()

    # CLI Arguments:
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--config", type=str, required=True,
                            help="json config file to be used (e.g. `original.json`)")
    cli_args = cli_parser.parse_args()

    logger.info("***** Starting main() *****")

    # --- Initializations ---
    # Read from config file and make args
    config_filename = "{}".format(cli_args.config)
    with open(os.path.join("config", config_filename)) as f:
        args = AttrDict(json.load(f))  # Note: don't add additional attributes to args
    logger.info("Training/evaluation parameters {}".format(args))

    # Choose model args
    model_args = classic_multi_label_model_conf

    # args.output_dir = ... # TODO add time-stamp to path
    tb_writer = init_tensorboard_writer(os.path.join(args.output_dir, f"tb_summary_for_{model_args.model_name}_model"))

    run_model.run(args, model_args, tb_writer)

    tb_writer.close()


if __name__ == '__main__':
    main()
