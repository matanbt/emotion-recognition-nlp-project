"""
Will be like run_goemotions.py
BUT it'll train the regression models and evaluate them (for regression only! no mapping yet)
(all evaluations should be on goEmotions dev-set)
"""

import argparse
import json
import logging
import os
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from attrdict import AttrDict

from transformers import (
    BertConfig,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)

import data_utils
from model_baseline import BertForMultiDimensionRegression
from utils import (
    init_logger,
    init_tensorboard_writer,
    set_seed,
    compute_metrics_regression
)
from data_loader import (
    GoEmotionsProcessor
)

from train import train
from evaluate import evaluate

MATRICS_FUNC = compute_metrics_regression
IDENTITY_FUNC = lambda x: x

logger = logging.getLogger(__name__)
tb_writer = None  # initialized in main()


def main(cli_args):
    logger.info("***** Starting main() *****")

    # --- Initializations ---
    # Read from config file and make args
    config_filename = "{}.json".format(cli_args.taxonomy)
    with open(os.path.join("config", config_filename)) as f:
        args = AttrDict(json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))

    # args.output_dir = ... # TODO add time-stamp to path

    init_logger()
    global tb_writer  # we define the TensorBoard-summary-writer to be used across this file
    tb_writer = init_tensorboard_writer(os.path.join(args.output_dir, "tb_summary"))

    set_seed(args)

    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        finetuning_task=args.task,
    )
    tokenizer = BertTokenizer.from_pretrained( #TODO  why do we use a custom tokenizer??
        args.tokenizer_name_or_path,
    )
    model = BertForMultiDimensionRegression.from_pretrained(
        args.model_name_or_path,
        config=config
    )

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

    # Process Data
    processor = GoEmotionsProcessor(args, tokenizer, args.max_seq_len)
    processor.perform_full_preprocess()

    # Load dataset
    train_dataset = processor.get_dataset_by_mode('train')
    dev_dataset = processor.get_dataset_by_mode('dev')
    test_dataset = processor.get_dataset_by_mode('test')

    if dev_dataset is None:
        args.evaluate_test_during_training = True  # If there is no dev dataset, only use test dataset

    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer, train_dataset, MATRICS_FUNC, IDENTITY_FUNC, logger, tb_writer, dev_dataset, test_dataset)
        logger.info("Training Sum: global_step = {}, average loss = {}".format(global_step, tr_loss))

    results = {}
    if args.do_eval:
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True))
        )
        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = BertForMultiDimensionRegression.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, test_dataset, "test", logger, tb_writer, MATRICS_FUNC, IDENTITY_FUNC, global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        # TODO: log this evaluations to TensorBoard as well
        with open(output_eval_file, "w") as f_w:
            for key in sorted(results.keys()):
                f_w.write("{} = {}\n".format(key, str(results[key])))

        tb_writer.close()

        logger.info("***** Finished main() *****")

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--taxonomy", type=str, required=True,
                            help="Taxonomy (original, ekman, group)")

    cli_args = cli_parser.parse_args()

    main(cli_args)
