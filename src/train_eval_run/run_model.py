import json
import logging
import os
import glob

import numpy as np
import torch
from attrdict import AttrDict

from transformers import (
    BertConfig,
    BertTokenizer
)

from .utils import (
    init_logger,
    init_tensorboard_writer,
    set_seed,
    compute_metrics_classification
)

from .train import train
from .evaluate import evaluate

logger = logging.getLogger(__name__)

# TODO move the following configurations to main.py
SIGMOID_FUNC = lambda x: 1 / (1 + np.exp(-x.detach().cpu().numpy()))
MATRICS_FUNC = compute_metrics_classification
TARGET_NAME = "one_hot_labels"


def run(cli_args, data_processor_class, model_class):
    logger.info("***** Starting main() *****")

    # --- Initializations ---
    # Read from config file and make args
    config_filename = "{}".format(cli_args.config)
    with open(os.path.join("config", config_filename)) as f:
        args = AttrDict(json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))

    # args.output_dir = ... # TODO add time-stamp to path

    init_logger()
    args.tb_writer= init_tensorboard_writer(os.path.join(args.output_dir, "tb_summary"))

    set_seed(args)

    # TODO replace this line somehow to be general
    label_list = []
    with open(os.path.join(args.data_dir, args.label_file), "r", encoding="utf-8") as f:
        for line in f:
            label_list.append(line.rstrip())
    # TODO ------

    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        finetuning_task=args.task,
        id2label={str(i): label for i, label in enumerate(label_list)},
        label2id={label: i for i, label in enumerate(label_list)}
    )
    tokenizer = BertTokenizer.from_pretrained( #TODO  why do we use a custom tokenizer??
        args.tokenizer_name_or_path,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        config=config
    )

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

    # Add more arguments
    args.compute_metrics = MATRICS_FUNC
    args.target_name = TARGET_NAME
    args.func_on_model_output = SIGMOID_FUNC

    # Process Data
    processor = data_processor_class(args, tokenizer, args.max_seq_len)
    processor.perform_full_preprocess()

    # Load dataset
    train_dataset = processor.get_dataset_by_mode('train')
    dev_dataset = processor.get_dataset_by_mode('dev')
    test_dataset = processor.get_dataset_by_mode('test')

    if dev_dataset is None:
        args.evaluate_test_during_training = True  # If there is no dev dataset, only use test dataset

    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer, train_dataset,
                                     dev_dataset, test_dataset)
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
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, test_dataset, "test", global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        # TODO: log this evaluations to TensorBoard as well
        with open(output_eval_file, "w") as f_w:
            for key in sorted(results.keys()):
                f_w.write("{} = {}\n".format(key, str(results[key])))

        args.tb_writer.close()

        logger.info("***** Finished main() *****")


