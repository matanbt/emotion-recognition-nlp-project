import json
import logging
import os
import glob

import torch
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    BertConfig,
    BertTokenizer
)

from .utils import (
    set_seed,
)

from .train import train
from .evaluate import evaluate

logger = logging.getLogger(__name__)


def run(args, model_args, tb_writer: SummaryWriter):
    """
    args - the full configuration for the run
    model_args - model specific arguments, includes model class, data processor, metrics functions etc.
    tv_writer - an instance to write to TensorBoard
    """

    logger.info("***** Starting run_model.run() *****")

    set_seed(args)

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    label_list = model_args.data_processor_class.get_labels_list(args)

    # Initiate all needed model's component
    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        finetuning_task=args.task,
        id2label={str(i): label for i, label in enumerate(label_list)},
        label2id={label: i for i, label in enumerate(label_list)}
    )
    tokenizer = BertTokenizer.from_pretrained(  # TODO - why do we use a custom tokenizer??
        args.tokenizer_name_or_path,
    )
    model = model_args.model_class.from_pretrained(
        args.model_name_or_path,
        config=config,
        args=args,
        **vars(model_args)
    )

    model.to(args.device)

    # Process Data
    processor = model_args.data_processor_class(args, tokenizer, args.max_seq_len,
                                                vad_mapper_name=model_args.vad_mapper_name)
    if model_args.vad_mapper_name is not None:  # regression case
        model_args.emotions_vads_lst = processor.get_emotions_vads_lst()
        # informs the model instance of the VADs
        model.emotions_vads_lst = torch.tensor(model_args.emotions_vads_lst, device=args.device)

    processor.perform_full_preprocess()

    # Load dataset
    train_dataset = processor.get_dataset_by_mode('train')
    dev_dataset = processor.get_dataset_by_mode('dev')
    test_dataset = processor.get_dataset_by_mode('test')

    if dev_dataset is None:
        args.evaluate_test_during_training = True  # If there is no dev dataset, only use test dataset

    if args.do_train:
        global_step, tr_loss = train(args, model, model_args, tokenizer, tb_writer,
                                     train_dataset, dev_dataset, test_dataset)
        logger.info("Training Sum: global_step = {}, average loss = {}".format(global_step, tr_loss))

    results = {}
    if args.do_eval:
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin",
                                                         recursive=True))
        )
        checkpoints.sort(key=lambda cpt: int(cpt.split('-')[-1]))  # sort by global-step

        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = int(checkpoint.split("-")[-1])
            model = model_args.model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, model_args, tb_writer, test_dataset, "test", global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            for key in sorted(results.keys()):
                f_w.write("{} = {}\n".format(key, str(results[key])))

        logger.info("***** Finished main() *****")
