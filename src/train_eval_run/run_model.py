import logging
import os
import glob

import torch

from transformers import (
    BertConfig,
    BertTokenizer
)

from .utils import (
    set_seed,
)

from .train import train
from .evaluate import evaluate
from src.train_eval_run.ModelConfig import ModelConfig

logger = logging.getLogger(__name__)


def run(args, data_processor_class, model_config: ModelConfig):

    logger.info("***** Starting run_model.run *****")

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
    tokenizer = BertTokenizer.from_pretrained(  # TODO - why do we use a custom tokenizer??
        args.tokenizer_name_or_path,
    )
    model = model_config.model_class.from_pretrained(
        args.model_name_or_path,
        model_config.num_dim,
        config=config
    )

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

    # Process Data
    processor = data_processor_class(args, tokenizer, args.max_seq_len, model_config.vad_mapper_name)
    processor.perform_full_preprocess()

    # Load dataset
    train_dataset = processor.get_dataset_by_mode('train')
    dev_dataset = processor.get_dataset_by_mode('dev')
    test_dataset = processor.get_dataset_by_mode('test')

    if dev_dataset is None:
        args.evaluate_test_during_training = True  # If there is no dev dataset, only use test dataset

    if args.do_train:
        global_step, tr_loss = train(args, model, model_config, tokenizer,
                                     train_dataset, dev_dataset, test_dataset)
        logger.info("Training Sum: global_step = {}, average loss = {}".format(global_step, tr_loss))

    results = {}
    if args.do_eval:
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin",
                                                         recursive=True))
        )
        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = model_config.model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, model_config, test_dataset, "test", global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        # TODO: log this evaluations to TensorBoard as well
        with open(output_eval_file, "w") as f_w:
            for key in sorted(results.keys()):
                f_w.write("{} = {}\n".format(key, str(results[key])))

        model_config.tb_writer.close()

        logger.info("***** Finished main() *****")
