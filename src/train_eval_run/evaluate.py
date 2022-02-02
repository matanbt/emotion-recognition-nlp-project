import os
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from src.train_eval_run.ModelConfig import ModelConfig

logger = logging.getLogger(__name__)


def evaluate(args, model, model_config: ModelConfig, eval_dataset, mode, global_step=None):

    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    if global_step != None:
        logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    targets = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        with torch.no_grad():
            outputs = model(**batch)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = model_config.func_on_model_output(logits)
            targets = batch[model_config.target_name].detach().cpu().numpy()  # TODO - why do we need detach here?
        else:
            preds = np.append(preds,  model_config.func_on_model_output(logits), axis=0)
            targets = np.append(targets, batch[model_config.target_name].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    results = {
        "loss": eval_loss
    }
    preds[preds > args.threshold] = 1
    preds[preds <= args.threshold] = 0
    result = model_config.compute_metrics(targets, preds)
    results.update(result)

    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir, "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))

    # logs the results to TensorBoard
    model_config.tb_writer.add_scalars(mode, results, global_step=global_step)

    return results
