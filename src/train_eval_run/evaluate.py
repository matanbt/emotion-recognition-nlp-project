import os
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def evaluate(args,
             model,
             model_args,
             tb_writer: SummaryWriter,
             eval_dataset,
             mode: str,
             global_step: int = None):
    """
    Evaluates the given model on the given dataset

    args - technical arguments
    model - an instance of the model to train (inherits from BertPreTrainedModel)
    model_args - model specific arguments
    tb_writer - a TensordBoard writer instance to write evaluation results
    eval_dataset - the evaluted dataset
    mode - the name of the dataset being evaluated (e.g. 'dev')
    """


    # Eval!
    if global_step is not None:
        logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))

    eval_loss, labels, targets, preds = only_eval(eval_dataset, model, model_args,
                                                  args.eval_batch_size, global_step=global_step)

    results = {
        "loss": eval_loss
    }

    if args.task == "one_label_classification":
        result = model_args.compute_metrics(labels, preds.argmax(axis=-1))

    elif args.task == "classification":  # classification case
        preds[preds > model_args.threshold] = 1
        preds[preds <= model_args.threshold] = 0
        result = model_args.compute_metrics(targets, preds)
    else:  # regression case
        result = model_args.compute_metrics(targets, preds, model_args.emotions_vads_lst,
                                            labels, mode, args)

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
    tb_writer.add_scalars(mode, results, global_step=global_step)

    return results



def only_eval(eval_dataset, model, model_args,
              eval_batch_size, global_step=None):
    """
    evaluate without analyzing
    """
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    targets = None
    labels = None

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        with torch.no_grad():
            outputs = model(**batch, global_step=global_step)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        curr_preds = model_args.func_on_model_output(logits)
        curr_labels = batch['one_hot_labels'].argmax(dim=-1)

        # Move Tensors to CPU (in case these are indeed tensors...)
        if isinstance(curr_preds, torch.Tensor):
            curr_preds = curr_preds.detach().cpu().numpy()
        if isinstance(batch[model_args.target_name], torch.Tensor):
            curr_targets = batch[model_args.target_name].detach().cpu().numpy()
        if isinstance(curr_labels, torch.Tensor):
            curr_labels = curr_labels.detach().cpu().numpy()


        if preds is None:
            preds = curr_preds
            targets = curr_targets
            labels = curr_labels
        else:
            preds = np.append(preds,  curr_preds, axis=0)
            targets = np.append(targets, curr_targets, axis=0)
            labels = np.append(labels, curr_labels, axis=0)

    eval_loss = eval_loss / nb_eval_steps

    return eval_loss, labels, targets, preds
