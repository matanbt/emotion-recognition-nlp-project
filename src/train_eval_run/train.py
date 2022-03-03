import os
import logging

import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from .evaluate import evaluate

logger = logging.getLogger(__name__)

def train(args,
          model: torch.nn.Module,
          model_args,
          tokenizer,
          tb_writer: SummaryWriter,
          train_dataset,
          dev_dataset=None,
          test_dataset=None):
    """
    Trains the given model

    args - technical arguments
    model - an instance of the model to train (inherits from BertPreTrainedModel)
    model_args - model specific arguments
    tb_writer - a TensordBoard writer instance to write evaluation results
    train_dataset, dev_dataset, test_dataset - the split dataset
    """

    logger.info("***** Preparing Training *****")

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(t_total * args.warmup_proportion),
        num_training_steps=t_total
    )

    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0

    args.evaluate_special_classifiers = False  # we don't allow classifiers eval while training

    model.zero_grad()  # TODO init the grads in an interesting way?
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            outputs = model(**batch)

            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    args.gradient_accumulation_steps >= len(train_dataloader) == (step + 1)
            ):
                # logs the training loss
                tb_writer.add_scalars("training", {'loss': loss.item()}, global_step=global_step)

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_test_during_training:
                        evaluate(args, model, model_args, tb_writer, test_dataset, "test", global_step)
                    else:
                        evaluate(args, model, model_args, tb_writer, dev_dataset, "dev", global_step)

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model_checkpoint(args, model, tokenizer, global_step, optimizer, scheduler)

            if 0 < args.max_steps < global_step:
                break

        if 0 < args.max_steps < global_step:
            break

    # Saves the final state of the model
    logger.info("Saving model final checkpoint...")
    save_model_checkpoint(args, model, tokenizer, global_step, optimizer, scheduler)

    logger.info("Performing final evaluation on dev-set")
    args.evaluate_special_classifiers = True  # allows classifiers eval in `compute_metrics` func
    save_training_to_csv(model, train_dataset, args)
    evaluate(args, model, model_args, tb_writer, dev_dataset, "dev", global_step)
    args.evaluate_special_classifiers = False

    logger.info("***** Finished Training *****")

    return global_step, tr_loss / global_step


def save_model_checkpoint(args,
                          model, tokenizer,
                          global_step,
                          optimizer, scheduler):
    """
        Saves the current model in a checkpoint dir
    """
    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to {}".format(output_dir))

    if args.save_optimizer:
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to {}".format(output_dir))


def save_training_to_csv(model, train_dataset, args, csv_f_name="trained_vad.csv"):
    """
        saves to VAD a numpy matrix of:
        | label | predicted VAD (V) |  predicted VAD (A) |  predicted VAD (D) |
    """

    eval_sampler = SequentialSampler(train_dataset)
    eval_dataloader = DataLoader(train_dataset, sampler=eval_sampler, batch_size=1)
    arr = np.zeros((len(train_dataset), 4))
    i = 0

    for batch in tqdm(eval_dataloader, desc="Saving Forward passes to CSV"):
        model.eval()

        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]

            arr[i] = [train_dataset['labels'][i][0]] + list(logits.squeeze(dim=0))

        i += 1

    path = os.path.join(args.summary_path, csv_f_name)
    np.savetxt(path, arr)
