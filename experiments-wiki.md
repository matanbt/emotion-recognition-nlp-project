# Experiments - Making it scalable!

## Motivation
We are going to perform **alot** of experiments, analyzing each one manually 
could be very frastrating, so utilizing `TensorBoard` and logging smartly should really help us.

## What's here?
Here we'll agree on conventions and inserts useful snippets.

## TensorBoard Writer
- We use `torch.utils.tensorboard` well documented here: https://pytorch.org/docs/stable/tensorboard.html.
- TensorBoard summaries wil be saved in `results/name_of_model_we_trained/tb_summary` for each model.
- running `tensorboard` command from the `results` dir, will provide web-UI  visualizing all the results.