# Emotion Recognition : From discretion to dimention - NLP Final Project

## Usage

 `$ python main.py --config {my-config-file.json}`

Where `config.json` is a configuration file located in `./config`.

- In order to reproduce the paper's experiments you may use the following table:

| Experiment      | Config                   |
| -----------     | -----------              |
| Classification model (baseline)             | `paper_exp/ge_baseline.json`   |
| MAE, 1NN             | `paper_exp/mae_nn.json`   |
| MAE, SVM        | `paper_exp/mae_svm.json`  |
| MSE, 1NN             | `paper_exp/mse_nn.json`        |
| MAE+CE, 1NN           | `paper_exp/mae_ce_nn.json`     
| MAE, VAD scaling 1, 1NN           | `paper_exp/mae_scale1_nn.json`     
| MAE, VAD scaling 2, 1NN       | `paper_exp/mae_scale2_nn.json` 
| MAE, VAD scaling 2, SVM           | `paper_exp/mae_scale2_svm.json`    



## Directories and Files
- `notebooks`: directory for all notebooks we write during the research.
- `src`: The code, based on pytorch implementation of GoEmotions baseline.
  - `data_processing`: package to preprocess all sorts of data. `data_loader.py` defines the essential data processor for our datasets.
  - `models`: package defines each model that was used.
  - `train_eval_run`: package that holds all the logic for training the evaluation, and also `run_model.py` which glues the latter two together.
  - `main.py`: CLI entry point to train / evaluate the models. 
- `data`: all the data and mappings we use in this project.
  - `goemotions`: 
    - `emotions.txt`: list of emotions, each line number represent the index of the emotions in GoEmotions dataset.
    - `train.tsv`, `dev.tsv`, `test.tsv`: GoEmotions dataset split done by the paper. 
  - ..

## Requirements:
- torch==1.4.0
- transformers==2.11.0
- datasets
- attrdict==2.0.1
- pandas
- tensorboard
- sklearn

## Acknowledgements
- We loosely based the core implementation of the model, on the PyTorch GoEmotions baseline implementation by @monologg, in [this repo](https://github.com/monologg/GoEmotions-pytorch).
- We found much use in the unified collection of emotions-related datasets, in [this repo](https://github.com/sarnthil/unify-emotion-datasets)
