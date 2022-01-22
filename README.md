# Emotion Recognition : From discretion to dimention - NLP Final Project

## Directories and Files
- `notebooks`: directory for all notebooks we write during the research.
- `src`: The code, based on pytorch implementation of GoEmotions baseline.
- `data`: all the data and mappings we use in this project.
  - `emotions.txt`: list of emotions, each line number represent the index of the emotions in GoEmotions dataset.
  - `train.tsv`, `dev.tsv`, `test.tsv`: GoEmotions dataset split done by the paper. 

## Requirements:
- torch==1.4.0
- transformers==2.11.0
- datasets
- attrdict==2.0.1
- pandas
- tensorboard
