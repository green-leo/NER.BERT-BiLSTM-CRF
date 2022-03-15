# NER.BERT-BiLSTM-CRF
 
A basic BERT-BiLSTM-CRF Model using in NER problem.

The code is modified from [abhishekkrthakur/bert-entity-extraction](https://github.com/abhishekkrthakur/bert-entity-extraction) repository that only uses BERT model to attack the same problem and dataset.
The dataset is in kaggle followed by [link](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus). 


### Requiments
> Transformer
>
> Torch


### Usage
1. Download pretrained-bert-model
```bash
# change pretrained_model_name you want to use in the file
python download_pretrained_model.py
```  

2. Custom your parameters in **config.py** file.
```bash
# The dir-path of your downloaded pretrained model 
# or the pretrained model name if you dont want to download it
BASE_MODEL_PATH = "./pretrained_model/bert_base_uncased"
```

2. Train your model
```bash
python train.py
```  

3. Predict a sentences through
```bash
# python predict.py Your_Sentence
python predict.py I am going to Paris next summer
```
