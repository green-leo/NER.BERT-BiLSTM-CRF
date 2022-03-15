from transformers import AutoModel, AutoTokenizer

bert = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)
bert.save_pretrained("./pretrained_model/bert_base_uncased")
tokenizer.save_pretrained("./pretrained_model/bert_base_uncased")