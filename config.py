import transformers

DEVICE = 'cpu'
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
NUM_WORKER = 2
BASE_MODEL_PATH = "./pretrained_model/bert_base_uncased"
MODEL_PATH = "./model/best-model.bin"
TRAINING_FILE = "./data/ner_dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True
)
