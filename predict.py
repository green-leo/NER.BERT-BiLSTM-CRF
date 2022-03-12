import joblib
import torch
import argparse

import config
import dataset
from model import EntityModel


def predict_sentence(model, sentence, enc_tag, enc_pos):
    sentence = sentence.split()
    test_dataset = dataset.EntityDataset(
        texts=[sentence], 
        pos=[[0] * len(sentence)], 
        tags=[[0] * len(sentence)]
    )
    
    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, pos, _ = model(**data)

        tag = enc_tag.inverse_transform(tag[0])
        pos = enc_pos.inverse_transform(pos[0])

    return tag, pos

if __name__ == "__main__":
    # make parser
    parser = argparse.ArgumentParser('Predict a string')
    parser.add_argument('sentence', metavar='String', nargs='+', type=str, 
                        default='This is the default sentence. Please enter a sentence for predicting.',
                        help='Enter a Sentence to predict')
    
    # parse args and get args.sentence
    args = parser.parse_args()
    sentence = args.sentence

    # join the arr -> string sentence (nargs='+', dont have to use "" when enter the string)
    sentence = ' '.join(sentence)
    tokenized_sentence = config.TOKENIZER.encode(sentence)

    # meta_data: enc_pos/enc_tag - POS/TAG label encoder
    meta_data = joblib.load("meta.bin")
    enc_pos = meta_data["enc_pos"]
    enc_tag = meta_data["enc_tag"]

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    # set up device, model
    device = torch.device(config.DEVICE)
    model = EntityModel(num_tag=num_tag, num_pos=num_pos)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    tag, pos = predict_sentence(model, sentence, enc_tag, enc_pos)

    print(sentence)
    print(tokenized_sentence)
    print(tag)
    print(pos)
    

# end if