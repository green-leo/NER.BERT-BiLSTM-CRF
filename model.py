import config
import torch
import transformers
import torch.nn as nn
from torchcrf import CRF


class EntityModel(nn.Module):
    def __init__(self, num_tag, num_pos):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.num_pos = num_pos
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH,return_dict=False)
        self.bilstm= nn.LSTM(768, 1024 // 2, num_layers=1, bidirectional=True, batch_first=True)
        
        self.dropout_tag = nn.Dropout(0.3)
        self.dropout_pos = nn.Dropout(0.3)
        
        self.hidden2tag_tag = nn.Linear(1024, self.num_tag)
        self.hidden2tag_pos = nn.Linear(1024, self.num_pos)

        self.crf_tag = CRF(self.num_tag, batch_first=True)
        self.crf_pos = CRF(self.num_pos, batch_first=True)
    
    
    # return the loss only, not encode the tag
    def forward(self, ids, mask, token_type_ids, target_pos, target_tag):
        # Bert - BiLSTM
        x, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        h, _ = self.bilstm(x)

        # drop out
        o_tag = self.dropout_tag(h)
        o_pos = self.dropout_pos(h)

        # Hidden2Tag (Linear)
        tag = self.hidden2tag_tag(o_tag)
        pos = self.hidden2tag_pos(o_pos)

        # Create Mask for crf layer - the current torchcrf require boolen mask tensor
        mask = torch.where(mask==1, True, False)

        # compute log-likelihood loss through crf.forward(), 
        # and use 'token_mean' reduction method for more stable loss
        # by default, reduction = 'sum'
        loss_tag = - self.crf_tag(tag, target_tag, mask=mask, reduction='token_mean')
        loss_pos = - self.crf_pos(pos, target_pos, mask=mask, reduction='token_mean')
        loss = (loss_tag + loss_pos) / 2
        
        return loss


    # encode the tag, dont return loss
    def encode(self, ids, mask, token_type_ids, target_pos, target_tag):
        # Bert - BiLSTM
        x, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        h, _ = self.bilstm(x)

        # drop out
        o_tag = self.dropout_tag(h)
        o_pos = self.dropout_pos(h)

        # Hidden2Tag (Linear)
        tag = self.hidden2tag_tag(o_tag)
        pos = self.hidden2tag_pos(o_pos)

        # Create Mask for crf layer
        mask = torch.where(mask==1, True, False)
        
        # encode the tag by crf.decode
        tag = self.crf_tag.decode(tag, mask=mask)
        pos = self.crf_pos.decode(pos, mask=mask)

        return tag, pos