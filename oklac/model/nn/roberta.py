import torch
import torch.autograd as autograd
import torch.nn as nn
from itertools import chain
from transformers import BertModel,BertConfig
from ..layers.crf import CRF
from ..layers.bilstm import BILSTM
import time

class RoBERTa_LSTM(nn.Module):

    def __init__(self, out_numbers, model_config, device,bert_route='bert-base-chinese',  num_layers=1, classifier = 0):
        super(RoBERTa_LSTM, self).__init__()
        classes = []
        if (model_config['crf'] == False):
            self.add = 0
            for idx, out_number in enumerate(out_numbers):
                classes.append(out_number)
        else:
            self.add = 2
            for idx, out_number in enumerate(out_numbers):
                classes.append(out_number + 2)
        self.bert_encoder = BertModel.from_pretrained('../../../roberta')



        # also input dim of LSTM
        self.bert_out_dim = self.bert_encoder.config.hidden_size
        # LSTM layer
        self.lstm = BILSTM(input_size = self.bert_out_dim,
                           hidden_size= model_config['hidden_size'],
                           num_layer  = model_config['num_layer'],
                           bi_tag     = model_config['bi_tag'],
                           dropout_p  = model_config['dropout_p'],
                           out_numbers= classes,
                           device = device)

        self.crf = CRF(device = device,tagset_size=out_numbers, have_crf = model_config['crf'])
        if (classifier!=0):
            self.classifier = torch.nn.Linear(self.bert_out_dim*3,classifier).cuda()


    def forward(self, sent, length, out_number, masks = None, gaz =None):
        # sent,tags,masks: (batch * seq_length)
       # print("sent:",sent.device,"mask:",masks.device)
        bert_out = self.bert_encoder(sent, masks)[0]
        # bert_out: (batch * seq_length * bert_hidden=768)
        bert_out = self.lstm(bert_out,length,out_number+self.add)

        
        return bert_out
