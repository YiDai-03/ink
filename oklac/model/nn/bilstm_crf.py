#encoding:utf-8
import torch.nn as nn
import torch
from ..layers.embed_layer import Embed_Layer
from ..layers.crf import CRF
from ..layers.bilstm import BILSTM

class BiLSTM(nn.Module):
    def __init__(self,model_config,
                 embedding_dim,
                 vocab_size,
                 embedding_weight,
                 out_numbers,
                 device,
                 classifier = 0):
        super(BiLSTM ,self).__init__()
        self.embedding = Embed_Layer(vocab_size = vocab_size,
                                     embedding_weight = embedding_weight,
                                     embedding_dim = embedding_dim,
                                     dropout_emb=model_config['dropout_emb'],
                                     training=True)
        
        classes = []
        self.device = device
        if (model_config['crf'] == False):
            self.add = 0
            for idx, out_number in enumerate(out_numbers):
                classes.append(out_number)
        else:
            self.add = 2
            for idx, out_number in enumerate(out_numbers):
                classes.append(out_number + 2)
        self.lstm = BILSTM(input_size = embedding_dim,
                           hidden_size= model_config['hidden_size'],
                           num_layer  = model_config['num_layer'],
                           bi_tag     = model_config['bi_tag'],
                           dropout_p  = model_config['dropout_p'],
                           out_numbers = classes,
                           device = device)
        self.crf = CRF(device = device,tagset_size=out_numbers, have_crf = model_config['crf'])
        if (classifier!=0):
            self.classifier = torch.nn.Linear(768*3,classifier).cuda()

    def forward(self, inputs,length, out_number, masks = None, gaz = None):

        x = self.embedding(inputs)
        x = self.lstm(x,length, out_number+self.add)
        return x

