#encoding:utf-8
import torch.nn as nn
from ..layers.embed_layer import Embed_Layer
from ..layers.crf import CRF

class CNN(nn.Module):
    def __init__(self,model_config,
                 embedding_dim,
                 vocab_size,
                 embedding_weight,
                 out_numbers,
                 device,
                 classifier = 0):
        super(CNN ,self).__init__()
        self.embedding = Embed_Layer(vocab_size = vocab_size,
                                     embedding_weight = embedding_weight,
                                     embedding_dim = embedding_dim,
                                     dropout_emb=model_config['dropout_emb'],
                                     training=True)
        classes = []
        self.add = 2 if (model_config['crf']) else 0
 
        
        self.hidden2tag = {}
        self.Conv1d = nn.Conv1d(in_channels = 80,out_channels = 80,kernel_size=3,stride = 1,padding = 0) #batch_size:256
        self.relu = nn.ReLU(inplace=True)
        for out_number in out_numbers:
            
            linear = nn.Linear(in_features=296, out_features= out_number+self.add)
            nn.init.xavier_uniform(linear.weight)
            self.hidden2tag[out_number+self.add] = linear
            self.hidden2tag[out_number+self.add] = self.hidden2tag[out_number+self.add].to(device)
        self.dropout = nn.Dropout(0.3)

        self.crf = CRF(device = device,tagset_size=out_numbers, have_crf = model_config['crf'])
        if (classifier!=0):
            self.classifier = torch.nn.Linear(768*3,classifier).cuda()

    def forward(self, inputs,length, out_number, masks = None, gaz = None):
        x = self.embedding(inputs)
        x = self.relu(self.Conv1d(x))


        h = self.Conv1d(x)

        h = self.dropout(h)
        h = self.hidden2tag[out_number+self.add](h)
        return h