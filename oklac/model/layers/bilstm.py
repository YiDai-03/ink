#encoding:utf-8
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence
from .model_utils import prepare_pack_padded_sequence

class BILSTM(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layer,
                 input_size,
                 dropout_p,
                 bi_tag,
                 out_numbers,
                 device):

        super(BILSTM,self).__init__()
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        
        self.hidden2tag = {}
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layer,
                            batch_first = True,
                            dropout = dropout_p,
                            bidirectional = bi_tag)

        bi_num = 2 if bi_tag else 1
        for out_number in out_numbers:

            linear = nn.Linear(in_features=hidden_size * bi_num, out_features= out_number)
            nn.init.xavier_uniform(linear.weight)
            self.hidden2tag[out_number] = linear
            self.hidden2tag[out_number] = self.hidden2tag[out_number].to(device)
        #self.linearc = nn.Linear(in_features=hidden_size * bi_num, out_features= 5)
        #self.linearn = nn.Linear(in_features=hidden_size * bi_num, out_features= 10)
        #nn.init.xavier_uniform(self.linearc.weight)
        #nn.init.xavier_uniform(self.linearn.weight)
        #print(self.linearc)
        #self.linearc.to(device)
        #self.linearn.to(device)

            

    def forward(self,inputs,length, out_number):


        inputs, length, desorted_indice = prepare_pack_padded_sequence(inputs, length)
        embeddings_packed = pack_padded_sequence(inputs, length, batch_first=True)
        output, _ = self.lstm(embeddings_packed)

        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output[desorted_indice]
        output = F.dropout(output, p=self.dropout_p, training=self.training)
        output = F.tanh(output)
        logit = self.hidden2tag[out_number](output)
      #  if (out_number ==5 ):
      #      logit = self.linearc(output)
      #  else:
       #     logit = self.linearn(output)
        return logit