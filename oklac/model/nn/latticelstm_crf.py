#encoding:utf-8
import torch.nn as nn
from ..layers.embed_layer import Embed_Layer
from ..layers.crf import CRF
from ..layers.lattice import LatticeLSTM
import torch

class Lattice(nn.Module):
    def __init__(self,model_config,
                 embedding_dim,
                 out_numbers,
                 vocab_size,
                 embedding_weight,
                 dict_size, #word dict
                 pretrain_dict_embedding, #word_dict
                 device,
                 classifier = 0
):
        super(Lattice ,self).__init__()
        self.embedding = Embed_Layer(vocab_size = vocab_size,
                                     embedding_weight = embedding_weight,
                                     embedding_dim = embedding_dim,
                                     dropout_emb=model_config['dropout_emb'],
                                     training=True)
        classes = []
        self.hidden2tag = {}
        if (model_config['crf'] == False):
            self.add = 0
            for idx, out_number in enumerate(out_numbers):
                classes.append(out_number)
        else:
            self.add = 2
            for idx, out_number in enumerate(out_numbers):
                classes.append(out_number + 2)
        self.forward_lstm = LatticeLSTM(input_dim = embedding_dim,
                           hidden_dim = model_config['hidden_size']//2,
                           word_drop = 0.5,
                           word_alphabet_size = dict_size,
                           word_emb_dim  = 300,
                           pretrain_word_emb = pretrain_dict_embedding,
                           left2right = True,
                           fix_word_emb=True, 
                           device = device,
                           use_bias=True
                           )
        self.backward_lstm = LatticeLSTM(input_dim = embedding_dim,
                           hidden_dim = model_config['hidden_size']//2,
                           word_drop = 0.5,
                           word_alphabet_size = dict_size,
                           word_emb_dim  = 300,
                           pretrain_word_emb = pretrain_dict_embedding,
                           left2right = False,
                           fix_word_emb=True, 
                           device = device,
                           use_bias=True
                           )
        for out_number in out_numbers:
            linear = nn.Linear(in_features=model_config['hidden_size'], out_features= out_number+self.add)
            nn.init.xavier_uniform(linear.weight)
            self.hidden2tag[out_number+self.add] = linear
            self.hidden2tag[out_number+self.add] = self.hidden2tag[out_number+self.add].cuda()
            
        
        self.crf = CRF(device = device,tagset_size=out_numbers, have_crf = model_config['crf'])
        if (classifier!=0):
            self.classifier = torch.nn.Linear(768*3,classifier).cuda()
            
    def forward(self, inputs, length, out_number, masks = None, gaz = None):
        x = self.embedding(inputs) #batch,len,emb_size
        batch_size = inputs.size(0)
        sent_len = inputs.size(1)
        #word_embs = self.word_embeddings(word_inputs)

        # packed_words = pack_padded_sequence(word_embs, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.forward_lstm(x, gaz, hidden)

        backward_hidden = None
        backward_lstm_out, backward_hidden = self.backward_lstm(x, gaz, backward_hidden)
        lstm_out = torch.cat([lstm_out, backward_lstm_out], 2)
        # lstm_out, _ = pad_packed_sequence(lstm_out)
        #lstm_out = self.droplstm(lstm_out)
        lstm_out = self.hidden2tag[out_number+self.add](lstm_out)
        return lstm_out
        
        

