#encoding:utf-8
import json
from torchtext import data
from ..utils.gazetteer import Gazetteer
from ..config.new_config import configs as config
from ..config.new_config import BASE_DIR
import numpy
from transformers import BertTokenizer
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import re
import torch
import random


class MyTokenizer(object):
    
    def __init__(self, pretrained_model_path=None, mask_entity=False):
        self.pretrained_model_path = pretrained_model_path or 'bert-base-chinese'
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)
        self.mask_entity = mask_entity
    
    def group_tokenize(self, item):
        etts = []
        sentence = item['text']
        info = item['info']
        if (len(info)<2):
            return [],[]
        results = []
        for head_idx in range(len(info)-1):
            tail_idx = head_idx + 1
            head_dict = {}
            head_dict['name'] = info[head_idx]['word']

            head_dict['pos'] = [info[head_idx]['start']]
            head_dict['pos'].append(info[head_idx]['end']+1)

            while tail_idx<len(info):
                tail_dict = {}
                tail_dict['name'] = info[tail_idx]['word']
                tail_dict['pos'] = [info[tail_idx]['start']]
                tail_dict['pos'].append(info[tail_idx]['end']+1)
                etts.append({'head':head_dict,'tail':tail_dict})
                etts.append({'tail':head_dict,'head':tail_dict})
                query1 = {'h':head_dict,'t':tail_dict,'text':sentence}
                query2 = {'h':tail_dict,'t':head_dict,'text':sentence}
                token,e1,e2 = self.tokenize(query1)
                results.append({'token':token,'e1':e1,'e2':e2})
                token,e1,e2 = self.tokenize(query2)
                results.append({'token':token,'e1':e1,'e2':e2})
                tail_idx+=1

        return results, etts

    def tokenize(self, item):
        sentence = item['text']
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            pos_min = pos_head
            pos_max = pos_tail
            rev = False

        sent0 = self.bert_tokenizer.tokenize(sentence[:pos_min[0]])
        ent0 = self.bert_tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
        sent1 = self.bert_tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
        ent1 = self.bert_tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
        sent2 = self.bert_tokenizer.tokenize(sentence[pos_max[1]:])

        if rev:
            if self.mask_entity:
                ent0 = ['[unused6]']
                ent1 = ['[unused5]']
            pos_tail = [len(sent0), len(sent0) + len(ent0)]
            pos_head = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
        else:
            if self.mask_entity:
                ent0 = ['[unused5]']
                ent1 = ['[unused6]']
            pos_head = [len(sent0), len(sent0) + len(ent0)]
            pos_tail = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
        tokens = sent0 + ent0 + sent1 + ent1 + sent2

        re_tokens = ['[CLS]']
        cur_pos = 0
        pos1 = [0, 0]
        pos2 = [0, 0]
        for token in tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                pos1[0] = len(re_tokens)
                re_tokens.append('[unused1]')
            if cur_pos == pos_tail[0]:
                pos2[0] = len(re_tokens)
                re_tokens.append('[unused2]')
            re_tokens.append(token)
            if cur_pos == pos_head[1] - 1:
                re_tokens.append('[unused3]')
                pos1[1] = len(re_tokens)
            if cur_pos == pos_tail[1] - 1:
                re_tokens.append('[unused4]')
                pos2[1] = len(re_tokens)
            cur_pos += 1
        re_tokens.append('[SEP]')
        return re_tokens[1:-1], pos1, pos2

class REDataset(data.Dataset):
    def __init__(self, data_file_path, tagset_path, pretrained_model_path=None, max_len=128, ner2re=False):
        if ner2re:
            self.data_file_path = data_file_path
            self.tagset_path = tagset_path
            self.pretrained_model_path = pretrained_model_path or 'bert-base-chinese'
            self.tokenizer = MyTokenizer(pretrained_model_path=self.pretrained_model_path)
            self.max_len = max_len
            self.tokens_list, self.e1_mask_list,self.e2_mask_list, self.tags, self.etts = trans(data_file_path,tokenizer=self.tokenizer,max_len=self.max_len)
            self.tag2idx = get_tag2idx(self.tagset_path)
        else:    
            self.data_file_path = data_file_path
            self.tagset_path = tagset_path
            self.pretrained_model_path = pretrained_model_path or 'bert-base-chinese'
            self.tokenizer = MyTokenizer(pretrained_model_path=self.pretrained_model_path)
            self.max_len = max_len
            self.tokens_list, self.e1_mask_list, self.e2_mask_list, self.tags = read_data(data_file_path, tokenizer=self.tokenizer, max_len=self.max_len)
            self.tag2idx = get_tag2idx(self.tagset_path)


    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_tokens = self.tokens_list[idx]
        sample_e1_mask = self.e1_mask_list[idx]
        sample_e2_mask = self.e2_mask_list[idx]
        sample_tag = self.tags[idx]
        encoded = self.tokenizer.bert_tokenizer.encode_plus(sample_tokens, max_length=self.max_len, pad_to_max_length=True)
        sample_token_ids = encoded['input_ids']
        sample_token_type_ids = encoded['token_type_ids']
        sample_attention_mask = encoded['attention_mask']
        try:
            sample_tag_id = torch.tensor(self.tag2idx[sample_tag])
        except:
            sample_tag_id = sample_tag
        sample = {
            'token_ids': torch.tensor(sample_token_ids),
            'token_type_ids': torch.tensor(sample_token_type_ids),
            'attention_mask': torch.tensor(sample_attention_mask),
            'e1_mask': torch.tensor(sample_e1_mask),
            'e2_mask': torch.tensor(sample_e2_mask),
            'tag_id': sample_tag_id
        }
        return sample

class TypDataset(data.Dataset):
    def __init__(self,
                 data_path,
                 x_var,y_var,
                 skip_header,
                 is_train_mode,
                 batch_size,
                 max_sentence_length = None,
                 default_token = False,
                 pretrained = 'bert'
                 ):
        if (default_token):

            if (pretrained=='roberta'):
                t = BertTokenizer.from_pretrained(BASE_DIR+'/output/checkpoints/roberta') 
          
            else:
                t = BertTokenizer.from_pretrained('bert-base-chinese') 
            tokenizer = lambda x: t.convert_tokens_to_ids(x.split())
        else:
            tokenizer = lambda x: [int(c) for c in x.split()]
        label_tok = lambda x: [int(c) for c in x.split()]
        padding = lambda x: [(0 if idx>=len(x) else x[idx]) for idx in range(max_sentence_length)]
        self.data =  {'lcontexts':[],'rcontexts':[],'labels':[],'mentions':[],'left_masks':[],'right_masks':[],'mention_masks':[]}
        with open(data_path,'r') as fr:
            for i,line in enumerate(fr):
                if i == 0 and skip_header:
                    continue
                df = json.loads(line)
                sentence = df[x_var] 
                label = df[y_var]

                lb = [int((i+1) in label) for i in range(10)]

                span = df['span']
                whole = len(sentence.split())

                if (whole<=(span[1]) or span[0]==0):
                    continue
                      #  spans.append(df['span'])
                left_context = (sentence.split())[0:span[0]]
                left_context = t.encode_plus(left_context, max_length=max_sentence_length, pad_to_max_length=True)

                left_mask = left_context['attention_mask']
                left_context = left_context['input_ids']
                right_context = (sentence.split())[span[1]:]

                right_context = t.encode_plus(right_context, max_length=max_sentence_length, pad_to_max_length=True)
 
                right_mask = right_context['attention_mask']
                right_context = right_context['input_ids']
                mention = (sentence.split())[span[0]:span[1]]

                mention = t.encode_plus(mention, max_length=max_sentence_length, pad_to_max_length=True)
                mention_mask = mention['attention_mask']
                mention = mention['input_ids']
                self.data['lcontexts'].append(left_context)
                self.data['rcontexts'].append(right_context)
                self.data['labels'].append(lb)
                self.data['mentions'].append(mention)
                self.data['left_masks'].append(left_mask)
                self.data['right_masks'].append(right_mask)
                self.data['mention_masks'].append(mention_mask)
    def __len__(self):
        return len(self.data['mentions'])
    def __getitem__(self,index):
        return (self.data['lcontexts'][index],self.data['rcontexts'][index],self.data['labels'][index],self.data['mentions'][index],self.data['left_masks'][index],self.data['right_masks'][index],self.data['mention_masks'][index])




class MultiDataset(data.Dataset):
    def __init__(self,
                num_threads,
                 data_paths,
                 x_var,y_var,
                 skip_header,
                 is_train_mode,
                 batch_size,
                 max_sentence_length = None,
                 default_token = False,
                 gaz = None,
                 vocab = None,
                 rev_vocab = None,
                 pretrained = 'bert'
                 ):
        self.data_group = []
        self.data = {'sents':[],'labels':[],'gaz_lists':[],'lengths':[],'sources':[]}
        if (default_token):

            if (pretrained=='roberta'):
                t = BertTokenizer.from_pretrained(BASE_DIR+'/output/checkpoints/roberta') 
          
            else:
                t = BertTokenizer.from_pretrained('bert-base-chinese') 
            tokenizer = lambda x: t.convert_tokens_to_ids(x.split())
        else:
            tokenizer = lambda x: [int(c) for c in x.split()]
        label_tok = lambda x: [int(c) for c in x.split()]
        padding = lambda x: [(0 if idx>=len(x) else x[idx]) for idx in range(max_sentence_length)]
        for idx,data_path in enumerate(data_paths):
            self.data_group.append({})
            sents = []
            labels = []
            gaz_lists = []
            lengths = []
            sources = []

            mention_masks = []
            with open(data_path,'r') as fr:
                for i,line in enumerate(fr):
                    if i == 0 and skip_header:
                        continue
                    df = json.loads(line)
                    sentence = df[x_var] 
                    label = df[y_var]
                    sources.append(idx)

                    try:
                        sents.append(padding(tokenizer(sentence)))
                        labels.append(padding(label_tok(label)))
                        lengths.append(len(sentence.split()) if (len(sentence.split())<max_sentence_length) else max_sentence_length)
                    except:
                        pass
            self.data_group[idx]['sents']=sents
            self.data_group[idx]['labels']=labels
            self.data_group[idx]['gaz_lists']=gaz_lists
            self.data_group[idx]['lengths']=lengths
            self.data_group[idx]['sources']=sources
        #    self.data_group[idx]['spans']=spans
        #    self.data_group[idx]['mentions']=mentions
         #   self.data_group[idx]['left_masks']=left_masks
        #    self.data_group[idx]['right_masks']=right_masks
       #     self.data_group[idx]['mention_masks']=mention_masks
        offset = 0
        batch_sz = batch_size*num_threads
        while True:
            endoffile = True
            for group in self.data_group:
                for k in group.keys():
                    d = group[k]

                    if (offset+batch_sz<len(d)):
                        self.data[k].extend(d[offset:offset+batch_sz])
                        endoffile = False
            offset+=batch_sz
            if endoffile:
                break

        if gaz!=None:
            for i,sent in enumerate(self.data['sents']):
                gaz_pack = []
                sent_len = self.data['lengths'][i]          
                for idx in range(sent_len):
                    match_out = gaz.enumerateMatchList(sent[idx:sent_len])
                    if (len(match_out[0])==0):
                        match_out = []
                    gaz_pack.append(match_out)
                while (len(gaz_pack)<max_sentence_length):
                    gaz_pack.append([])
                self.data['gaz_lists'].append(gaz_pack)
        else:
            self.data['gaz_lists']= [0]*len(self.data['sents'])
    def __len__(self):
        return len(self.data['sents'])
    def __getitem__(self,index):
        return (self.data['sents'][index],self.data['gaz_lists'][index],self.data['labels'][index],self.data['lengths'][index],self.data['sources'][index])



def convert_pos_to_mask(e_pos, max_len=128):
    e_pos_mask = [0] * max_len
    for i in range(e_pos[0], e_pos[1]):
        e_pos_mask[i] = 1
    return e_pos_mask

def trans(input_file,tokenizer=None,max_len=128):
    tokens_list = []
    e1_mask_list = []
    e2_mask_list = []
    tags = []

    with open(input_file, 'r', encoding='utf-8') as f_in:
        liness = 0
        etts = []
        for line in tqdm(f_in):
            liness+=1
            line = line.strip()
            item = json.loads(line)
            if tokenizer is None:
                tokenizer = MyTokenizer()
            results, _etts = tokenizer.group_tokenize(item)
            etts.extend(_etts)
            for result in results:

                tokens = result['token']
                pos_e1 = result['e1']
                pos_e2 = result['e2']
                if pos_e1[0] < max_len - 1 and pos_e1[1] < max_len and \
                        pos_e2[0] < max_len - 1 and pos_e2[1] < max_len:
                    tokens_list.append(tokens)
                    e1_mask = convert_pos_to_mask(pos_e1, max_len)
                    e2_mask = convert_pos_to_mask(pos_e2, max_len)
                    e1_mask_list.append(e1_mask)
                    e2_mask_list.append(e2_mask)
                    tags.append(item['text'])
    return tokens_list, e1_mask_list, e2_mask_list,tags, etts

def read_data(input_file, tokenizer=None, max_len=128):
    tokens_list = []
    e1_mask_list = []
    e2_mask_list = []
    tags = []
    with open(input_file, 'r', encoding='utf-8') as f_in:
        liness = 0
        for line in tqdm(f_in):
            liness+=1
            #if (liness>180000):
            #    break
            line = line.strip()
            item = json.loads(line)
            if tokenizer is None:
                tokenizer = MyTokenizer()
            tokens, pos_e1, pos_e2 = tokenizer.tokenize(item)
            if pos_e1[0] < max_len - 1 and pos_e1[1] < max_len and \
                    pos_e2[0] < max_len - 1 and pos_e2[1] < max_len:
                tokens_list.append(tokens)
                e1_mask = convert_pos_to_mask(pos_e1, max_len)
                e2_mask = convert_pos_to_mask(pos_e2, max_len)
                e1_mask_list.append(e1_mask)
                e2_mask_list.append(e2_mask)
                tag = item['relation']
                tags.append(tag)
    return tokens_list, e1_mask_list, e2_mask_list, tags


def save_tagset(tagset, output_file):
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(tagset))


def get_tag2idx(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((tag, idx) for idx, tag in enumerate(tagset))


def get_idx2tag(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((idx, tag) for idx, tag in enumerate(tagset))


