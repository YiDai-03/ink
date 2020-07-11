import argparse
import torch
import numpy as np
import warnings
import os
from torch import optim
from itertools import chain
from .test.predicter import Predicter
from .test.predict_utils import test_write
from .train.metrics import F1_score
from .train.trainer import Trainer
from .train.train_utils import restore_checkpoint,model_device
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from .io.dataset import MultiDataset, REDataset, get_idx2tag, TypDataset
from .io.data_transformer import DataTransformer
from .io.downloader import Downloader

from .utils.utils import seed_everything, list2dic
from .config.new_config import configs as config

from .config.new_config import RE as hparams


from .model.nn.cnn import CNN
from .model.nn.bilstm_crf import BiLSTM
from .model.nn.latticelstm_crf import Lattice
from .model.nn.bert import BERT_LSTM
from .model.nn.roberta import RoBERTa_LSTM
from .model.nn.relation import SentenceRE
from .callback.modelcheckpoint import ModelCheckpoint
from tqdm import tqdm

import torch.distributed as dist
import zipfile
import abc
from transformers import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

def synchronize():

    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

class Component(abc.ABC):
    def __init__(self, config, local_rank=0):   
        super().__init__()
        self.config = config
        self.local_rank = local_rank
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        self.is_distributed = (self.num_gpus>1)
        if self.is_distributed:
            torch.cuda.set_device(self.local_rank)
            self.device = 'cuda:%d'%(self.local_rank)
            torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
            synchronize()
        else:
            self.device = 'cuda:%d' % config['n_gpus'][0] if len(config['n_gpus']) else 'cpu'
            if (len(config['n_gpus'])):
                torch.cuda.set_device(config['n_gpus'][0])
    @abc.abstractmethod
    def train():
        pass
    @abc.abstractmethod
    def predict():
        pass

class RelationExtractor(Component):
    def __init__(self, config, local_rank=0):
        super().__init__(config, local_rank)
        if self.local_rank == 0:
            self.dl = Downloader()
            if (not os.path.exists(hparams['dataset']+'/REdataset')):
                self.dl.download("REdataset",hparams['dataset']+'/REdataset.zip')
                f = zipfile.ZipFile(hparams['dataset']+'/REdataset.zip','r')
                for file in f.namelist():
                    print("extracting:",str(file),'...')
                    f.extract(file,hparams['dataset']+'/REdataset')
                print("extracted.")
   #     self.logger = init_logger(log_name=self.model_config['name'], log_dir=config['log_dir'])
    def train(self):
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        RE_train_dataset =  REDataset(hparams['train_file'], tagset_path=hparams['tagset_file'],
                                      pretrained_model_path=None,
                                      max_len=hparams['max_len'])
        RE_val_dataset =  REDataset(hparams['validation_file'], tagset_path=hparams['tagset_file'],
                                      pretrained_model_path=None,
                                      max_len=hparams['max_len'])
        self.REModel = SentenceRE(embedding_dim=768,dropout=0.1,tagset_size=49)
        self.REoptimizer = torch.optim.Adam(self.REModel.parameters(), lr=1e-5, weight_decay=0)
        REsampler = DistributedSampler(RE_train_dataset,shuffle=False) if self.is_distributed else None
        self.RE_train_loader = DataLoader(RE_train_dataset, batch_size=32, shuffle=False,sampler=REsampler)
        

        self.RE_val_loader = DataLoader(RE_val_dataset, batch_size=32, shuffle=False)              
        self.REModel = self.REModel.cuda()
        if self.is_distributed:
            self.REModel = torch.nn.parallel.DistributedDataParallel(
            self.REModel, device_ids=[self.local_rank], output_device=self.local_rank,
        find_unused_parameters=True
            )  
        else:
            pass

        if (self.REModel):
            epoch_offset = 0
            best_val = float("inf")
            for epoch in range(epoch_offset, 20):
                if self.local_rank==0:
                    print("Epoch: {}".format(epoch))
                self.REModel.train()
                idx=0
                for i_batch, sample_batched in enumerate(tqdm(self.RE_train_loader, desc='Training')):
                    idx+=32

                    token_ids = sample_batched['token_ids'].cuda()
                    token_type_ids = sample_batched['token_type_ids'].cuda()
                    attention_mask = sample_batched['attention_mask'].cuda()
                    e1_mask = sample_batched['e1_mask'].cuda()
                    e2_mask = sample_batched['e2_mask'].cuda()
                    tag_ids = sample_batched['tag_id'].cuda()
                    self.REModel.zero_grad()
                    logits = self.REModel(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask)
                    pred_tag_ids = logits.argmax(1)
                    ids = pred_tag_ids.cpu().numpy().tolist()


                    loss = self.criterion(logits, tag_ids)
                    loss.backward()
                    self.REoptimizer.step()
                    self.REoptimizer.zero_grad()
                if (self.local_rank==0):
                    with torch.no_grad():
                        val_loss = float(0.0)
                        for i_batch, sample_batched in enumerate(tqdm(self.RE_val_loader)):
                            token_ids = sample_batched['token_ids'].cuda()
                            token_type_ids = sample_batched['token_type_ids'].cuda()
                            attention_mask = sample_batched['attention_mask'].cuda()
                            e1_mask = sample_batched['e1_mask'].cuda()
                            e2_mask = sample_batched['e2_mask'].cuda()
                            tag_ids = sample_batched['tag_id'].cuda()
                            logits = self.REModel(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask)
                            pred_tag_ids = logits.argmax(1)
                            ids = pred_tag_ids.cpu().numpy().tolist()


                            val_loss += self.criterion(logits, tag_ids)
                        if (float(val_loss) < float(best_val)):
                            if (self.local_rank==0):
                                print("val_loss:",float(best_val),'->',float(val_loss))
                            best_val = val_loss
                            if self.is_distributed:
                                state = { 'state_dict': self.REModel.module.state_dict(),
                                    'optimizer': self.REoptimizer.state_dict()}
                            else:
                                state = { 'state_dict': self.REModel.state_dict(),
                                    'optimizer': self.REoptimizer.state_dict()}     
                            torch.save(state,hparams['checkpoint'])
                if self.is_distributed:
                    dist.barrier()

    def predict(self):
        RE_test_dataset =  REDataset(hparams['ner_source'], tagset_path=hparams['tagset_file'],
                                      pretrained_model_path=None,
                                      max_len=hparams['max_len'],ner2re = True)
        etts = RE_test_dataset.etts



        if self.local_rank==0:
            self.REModel = SentenceRE(embedding_dim=768,dropout=0.1,tagset_size=49)
            self.REoptimizer = torch.optim.Adam(self.REModel.parameters(), lr=1e-5, weight_decay=0)
            self.RE_test_loader = DataLoader(RE_test_dataset, batch_size=32, shuffle=False)
            self.REModel = self.REModel.cuda()
            checkpoint = torch.load(hparams['checkpoint'])
            self.REModel.load_state_dict(checkpoint['state_dict'])
            self.REoptimizer.load_state_dict(checkpoint['optimizer'])
            epoch_offset = 0
            idx2tag =  get_idx2tag(hparams['tagset_file'])
            with torch.no_grad():
                output_file = open(hparams['result'],'w')
                idx=0
                cur_sent = ''
                info = {'sent':'','info':[]}
                for i_batch, sample_batched in enumerate(tqdm(self.RE_test_loader, desc='Testing')):
                    token_ids = sample_batched['token_ids'].cuda()
                    token_type_ids = sample_batched['token_type_ids'].cuda()
                    attention_mask = sample_batched['attention_mask'].cuda()
                    e1_mask = sample_batched['e1_mask'].cuda()
                    e2_mask = sample_batched['e2_mask'].cuda()
                    tag_ids = sample_batched['tag_id']
                    logits = self.REModel(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask)
                    pred_tag_ids = logits.argmax(1)
                    ids = pred_tag_ids.cpu().numpy().tolist()
                    for rel_id, source in zip(ids,tag_ids):
                        if (idx2tag[rel_id]=='unknown'):
                            idx+=1
                            continue
                        if (source == cur_sent):
                            info['info'].append(etts[idx]['head']['name']+'->'+etts[idx]['tail']['name']+':'+idx2tag[rel_id])
                        else:
                            cur_sent = source
                            print(str(info),file=output_file)
                            info = {'sent':source,'info':[etts[idx]['head']['name']+'->'+etts[idx]['tail']['name']+':'+idx2tag[rel_id]]}
                        idx+=1

class Labeler(Component):

    def __init__(self, config, local_rank=0):
        super().__init__(config, local_rank)
        self.model_config = config['model']
   
        seed_everything(seed = config['seed'])

        self.default_tk = True if ('bert' in self.model_config['name']) else False #tokenize


        if self.local_rank == 0:
            self.dl = Downloader()

            if ((not os.path.exists(config['embedding_dict_path']))):
                self.dl.download(url='embedding',file_path=config['embedding_dict_path'])
            if (('bert' in config['model']['name']) and (not os.path.exists(config['pretrained_path'][:-4]))):
                self.dl.download(url='roberta',file_path=config['pretrained_path'])
                f = zipfile.ZipFile(config['pretrained_path'],'r')
                for file in f.namelist():
                    print("extracting:",str(file),'...')
                    f.extract(file,config['pretrained_path'][:-4])
                print("extracted.")
            if (os.path.exists(config['dataset_path']+'/raw.zip')):
                pass
            else:
                self.dl.download(url='dataset',file_path = config['dataset_path']+'/raw.zip')
                f = zipfile.ZipFile(config['dataset_path']+'/raw.zip','r')
                for file in f.namelist():
                    print("extracting:",str(file),'...')
                    f.extract(file,config['dataset_path'])
                print('extracted.')
        if self.is_distributed:
            dist.barrier()

    def train(self):
        bs = self.config['batch_size']
        if ('lattice' in self.model_config['name']):
            self.config['batch_size']=1
        word_dic = True if ('lattice' in self.config['model']['name']) else False
        out_numbers = [] 
        self.data_transformer = DataTransformer(
                                       is_train_mode = True,
                                       all_data_path = self.config['all_data_path'],
                                       vocab_path    = self.config['vocab_path'],
                                       rev_vocab_path    = self.config['rev_vocab_path'],
                                       max_features  = self.config['max_features'],
                                       label_to_id   = None,
                                       train_file    = None,
                                       valid_file      = None,
                                       valid_size      = self.config['valid_size'],
                                       min_freq      = self.config['min_freq'],
                                       seed          = self.config['seed'],
                                       default_token = self.default_tk,
                                       word_dic      = word_dic)
        self.data_transformer.build_vocab()
        embedding_weight, words_embedding, gaz_tree = self.data_transformer.build_embedding_matrix(embedding_path = config['embedding_weight_path'], dict_path = config['embedding_dict_path'])
        train_files = []
        val_files = []
        typing_train = None
        typing_val = None
        self.typing_tags = {}
        for task in self.config['multi-task']: 
            try:
                out_numbers.append(task['num_tag']) 
                self.data_transformer.label_to_id = task['label_to_id'] 
                self.data_transformer.train_file = task['train_file_path']
                self.data_transformer.valid_file = task['valid_file_path']
                self.data_transformer.sentence2id(raw_data_path   = task['raw_train_path'],
                                 raw_target_path = task['raw_target_path'],
                                 x_var           = task['x_var'],
                                 y_var           = task['y_var'])

        
                task_train_file,valid_train_file = task['train_file_path'],task['valid_file_path']
                train_files.append(task_train_file)
                val_files.append(valid_train_file)

                x,y = task['x_var'],task['y_var']
            except:
              # not sequential
                if (task['name'] == 'typing'):
                    self.data_transformer.train_file = task['train_file_path']
                    self.data_transformer.valid_file = task['valid_file_path']
                    self.data_transformer.label_to_id = task['label_to_id'] 
                    self.data_transformer.CFET2id(raw_data_path   = task['raw_train_path'])
                    typing_train = ( task['train_file_path'])
                    typing_val = (task['valid_file_path'])

                    self.data_transformer.train_file = task['external_train_path']
                    self.data_transformer.valid_file = task['external_valid_path']
                    self.data_transformer.incre(raw_data_path = task['raw_train_path'],x_var = task['x_var'], y_var = task['y_var'])
                    train_files.append( task['external_train_path'])
                    self.typing_tags = list2dic(task['dict'])




        pretrained = 'roberta' if 'roberta' in self.model_config['name'] else None
        train_dataset = MultiDataset(num_threads = self.num_gpus,data_paths = train_files,x_var = x, y_var = y, skip_header = False, is_train_mode = True, batch_size = config['batch_size'],max_sentence_length = self.config['max_length'], default_token=self.default_tk, gaz=gaz_tree, vocab = self.data_transformer.vocab,rev_vocab = self.data_transformer.rev_vocab,pretrained = pretrained)
        trainSampler = DistributedSampler(train_dataset,shuffle=False) if self.is_distributed else None
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=False, sampler=trainSampler)
            
        val_dataset = MultiDataset(num_threads = self.num_gpus,data_paths = val_files,x_var = x, y_var = y, skip_header = False, is_train_mode = True,  batch_size = config['batch_size'],max_sentence_length = self.config['max_length'], default_token=self.default_tk, gaz=gaz_tree, vocab = self.data_transformer.vocab,rev_vocab = self.data_transformer.rev_vocab, pretrained = pretrained)

        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)

        Tptraindataset = None
        Tpvaldataset = None
        tp_train_loader = None
        tp_val_loader = None
        if typing_train:
            tptraindataset = TypDataset(data_path = typing_train,x_var = x, y_var = y, skip_header=False, is_train_mode=True, batch_size=config['batch_size'],max_sentence_length=self.config['max_length'],default_token=self.default_tk)
            trainSampler = DistributedSampler(tp_train_dataset,shuffle=False) if self.is_distributed else None
            tp_train_loader = DataLoader(tptraindataset, batch_size=self.config['batch_size'], shuffle=False, sampler=trainSampler)

            tpvaldataset = TypDataset(data_path = typing_val,x_var = x, y_var = y, skip_header=False, is_train_mode=True, batch_size=config['batch_size'],max_sentence_length=self.config['max_length'],default_token=self.default_tk)
            tp_val_loader = DataLoader(tpvaldataset, batch_size=self.config['batch_size'], shuffle=False)
        arch = self.model_config['name']
        classifier_out = len(self.typing_tags.keys())
        if (arch == 'cnn_crf' or arch == 'cnn'):
            self.model = CNN(out_numbers      = out_numbers,
                      embedding_dim    = self.config['embedding_dim'],
                      model_config     = self.config['models'][arch],
                      embedding_weight = embedding_weight,
                      vocab_size       = len(self.data_transformer.vocab),
                      device           = self.device,
                      classifier       = classifier_out)
        elif (arch =='bilstm' or arch == 'bilstm_crf'):
            self.model = BiLSTM(out_numbers      = out_numbers,
                      embedding_dim    = self.config['embedding_dim'],
                      model_config     = self.config['models'][arch],
                      embedding_weight = embedding_weight,
                      vocab_size       = len(self.data_transformer.vocab),
                      device           = self.device,
                      classifier       = classifier_out)

        elif (arch == 'lattice_lstm'):
            self.model = Lattice(out_numbers      = out_numbers,
                      embedding_dim    = self.config['embedding_dim'],
                      model_config     = self.config['models'][arch],
                      embedding_weight = embedding_weight,
                      vocab_size       = len(self.data_transformer.vocab),
                      dict_size = len(self.data_transformer.word_vocab),
                      pretrain_dict_embedding = words_embedding,
                      device           = self.device,
                      classifier       = classifier_out)
        elif (arch == 'bert_lstm'):
            self.model = BERT_LSTM(out_numbers      = out_numbers,
                      model_config     = self.config['models'][arch],
                      device           = self.device,
                      classifier       = classifier_out)
        elif (arch == 'roberta_lstm'):
            self.model = RoBERTa_LSTM(out_numbers      = out_numbers,
                      model_config     = self.config['models'][arch],
                      device           = self.device,
                      classifier       = classifier_out)


        self.optimizer = AdamW(params=self.model.parameters(),lr=self.config['learning_rate'])



        self.model = self.model.cuda()
        if self.is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[self.local_rank], output_device=self.local_rank,
        find_unused_parameters=True
            )



        self.model_checkpoint = ModelCheckpoint(checkpoint_dir   = self.config['checkpoint_dir'],
                                       mode             = self.config['mode'],
                                       monitor          = self.config['monitor'],
                                       save_best_only   = self.config['save_best_only'],
                                       best_model_name  = self.config['best_model_name'],
                                       epoch_model_name = self.config['epoch_model_name'],
                                       arch             = arch)



        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.7, patience=4,
                                                           verbose=True, min_lr=1e-9)
      #  self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=30,num_training_steps=len(train_loader))
        self.trainer = Trainer(model            = self.model,
                      model_name       = arch,
                      train_data       = train_loader,
                      val_data         = val_loader,
                      optimizer        = self.optimizer,
                      epochs           = self.config['epochs'],
                      label_to_id      = self.config['multi-task'][0]["label_to_id"],
                      evaluate         = F1_score(num_classes=8),
                      model_checkpoint = self.model_checkpoint,
                      resume           = self.config['resume'],
                      lr_scheduler     = self.scheduler,
                      n_gpu            = self.config['n_gpus'],
                      num_tasks        = 2,
                      device           = self.device,
                      avg_batch_loss   = True,
                      distributed      = self.is_distributed,
                      rank             =self.local_rank,
                      typ_train_data   = tp_train_loader,
                      typ_val_data     = tp_val_loader)


      
        if self.local_rank == 0:
            self.trainer.summary()

        self.trainer.train()

        if len(self.config['n_gpus']) > 0:
            torch.cuda.empty_cache()










    def predict(self):
        if (not self.local_rank == 0):
            return
        arch  = self.config['model']['name']

        seed_everything(seed = self.config['seed'])

        self.checkpoint_path = os.path.join(self.config['checkpoint_dir'].format(arch =arch),
                                    self.config['best_model_name'].format(arch = arch))

        word_dic = True if ('lattice' in self.config['model']['name']) else False

        self.data_transformer = DataTransformer(
                     vocab_path    = self.config['vocab_path'],
                     rev_vocab_path    = self.config['rev_vocab_path'],
                      all_data_path = self.config['all_data_path'],

                     skip_header   = False,
                     is_train_mode = False,
                     default_token = self.default_tk,
                     seed          = self.config['seed'],
                     word_dic     = word_dic)
        self.data_transformer.build_vocab()
        embedding_weight, words_embedding, gaz_tree = self.data_transformer.build_embedding_matrix(embedding_path = self.config['embedding_weight_path'], dict_path = self.config['embedding_dict_path'])
        bs = self.config['batch_size']

        arch = self.model_config['name'] 
        
        if ('lattice' in arch):
            bs = 1
        test_files = []
        typing_test = None
        self.typing_tags = {}
        out_numbers = [] 
        for idx,task in enumerate(self.config['multi-task']): 
            if (not 'result_path' in task.keys()):
                del self.config['multi-task'][idx]
                continue
            if (task['name']!='typing'):
                out_numbers.append(task['num_tag']) 
                self.data_transformer.label_to_id = task['label_to_id'] 
                self.data_transformer.test_file = task['test_file_path']
                self.data_transformer.sentence2id(raw_data_path   = task['raw_test_path'],
                                 raw_target_path = None,
                                 x_var           = task['x_var'],
                                 y_var           = task['y_var'])

        
                task_test_file = task['test_file_path']
                test_files.append(task_test_file)


                x,y = task['x_var'],task['y_var']
            else:
                self.data_transformer.test_file = task['test_file_path']
                self.data_transformer.label_to_id = task['label_to_id'] 
                self.data_transformer.CFET2id(raw_data_path   = task['raw_test_path'])
                typing_test = ( task['test_file_path'])
                self.typing_tags = list2dic(task['dict'])


        test_dataset = MultiDataset(num_threads = self.num_gpus,data_paths = test_files,x_var = x, y_var = y, skip_header = False, is_train_mode = True, batch_size = config['batch_size'],max_sentence_length = self.config['max_length'], default_token=self.default_tk, gaz=gaz_tree, vocab = self.data_transformer.vocab,rev_vocab = self.data_transformer.rev_vocab)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)

        tptestdataset = None

        tp_test_loader = None
        if typing_test:
            tptestdataset = TypDataset(data_path = typing_test,x_var = x, y_var = y, skip_header=False, is_train_mode=False, batch_size=config['batch_size'],max_sentence_length=self.config['max_length'],default_token=self.default_tk)
            testSampler = DistributedSampler(tptestdataset,shuffle=False) if self.is_distributed else None
            tp_test_loader = DataLoader(tptestdataset, batch_size=self.config['batch_size'], shuffle=False, sampler=testSampler)


        classifier_out = len(self.typing_tags.keys())
        if (arch == 'cnn_crf' or arch == 'cnn'):
            self.model = CNN(out_numbers      = out_numbers,
                      embedding_dim    = self.config['embedding_dim'],
                      model_config     = self.config['models'][arch],
                      embedding_weight = embedding_weight,
                      vocab_size       = len(self.data_transformer.vocab),
                      device           = self.device,
                      classifier       = classifier_out)
        elif (arch =='bilstm' or arch == 'bilstm_crf'):
            self.model = BiLSTM(out_numbers      = out_numbers,
                      embedding_dim    = self.config['embedding_dim'],
                      model_config     = self.config['models'][arch],
                      embedding_weight = embedding_weight,
                      vocab_size       = len(self.data_transformer.vocab),
                      device           = self.device,
                      classifier       = classifier_out)

        elif (arch == 'lattice_lstm'):
            self.model = Lattice(out_numbers      = out_numbers,
                      embedding_dim    = self.config['embedding_dim'],
                      model_config     = self.config['models'][arch],
                      embedding_weight = embedding_weight,
                      vocab_size       = len(self.data_transformer.vocab),
                      dict_size = len(self.data_transformer.word_vocab),
                      pretrain_dict_embedding = words_embedding,
                      device           = self.device,
                      classifier       = classifier_out)
        elif (arch == 'bert_lstm'):
            self.model = BERT_LSTM(out_numbers      = out_numbers,
                      model_config     = self.config['models'][arch],
                      device           = self.device,
                      classifier       = classifier_out)
        elif (arch == 'roberta_lstm'):
            self.model = RoBERTa_LSTM(out_numbers      = out_numbers,
                      model_config     = self.config['models'][arch],
                      device           = self.device,
                      classifier       = classifier_out)

        self.model = self.model.cuda()
        checkpoint = torch.load(os.path.sep.join([self.config['checkpoint_dir'],self.config['best_model_name'].format(arch=arch)]))
        self.model.load_state_dict(checkpoint['state_dict'])
        predicter = Predicter(model           = self.model,
                          model_name      = arch, 

                          n_gpu           = self.config['n_gpus'],
                          test_data       = test_loader,
                          checkpoint_path = self.checkpoint_path,
                          label_to_id     = self.config['multi-task'][0]["label_to_id"],
                          distributed      = self.is_distributed,
                          tp_dict = self.typing_tags,
                          typ_test_data   = tp_test_loader
                          )


        predictions = predicter.predict()

        for task,pred in zip(self.config['multi-task'],predictions):
            result_path = task['result_path']
            raw_test_path = task['raw_test_path']
            if task['name']=='typing':
                raw_text_path = None
            else:
                raw_text_path = task['raw_test_path']

            test_write(data = pred,filename = result_path,raw_text_path=raw_text_path, name=task['name'])
            print("current task is %s and output file is at %s"%(task['name'],result_path))

        if len(config['n_gpus']) > 0:
            torch.cuda.empty_cache()

   