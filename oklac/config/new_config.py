#encoding:utf-8
from os import path
import multiprocessing
import os

BASE_DIR = 'nlptooldemo'
RE = {
'pretrained_model_path' :os.path.join(BASE_DIR, '../../bert/pretrained_models/bert-base-chinese'),
'train_file' : os.path.join(BASE_DIR, 'dataset/REdataset/train_small.jsonl'),
'validation_file' : os.path.join(BASE_DIR, 'dataset/REdataset/val_small.jsonl'),
'output_dir' : os.path.join(BASE_DIR, 'dataset/REdataset/saved_models'),
'tagset_file' : os.path.join(BASE_DIR, 'dataset/REdataset/relation.txt'),
'max_len':128,
'ner_source': path.sep.join([BASE_DIR, 'output/result/predict_result.txt']),
'checkpoint':path.sep.join([BASE_DIR,'output/RE_checkpoint/re-best.pth']),
'result':path.sep.join([BASE_DIR,'output/re.txt']),
'dataset':os.path.join(BASE_DIR,'dataset')
}
configs = {
    'dataset_path':path.sep.join([BASE_DIR,'dataset/raw']), 
    'all_data_path': [path.sep.join([BASE_DIR,'dataset/raw/msr_cws_source.txt.txt']),path.sep.join([BASE_DIR,'dataset/raw/pku_cws_source.txt']),path.sep.join([BASE_DIR,'dataset/raw/pd98_cws_source.txt']) ],  
    'pretrained_path':path.sep.join([BASE_DIR, 
                                            'output/checkpoints/roberta.zip']),# for word embedding
    'test_file_path': path.sep.join([BASE_DIR,'dataset/processed/test.json']),   
    'embedding_weight_path': path.sep.join([BASE_DIR, 
                                            'output/embedding/sgns300']), # for character embedding
    'embedding_dict_path': path.sep.join([BASE_DIR, 
                                            'output/embedding/sgns300']),# for word embedding
    'vocab_path': path.sep.join([BASE_DIR,'dataset/processed/vocab.pkl']), 
    'rev_vocab_path': path.sep.join([BASE_DIR,'dataset/processed/rev_vocab.pkl']), 

    'log_dir': path.sep.join([BASE_DIR, 'output/log']), 
    'writer_dir': path.sep.join([BASE_DIR, 'output/TSboard']),
    'figure_dir': path.sep.join([BASE_DIR, 'output/figure']),
    'checkpoint_dir': path.sep.join([BASE_DIR, 'output/checkpoints/bilstm_crf']),
    'embedding_dir': path.sep.join([BASE_DIR, 'output/embedding']),
    'valid_size': 0.5, 
    'min_freq': 1,   
    'max_length': 80,  
    'max_features': 100000, 
    'embedding_dim':300, 

    'batch_size': 32,  
    'epochs': 100,    
    'start_epoch': 1,
    'learning_rate':1e-6,#2e-6, #for bert
    'weight_decay': 5e-4, 
    'n_gpus': [6], 
    'x_var':'source', 
        'mode': 'min',    
    'y_var':'target', 
    'seed': 2018,   
    'monitor': 'val_loss',  
    'save_best_only':True, 
    'best_model_name': '{arch}-best-cws1.pth', 
    'epoch_model_name': '{arch}-{epoch}-{val_loss}.pth', 
    'save_checkpoint_freq': 10,

    'multi-task':
    [ 

    {    'label_to_id': {
  
        "B-PER": 1,  
        "I-PER": 2,
        "B-LOC": 3, 
        "I-LOC": 4,
        "B-ORG": 5,  
        "I-ORG": 6,
        "B-T":7,
        "I-T":7,
        "O": 7,
        "B_PER": 1,  
        "I_PER": 2,
        "B_LOC": 3, 
        "I_LOC": 4,
        "B_ORG": 5,  
        "I_ORG": 6,
        "B_T":7,
        "I_T":7

  },
    'num_tag':8, #source_BIO_2014_cropus
    'raw_train_path': path.sep.join([BASE_DIR,'dataset/raw/msra_ner_source.txt']),  
    'raw_target_path': path.sep.join([BASE_DIR,'dataset/raw/msra_ner_target.txt']), 
  #  'raw_train_path': path.sep.join([BASE_DIR,'dataset/raw/source_BIO_2014_cropus.txt']),  
  #  'raw_target_path': path.sep.join([BASE_DIR,'dataset/raw/target_BIO_2014_cropus.txt']), 
   # 'raw_train_path': path.sep.join([BASE_DIR,'dataset/raw/msra_ner_source1.txt']), 
  #  'raw_target_path': path.sep.join([BASE_DIR,'dataset/raw/msra_ner_target1.txt']), 
    'raw_test_path': path.sep.join([BASE_DIR,'dataset/raw/test.txt']),       
    'test_file_path': path.sep.join([BASE_DIR,'dataset/processed/test.json']),   
    'result_path': path.sep.join([BASE_DIR, 'output/result/predict_result.txt']),
    'x_var':'source', 
    'y_var':'target',
    'train_file_path': path.sep.join([BASE_DIR,'dataset/processed/train.json']), 
    'valid_file_path': path.sep.join([BASE_DIR,'dataset/processed/validb.json']),
    'name':"NER"
    },

 {
        'dict':path.sep.join([BASE_DIR,'dataset/raw/type_dict.txt']),
        'x_var':'source', 
    'y_var':'target',
    'label_to_id': {
  
        "PER": 1,  
        "LOC": 3,  
        "ORG": 5,
        "O": 7

  },
    'name':"typing",
    'raw_test_path':path.sep.join([BASE_DIR,'dataset/raw/test_typ.txt']),       
    'raw_train_path':path.sep.join([BASE_DIR,'dataset/raw/typing31.json']),
    'train_file_path':path.sep.join([BASE_DIR,'dataset/processed/train_typ_.json']),
    'valid_file_path':path.sep.join([BASE_DIR,'dataset/processed/valid_typ_.json']),   
    'external_train_path':path.sep.join([BASE_DIR,'dataset/processed/train_ext_.json']),   
    'external_valid_path':path.sep.join([BASE_DIR,'dataset/processed/valid_ext_.json'])  ,
        'test_file_path': path.sep.join([BASE_DIR,'dataset/processed/test_typ.json']),   
    'num_tag':10,
    'result_path':path.sep.join([BASE_DIR, 'output/result/predict_result_typ.txt'])
    },
    {'num_tag':8,
    'name':'add'}

    
    ],
    'seed':2018,
    'resume':False,
    'model':{'name':'bert_lstm'
    },
    
    'models': {
    'bert_lstm':{'hidden_size': 200,
                              'bi_tag': True,
                             'dropout_p':0.5,
                             'dropout_emb':0.0,
                             'num_layer': 1,
                             'use_cuda':True,
                             'crf':True},
    'roberta_lstm':{'hidden_size': 200,
                              'bi_tag': True,
                             'dropout_p':0.5,
                             'dropout_emb':0.0,
                             'num_layer': 1,
                             'use_cuda':True,
                             'crf':True},
    'lattice_lstm':{'hidden_size': 200,
                             'bi_tag': True,
                             'dropout_p':0.5,
                             'dropout_emb':0.0,
                             'num_layer': 1,
                             'use_cuda':True,
                             'crf':True},
    'cnn_crf':{'hidden_size': 200,
                             'bi_tag': True,
                             'dropout_p':0.5,
                             'dropout_emb':0.0,
                             'num_layer': 1,
                             'use_cuda':True,
                             'crf':True},
    'bilstm':{'hidden_size': 200,
                             'bi_tag': True,
                             'dropout_p':0.5,
                             'dropout_emb':0.0,
                             'num_layer': 1,
                             'use_cuda':True,
                             'crf':False},
    'bilstm_crf':{'hidden_size': 200,
                         'bi_tag': True,
                             'dropout_p':0.5,
                             'dropout_emb':0.0,
                             'num_layer': 1,
                             'use_cuda':True,
                             'crf':True}
              }
}
"""

 ,
    {    
    'label_to_id': {    
        "0":1,
        "1":2 
    }
,    
    'num_tag':3,
    'raw_train_path': path.sep.join([BASE_DIR,'dataset/raw/msr_cws_source.txt']),  
    'raw_target_path': path.sep.join([BASE_DIR,'dataset/raw/msr_cws_target.txt']), 
    'raw_test_path': path.sep.join([BASE_DIR,'dataset/raw/test_cws.txt']),       
    'test_file_path': path.sep.join([BASE_DIR,'dataset/processed/test_cws.json']),   
    'result_path': path.sep.join([BASE_DIR, 'output/result/predict_result_cws.txt']),
    'x_var':'source', 
    'y_var':'target',
     'train_file_path': path.sep.join([BASE_DIR,'dataset/processed/train_cws.json']), 
    'valid_file_path': path.sep.join([BASE_DIR,'dataset/processed/valid_cws.json']),
    'name':"CWS"
    },
        {
        'dict':path.sep.join([BASE_DIR,'dataset/processed/valid_cws.json']),
        'x_var':'source', 
    'y_var':'target',
    'label_to_id': {
  
        "PER": 1,  
        "LOC": 3,  
        "ORG": 5,
        "O": 7

  },
    'name':"typing",
    'raw_train_path':path.sep.join([BASE_DIR,'dataset/raw/typing31.json']),
    'train_file_path':path.sep.join([BASE_DIR,'dataset/processed/train_typ31.json']),
    'valid_file_path':path.sep.join([BASE_DIR,'dataset/processed/valid_typ31.json']),   
    'external_train_path':path.sep.join([BASE_DIR,'dataset/processed/train_ext31.json']),   
    'external_valid_path':path.sep.join([BASE_DIR,'dataset/processed/valid_ext31.json'])  ,
    'num_tag':10
    },
    {'num_tag':8,
    'name':'add'}

    

"""