#encoding:utf-8
import os
import sys
import time
import numpy as np
import torch
from itertools import chain
from torch.autograd import Variable
from ..callback.progressbar import ProgressBar
from ..utils.utils import AverageMeter
from .train_utils import restore_checkpoint,model_device
from .metrics import Entity_Score, Word_Score
from .train_utils import batchify_with_label
from ..config.new_config import configs as config
from collections import Counter
from tqdm import tqdm
from transformers import BertModel

class Trainer(object):
    def __init__(self,model,
                 model_name,
                 train_data,
                 val_data,
                 optimizer,
                 epochs,
                 evaluate,
                 num_tasks = 1,
                 device = None,
                 avg_batch_loss   = False,
                 distributed = True,
                 rank = 0,
                 label_to_id      = None,
                 n_gpu            = None,
                 lr_scheduler     = None,
                 resume           = None,
                 model_checkpoint = None,
                 writer           = None,
                 verbose = 1,
                 typ_train_data = None,
                 typ_val_data = None):
        self.model            = model     
        self.model_name       = model_name      
        self.train_data       = train_data         
        self.val_data         = val_data          
        self.epochs           = epochs          
        self.optimizer        = optimizer          
        self.verbose          = verbose            
        self.writer           = writer         
        self.resume           = resume           
        self.model_checkpoint = model_checkpoint 
        self.lr_scheduler     = lr_scheduler     
        self.evaluate         = evaluate         
        self.n_gpu            = n_gpu             
        self.avg_batch_loss   = avg_batch_loss     
        self.num_tasks        = num_tasks
        self.typ_train_data = typ_train_data
        self.typ_val_data = typ_val_data
        self.task={}
        idx = 0
        for t in config['multi-task']:
            if (t['name']=='typing'):
                continue
            num = t['num_tag']
            self.task[idx]=num
            idx+=1
        self.id_to_label      =  {1:"B_PER",2:"I_PER",3:"B_LOC",4:"I_LOC",5:"B_ORG",6:"I_ORG",7:"O"}
        self._reset()
        self.device = device
        self.distributed = distributed
        self.rank = rank


    def _reset(self):
        self.val_word_score = Word_Score()
        self.train_entity_score = Entity_Score(id_to_label=self.id_to_label)
        self.val_entity_score   = Entity_Score(id_to_label=self.id_to_label)
        self.batch_num         = len(self.train_data)
        if ('lattice' in self.model_name):
            self.progressbar       = ProgressBar(n_batch = self.batch_num/32,eval_name='acc',loss_name='loss')
        else:
            self.progressbar       = ProgressBar(n_batch = self.batch_num,eval_name='acc',loss_name='loss')
        if (self.typ_train_data):
            self.typ_progressbar = ProgressBar(n_batch = len(self.typ_train_data),eval_name = 'acc',loss_name='loss')

        self.model,self.device = model_device(n_gpu=self.n_gpu,model = self.model)
        self.start_epoch = 1

        if self.resume:
            arch = self.model_checkpoint.arch
            resume_path = os.path.join(self.model_checkpoint.checkpoint_dir.format(arch = arch),
                                       self.model_checkpoint.best_model_name.format(arch = arch))
            print("\nLoading checkpoint: {} ...".format(resume_path))
            resume_list = restore_checkpoint(resume_path = resume_path,model = self.model,optimizer = self.optimizer)
            self.model     = resume_list[0]
            self.optimizer = resume_list[1]
            best           = resume_list[2]
            self.start_epoch = resume_list[3]
        

            if self.model_checkpoint:
                self.model_checkpoint.best = best
            print("\nCheckpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))

    def summary(self):
        trainModel = self.model.module if self.distributed else self.model
        model_parameters = filter(lambda p: p.requires_grad, trainModel.parameters())

        params = sum([np.prod(p.size()) for p in model_parameters])




    def _save_info(self,epoch,val_loss):
        state = {
            'epoch': epoch,
            'arch': self.model_checkpoint.arch,
            'state_dict': self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'val_loss': round(val_loss,4)
        }
        return state
        
    def sequence_mask(self, sequence_length, max_len, device=None):   # sequence_length :(batch_size, )
        batch_size = sequence_length.size(0)     
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_range_expand = Variable(seq_range_expand)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
        return seq_range_expand < seq_length_expand 
    def _typ_train_epoch(self):
        model = self.model
        if self.distributed:
            model = model.module

        model.train()
        loss_fn = torch.nn.BCEWithLogitsLoss()
        try:
            encoder = model.bert_encoder
        except:
            encoder = BertModel.from_pretrained('bert-base-chinese')

        for idx,(lc,rc,label,m,lm,rm,mm) in enumerate(self.typ_train_data):
            if (idx==2):
                break
            lc = (torch.stack(lc,dim=0)).t().cuda()
            lm = (torch.stack(lm,dim=0)).t().cuda()
            hl = encoder(lc,lm)[1]
            del lm
            rc = (torch.stack(rc,dim=0)).t().cuda()
            rm = (torch.stack(rm,dim=0)).t().cuda()
            hr = encoder(rc,rm)[1]
            del rm
            m = (torch.stack(m,dim=0)).t().cuda()
            mm = (torch.stack(mm,dim=0)).t().cuda()
            hm = encoder(m,mm)[1]
            del mm

            joined = torch.cat((hl,hm,hr),dim=1)
            tag = model.classifier(joined)
            label = (torch.stack(label,dim=0)).t().float().cuda()
            loss = loss_fn(tag,label)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.typ_progressbar.step(batch_idx=idx,
                                      loss     = loss.item(),
                                      acc      = 0,
                                      f1       = 0,
                                      use_time = 0)         


    def _typ_valid_epoch(self):
        model = self.model
        if self.distributed:
            model = model.module
        model.eval()

        try:
            encoder = model.bert_encoder
        except:
            encoder = BertModel.from_pretrained('bert-base-chinese')
        loss_fn = torch.nn.BCEWithLogitsLoss()
        tt = torch.nn.Sigmoid()
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        with torch.no_grad():
            for idx,(lc,rc,label,m,lm,rm,mm) in tqdm(enumerate(self.typ_val_data)):
                if (idx==2):
                    break
                lc = (torch.stack(lc,dim=0)).t().cuda()
                lm = (torch.stack(lm,dim=0)).t().cuda()
                hl = encoder(lc,lm)[1]
                del lm
                rc = (torch.stack(rc,dim=0)).t().cuda()
                rm = (torch.stack(rm,dim=0)).t().cuda()
                hr = encoder(rc,rm)[1]
                del rm
                m = (torch.stack(m,dim=0)).t().cuda()
                mm = (torch.stack(mm,dim=0)).t().cuda()
                hm = encoder(m,mm)[1]
                del mm

                joined = torch.cat((hl,hm,hr),dim=1)
                tag = model.classifier(joined)
                label = (torch.stack(label,dim=0)).t().float()
                rr = tt(tag)
                hits = (rr>0.4).float().cpu()


                confusion = hits/label
                true_positives += torch.sum(torch.sum(confusion==1,dim=1),dim=0).item() 
                false_positives += torch.sum(torch.sum(confusion==float('inf'),dim=1),dim=0).item() 
                true_negatives += torch.sum(torch.sum(torch.isnan(confusion),dim=1),dim=0).item() 
                false_negatives += torch.sum(torch.sum(confusion==0,dim=1),dim=0).item() 
            print("\nEntity Typing prec: [%f] recall: [%f]"%(true_positives/(true_positives+false_positives),true_positives/(true_positives+false_negatives) ))


    def _valid_epoch(self):
        if self.distributed:
            self.model.module.eval()
        else:
            self.model.eval()
        val_losses_ner = AverageMeter()
        val_losses_cws = AverageMeter()#!
        val_acc_ner    = AverageMeter()
        val_acc_cws = AverageMeter()#!
        val_f1     = AverageMeter()
        

        for idx,(inputs,gaz,target,length,source) in enumerate(self.val_data):
            check = sum(source.numpy().tolist())
            if (check%self.num_tasks!=0):
                continue
            if (idx==2):
                break

            inputs = (torch.stack(inputs,dim=0)).t().cuda()

            target = (torch.stack(target,dim=0)).t().cuda()

            batch_size = inputs.size(0)

            masks = self.sequence_mask(length, config['max_length'],self.device) if ('bert' in self.model_name) else None
            train_task = self.task[check/batch_size]
            if (train_task==0):
                continue
            if masks!=None:
                masks = masks.cuda()
            length=length.cuda()

            outputs = self.model(inputs, length ,out_number = train_task, masks=masks, gaz=gaz)
            mask,target = batchify_with_label(inputs = inputs,target = target,outputs = outputs)
            if self.distributed:
                loss = self.model.module.crf.neg_log_likelihood_loss(outputs, mask,target,out_number = train_task) 
            else:
                loss = self.model.crf.neg_log_likelihood_loss(outputs, mask,target,out_number = train_task) 
            if self.avg_batch_loss:
                loss /=  batch_size
            if self.distributed:
                _,predicts = self.model.module.crf(outputs, mask,out_number = train_task)
            else:
                _,predicts = self.model.crf(outputs, mask,out_number = train_task)  
            acc,f1 = self.evaluate(predicts,target=target)
            if (self.task[check/batch_size]==8):
                val_losses_ner.update(loss.item(),batch_size)
                val_acc_ner.update(acc.item(),batch_size)

            else:
                val_losses_cws.update(loss.item(),batch_size)
                val_acc_cws.update(acc.item(),batch_size)
            val_f1.update(f1.item(),batch_size)
            if self.device != 'cpu':
                predicts = predicts.cpu().numpy()
                target = target.cpu().numpy()
            if (self.task[check/batch_size]==8):
                self.val_entity_score.update(pred_paths=predicts, label_paths=target)
            elif (self.task[check/batch_size]==3):
                self.val_word_score.update(pred_paths=predicts, label_paths=target)

        return {
            'val_loss': val_losses_cws.avg,
            'val_loss2': val_losses_cws.avg,
            'val_acc': val_acc_ner.avg,
            'val_acc2': val_acc_cws.avg,
            'val_f1': val_f1.avg
        }



    def _train_epoch(self):
        if self.distributed:
            self.model.module.train()
        else:
            self.model.train()
        train_loss_ner = AverageMeter()
        train_acc_ner  = AverageMeter()
        train_f1   = AverageMeter()
        train_loss_cws = AverageMeter()
        train_acc_cws  = AverageMeter()

                
        batch_loss_ner = 0
        acc_ner=0
        acc_cws=0
        batch_loss_cws = 0 
        loss_c=0

        for idx,(inputs,gaz,target,length,source) in enumerate(self.train_data):
            check = sum(source.numpy().tolist())

            if (idx==2):
                break
            if (check%self.num_tasks!=0):
                continue
            inputs = (torch.stack(inputs,dim=0)).t().cuda()
            target = (torch.stack(target,dim=0)).t().cuda()
            batch_size = inputs.size(0)

            train_task = self.task[check/batch_size]
            try:
                length = length.cuda()
                masks = self.sequence_mask(length, config['max_length'],self.device) if ('bert' in self.model_name) else None
                if (masks!=None):
                    masks = masks.cuda()
                outputs = self.model(inputs, length, out_number = train_task, masks = masks, gaz=gaz)

                mask, target = batchify_with_label(inputs=inputs, target=target, outputs=outputs)
                if self.distributed:
                    loss    = self.model.module.crf.neg_log_likelihood_loss(outputs,mask,target, out_number = train_task)
                else:
                    loss    = self.model.crf.neg_log_likelihood_loss(outputs,mask,target, out_number = train_task)
                if self.avg_batch_loss:
                    loss  /= batch_size
                if self.distributed:   
                    _,predicts = self.model.module.crf(outputs,mask,out_number = train_task)
                else:
                    _,predicts = self.model.crf(outputs,mask,out_number = train_task)         
                acc,f1 = self.evaluate(predicts,target)

                if (not 'lattice' in self.model_name):
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    if (idx%config['batch_size']==0):
                        loss_c = loss
                    else:
                        loss_c+=loss

                    if ((idx+1)%config['batch_size']==0):
                        loss=0
                        loss_c.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.model.zero_grad()


            
                if (self.task[check/batch_size]==8):
                    train_loss_ner.update(loss.item(),batch_size)
                    train_acc_ner.update(acc.item(),batch_size)
                    train_f1.update(f1.item(),batch_size)
                else:
                    train_loss_cws.update(loss.item(),batch_size)
                    train_acc_cws.update(acc.item(),batch_size)


                if self.device != 'cpu':
                    predicts = predicts.cpu().numpy()
                    target   = target.cpu().numpy()
                #self.train_entity_score.update(pred_paths=predicts,label_paths=target)
            except:
                pass
            if (self.verbose >= 1):
                if ('lattice' in self.model_name and ((idx+1)%config['batch_size']==0)):
                    self.progressbar.step(batch_idx=idx,
                                      loss     = loss_c.item(),
                                      acc      = acc.item(),
                                      f1       = f1.item(),
                                      use_time = 0)                 
                else:
                    self.progressbar.step(batch_idx=idx,
                                      loss     = loss.item(),
                                      acc      = acc.item(),
                                      f1       = f1.item(),
                                      use_time = 0)

        train_log = {
            'loss': train_loss_ner.avg,
            'acc': train_acc_ner.avg,
            'loss2': train_loss_cws.avg,
            'acc2': train_acc_cws.avg,
            'f1': train_f1.avg,
        }
        return train_log

    def train(self):
        for epoch in range(self.start_epoch,self.start_epoch+self.epochs):
            if (self.rank == 0):
                print("----------------- training start -----------------------")
                print("Epoch {i}/{epochs}......".format(i=epoch, epochs=self.start_epoch+self.epochs -1))

            train_log = self._train_epoch()
            if (self.typ_train_data):
                self._typ_train_epoch()
                self._typ_valid_epoch()


            if self.rank == 0:

                val_log = self._valid_epoch()

                logs = dict(train_log,**val_log)

                self.val_word_score.result()
                self.val_entity_score.result()
                print('\nEpoch: %d - loss: %.4f acc: %.4f - f1: %.4f val_loss: %.4f - val_acc: %.4f - val_f1: %.4f - val_acc2: %.4f'%(
                            epoch,logs['loss'],logs['acc'],logs['f1'],logs['val_loss'],logs['val_acc'],logs['val_f1'],logs['val_acc2'])
                             )

                if self.lr_scheduler:
                    self.lr_scheduler.step(logs['loss'],epoch)


                if self.model_checkpoint:
                    state = self._save_info(epoch,val_loss = logs['val_loss'])
                    self.model_checkpoint.step(current=logs[self.model_checkpoint.monitor],state = state)
