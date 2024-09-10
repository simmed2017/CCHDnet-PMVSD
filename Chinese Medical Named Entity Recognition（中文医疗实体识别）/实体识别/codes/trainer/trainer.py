import torch
import torch.nn as nn
import os
import json
import jsonlines
from copy import deepcopy
import shutil
import math
import numpy as np
import torch.nn.functional as F
from d2l import torch as d2l
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import ProgressBar, TokenRematch, get_time, save_args
from metrics import er_metric, re_metric, gen_metric, rc_metric, p2so_metric
from optimizer import GPLinkerOptimizer
from loss import multilabel_categorical_crossentropy, sparse_multilabel_categorical_crossentropy


def kl_div_for_gplinker(a,b,reduction='sum',rdrop_type='softmax'):
    if rdrop_type=='softmax':
        a_2 = a.softmax(dim=-1).reshape(-1)
        b_2 = b.softmax(dim=-1).reshape(-1)
    else:
        a_2 = torch.sigmoid(a).reshape(-1)
        b_2 = torch.sigmoid(b).reshape(-1)
    a = a.reshape(-1)
    b = b.reshape(-1)
    kl_val = torch.dot(a_2-b_2,a-b)
    if reduction != 'sum':
        kl_val = kl_val/a.shape[0]
    return kl_val


class Trainer(object):
    def __init__(
            self,
            args,
            data_processor,
            logger,
            model=None,
            tokenizer=None,
            train_dataset=None,
            eval_dataset=None,
    ):

        self.args = args
        self.model = model
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        

        if train_dataset is not None and isinstance(train_dataset, Dataset):
            self.train_dataset = train_dataset

        if eval_dataset is not None and isinstance(eval_dataset, Dataset):
            self.eval_dataset = eval_dataset

        self.logger = logger

    def train(self):
        args = self.args
        logger = self.logger
        model = self.model
        self.output_dir = os.path.join(args.output_dir, args.time)

        model.to(args.device)
            
        train_dataloader = self.get_train_dataloader()

        num_training_steps = len(train_dataloader) * args.epochs
        num_warmup_steps = num_training_steps * args.warmup_proportion
        num_examples = len(train_dataloader.dataset)
        
        if args.method_name == 'gper' or args.method_name == 'dual-re':
            optimizer = GPLinkerOptimizer(model, train_steps= len(train_dataloader)  * args.epochs)
        else:
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.args.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                        num_training_steps=num_training_steps)

        logger.info("***** Running training *****")
        logger.info("Num samples %d", num_examples)
        logger.info("Num epochs %d", args.epochs)
        logger.info("Num training steps %d", num_training_steps)
        logger.info("Num warmup steps %d", num_warmup_steps)

        global_step = 0
        best_step = None
        best_score = 0
        cnt_patience = 0
        
        animator = d2l.Animator(xlabel='epoch', xlim=[0, args.epochs], ylim=[0, 1], fmts=('k-', 'r--', 'y-.', 'm:', 'g--', 'b-.', 'c:'),
                                legend=[f'train loss/{args.loss_show_rate}', 'train_p', 'train_r', 'train_f1', 'val_p', 'val_r', 'val_f1'])
        # 统计指标
        metric = d2l.Accumulator(5)
        num_batches = len(train_dataloader)
        

        
        for epoch in range(args.epochs):
            pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
            for step, item in enumerate(train_dataloader):
                loss, train_p, train_r, train_f1 = self.training_step(model, item)
                logger.info('loss:{}   p:{}   r:{}   f1:{}'.format(loss,train_p,train_r,train_f1))
                loss = loss.item()
                metric.add(loss, train_p, train_r, train_f1, 1)
                pbar(step, {'loss': loss})

                if args.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                if args.method_name != 'gper' and args.method_name != 'dual-re':
                    scheduler.step()
                optimizer.zero_grad()

                global_step += 1
            # 一个epoch结束

            val_p, val_r, val_f1 = self.evaluate()
            print('\nepoch:{} evaluate:P:{}\tR:{}\tF:{}'.format(epoch+1,val_p, val_r, val_f1))
            animator.add(
                global_step / num_batches, 
                (# metric[0] / metric[-1] / args.loss_show_rate, # loss太大，除以loss_show_rate才能在[0,1]范围内看到
                    loss / args.loss_show_rate,
                    train_p,  # metric[1] / metric[-1],
                    train_r,  # metric[2] / metric[-1],
                    train_f1, # metric[3] / metric[-1],
                    val_p,
                    val_r,
                    val_f1))
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            d2l.plt.savefig(os.path.join(self.output_dir, '训练过程.jpg'), dpi=300)

            if args.save_metric == 'step':
                save_metric = global_step
            elif args.save_metric == 'epoch':
                save_metric = epoch
            elif args.save_metric == 'loss':
                # e的700次方刚好大于0，不存在数值问题
                # 除以10，避免loss太大，exp(-loss)次方由于数值问题会小于0，导致存不上，最大可以处理7000的loss
                save_metric = math.exp(- loss / 10) # math.exp(- metric[0] / metric[-1] / 10)
            elif args.save_metric == 'p':
                save_metric = val_p
            elif args.save_metric == 'r':
                save_metric = val_r
            elif args.save_metric == 'f1':
                save_metric = val_f1

            if save_metric > best_score:
                best_score = save_metric
                best_step = global_step
                cnt_patience = 0
                self.args.loss = loss # metric[0] / metric[-1]
                self.args.train_p, self.args.train_r, self.args.train_f1 = train_p, train_r, train_f1
                                    #  metric[1] / metric[-1], metric[2] / metric[-1], metric[3] / metric[-1]
                self.args.val_p, self.args.var_r, self.args.val_f1 = val_p, val_r, val_f1
                self._save_checkpoint(model)
            else:
                cnt_patience += 1
                self.logger.info("Earlystopper counter: %s out of %s", cnt_patience, args.earlystop_patience)
                if cnt_patience >= self.args.earlystop_patience:
                    break
                
            if args.method_name == 'gper' or args.method_name == 'dual-re' or args.method_name == 'baseline':
                self.args.loss = loss
                self.args.train_p, self.args.train_r, self.args.train_f1 = train_p, train_r, train_f1
                self._save_checkpoint(model)
        
        logger.info(f"\n***** {args.finetuned_model_name} model training stop *****" )
        logger.info(f'finished time: {get_time()}')
        logger.info(f"best val_{args.save_metric}: {best_score}, best step: {best_step}\n" )


        return global_step, best_step

    def evaluate(self):
        raise NotImplementedError

    def _save_checkpoint(self, model):
        args = self.args
        model = model.to(torch.device('cpu'))
        torch.save(model.state_dict(), os.path.join(self.output_dir, 'pytorch_model.pt'))
        self.logger.info('Saving models checkpoint to %s', self.output_dir)
        self.tokenizer.save_vocabulary(save_directory=self.output_dir)
        model.encoder.config.to_json_file(os.path.join(self.output_dir, 'config.json'))
        model = model.to(args.device)
        save_args(args, self.output_dir)
    
    
    def load_checkpoint(self):
        args = self.args
        load_dir = os.path.join(args.output_dir, args.model_version)
        self.logger.info(f'load model from {load_dir}')
        # 每次加载到cpu中，防止爆显存
        checkpoint = torch.load(os.path.join(load_dir, 'pytorch_model.pt'), map_location=torch.device('cpu'))
        if 'module' in list(checkpoint.keys())[0].split('.'):
            self.model = nn.DataParallel(self.model, device_ids=args.devices).to(args.device)
        self.model.load_state_dict(checkpoint)
    
    def training_step(self, model, item):
        raise NotImplementedError

    def get_train_dataloader(self):
        collate_fn = self.train_dataset.collate_fn if hasattr(self.train_dataset, 'collate_fn') else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

    def get_eval_dataloader(self):
        collate_fn = self.eval_dataset.collate_fn if hasattr(self.eval_dataset, 'collate_fn') else None
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

    def get_test_dataloader(self, test_dataset, batch_size=None):
        collate_fn = test_dataset.collate_fn if hasattr(test_dataset, 'collate_fn') else None
        if not batch_size:
            batch_size = self.args.eval_batch_size

        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        
class GPERTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPERTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
    def evaluate(self):
        args = self.args
        logger = self.logger
        test_samples = self.data_processor.get_dev_sample_for_predict()
        gold_datas = deepcopy(test_samples)
        num_examples = len(test_samples)
        logger.info("***** Running prediction for evaluation *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir,args.data_dir,args.model_version)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = os.path.join(output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples,mode='dev')
            for data in predict_data0:
                f.write(data)
        
        # 计算prf
        # 计算f1
        preds = []
        golds = []
        for index,data in enumerate(gold_datas):
            for spo in data['entity_list']:
                golds.append((index,spo['entity'].lower(),spo['entity_type']))

        for index,data in enumerate(predict_data0):
            for spo in data['entity_list']:
                preds.append((index,spo['entity'].lower(),spo['entity_type']))
        try:
            P = len(set(preds) & set(golds)) / len(set(preds))
            R = len(set(preds) & set(golds)) / len(set(golds))
            F = (2 * P * R) / (P + R)
        except:
            P = 0
            R = 0
            F = 0
        return P,R,F

    def training_step(self, model, item):
        model.train()
        device = self.args.device
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = kl_div_for_gplinker(logits[::2],logits[1::2],reduction='mean',rdrop_type=self.args.rdrop_type)

            loss = loss + loss_kl / 2 * self.args.rdrop_alpha
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
     
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, mode='test', isPbar=True):
        args = self.args
        if mode == 'test':
            predict_threshold = args.predict_threshold
        else:
            predict_threshold = 0.5
        model = self.model
        device = args.device
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text'].lower()
            if isPbar:
                pbar(step)
            model.eval()
            token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True, max_length=args.max_length, truncation=True)["offset_mapping"]
            new_span, entities = [], []
            for i in token2char_span_mapping:
                if i[0] == i[1]:
                    new_span.append([])
                else:
                    if i[0] + 1 == i[1]:
                        new_span.append([i[0]])
                    else:
                        new_span.append([i[0], i[-1] - 1])
            threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            score = model(input_ids, attention_mask, token_type_ids)
            outputs = torch.sigmoid(score[0].data).cpu().numpy()
            entity_list = []
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for l, h, t in zip(*np.where(outputs > predict_threshold)):
                entity_list.append(
                    {
                        'entity':text[new_span[h][0]:new_span[t][-1] + 1],
                        'entity_type':self.data_processor.id_to_entity[l]
                    }
                )
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict(self):
        args = self.args
        logger = self.logger
        test_samples = self.data_processor.get_test_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+str(args.predict_threshold))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = os.path.join(output_dir, 'entity_list.jsonl')
        if os.path.exists(output_dir):
            print('已经预测过')
            return
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples)
            for data in predict_data0:
                f.write(data)
        save_args(args,os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+str(args.predict_threshold)))
