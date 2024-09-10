import os
import json
import jsonlines
from collections import defaultdict
import random


class GPERDataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root,'CMeIE_train.json')
        self.dev_path = os.path.join(root, 'CMeIE_dev.json')
        self.test_path = os.path.join(root, 'CMeIE_test.json')
        self.schema_path = os.path.join(root, '53_schemas.json')
        entity_labels = []
        with jsonlines.open(self.schema_path,'r') as f:
            for line in f:
                entity_labels.append(line['subject_type'])
                entity_labels.append(line['object_type'])
        entity_labels = sorted(list(set(entity_labels)))
        self.entity_nums = len(entity_labels)
        self.entity_to_id = {entity_labels[i]:i for i in range(len(entity_labels))}
        self.id_to_entity = {v:k for k,v in self.entity_to_id.items()}
        print('实体数量:{}'.format(self.entity_nums))
        print('entity_to_id:{}'.format(self.entity_to_id))
        print('id_to_entity:{}'.format(self.id_to_entity))

        
        
    def get_train_sample(self):
        return self._pre_process(self.train_path, mode='train')

    def get_dev_sample(self):
        return self._pre_process(self.dev_path, mode='train')

    def get_dev_sample_for_predict(self):
        with open(self.dev_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list

    def get_test_sample(self):
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list

    def merge(self,data_1,data_2):
        data_1.extend(data_2)
        return data_1

    
    def search(self, sequence, pattern):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回0。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
        
    def _pre_process(self, path, mode):
        new_data = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if mode == 'train':
                random.shuffle(lines)
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in lines:
                num += 1
                line = json.loads(line)
                text = line['text']
                entity_list = []
                for ent in line['entity_list']:
                    entity_list.append((ent['entity'],ent['entity_type']))
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list":entity_list
                    })
        return new_data
