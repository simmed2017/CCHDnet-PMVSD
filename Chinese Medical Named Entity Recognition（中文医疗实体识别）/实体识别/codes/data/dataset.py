import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)


class GPERDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        text = item["text"].lower()
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True,padding='max_length', max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for ent, ent_type in item['entity_list']:
            ent = ent.lower()
            ent = self.tokenizer.encode(ent, add_special_tokens=False)
            ent_h = self.data_processor.search(input_ids,ent)
            if ent_h != -1:
                spoes.add((self.data_processor.entity_to_id[ent_type],ent_h, ent_h+len(ent)-1))

        entity_labels = [set() for i in range(self.data_processor.entity_nums)]
        for ent_id, ent_h, ent_t in spoes:
            entity_labels[ent_id].add((ent_h, ent_t)) 
        for label in entity_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        return entity_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels = []
        for item in examples:
            head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels
        
