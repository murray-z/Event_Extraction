# -*- coding: utf-8 -*-

import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from config import *

LABELS = ['企业合作', '安全事故']
TAGS = ['B-SUB', 'I-SUB', 'B-OBJ' 'I-OBJ', 'O', '<PAD>']  # '<PAD>' 用于padding序列标注


label2idx = {label: idx for idx, label in enumerate(LABELS)}
idx2label = {idx: label for idx, label in enumerate(LABELS)}
tag2idx = {tag: idx for idx, tag in enumerate(TAGS)}
idx2tag = {idx: tag for idx, tag in enumerate(TAGS)}
tokenizer = BertTokenizer.from_pretrained(config["model_name"])
o_idx = tag2idx["O"]
pad_idx = tag2idx["<PAD>"]

train_data_path = "./train.json"
dev_data_path = "./dev.json"
test_data_path = "./test.json"


class EventDataSet(Dataset):
    def __init__(self, data_path, config):
        texts = []
        labels = []
        tags = []
        max_seq_len = config["max_seq_len"]
        with open(data_path) as f:
            for idx, line in enumerate(f):
                line = json.loads(line)
                text = line["sentence"]
                label = line["label"]
                tag = line["tag"]
                texts.append(text)
                labels.append(label)
                tags.append(tag)

        self.input_ids, self.token_type_ids, self.attention_mask, self.tag_ids, self.label_ids = \
            self.padding(texts, labels, tags, max_seq_len)
        print("=====data_set:{}====".format(data_path))
        print("input_ids size: ", self.input_ids.size())
        print("token_type_ids size: ", self.token_type_ids.size())
        print("attention_mask size: ", self.attention_mask.size())
        print("tag_ids size: ", self.tag_ids.size())
        print("label_ids size: ", self.label_ids.size())
        print("\n")

    def padding(self, texts, labels, tags, max_seq_len):
        input_ids, token_type_ids, attention_mask = [], [], []
        for text in texts:
            res = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_seq_len, pad_to_max_length=True)
            input_ids.append(res['input_ids'])
            token_type_ids.append(res['token_type_ids'])
            attention_mask.append(res['attention_mask'])

        label_ids = [label2idx[l] for l in labels]
        tag_ids = []
        for tag in tags:
            if len(tag) <= max_seq_len-2:
                tmp_id = [pad_idx] + [tag2idx[t] for t in tag] + [pad_idx] + [pad_idx]*(max_seq_len-2-len(tag))
            else:
                tmp_id = [pad_idx] + [tag2idx[t] for t in tag[:(max_seq_len-2)]] + [pad_idx]
            tag_ids.append(tmp_id)

        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), \
               torch.tensor(tag_ids), torch.tensor(label_ids)

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, item):
        return self.input_ids[item], self.attention_mask[item], self.token_type_ids[item], \
               self.tag_ids[item], self.label_ids[item]