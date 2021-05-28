# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import BertConfig, BertModel


class EventExtraction(nn.Module):
    def __init__(self, config):
        super(EventExtraction, self).__init__()
        self.bert_config = BertConfig.from_pretrained(config["model_name"])
        self.bert_model = BertModel.from_pretrained(config["model_name"], config=self.bert_config)
        dropout = self.bert_config.hidden_dropout_prob
        self.linear1 = nn.Linear(self.bert_config.hidden_size, config["class_num"])
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(self.bert_config.hidden_size, config["tag_num"])
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert_model(input_ids, attention_mask, token_type_ids)
        # [cls] 用于分类
        pooled_cls = self.dropout1(outputs[1])
        # 用于序列标注，最后一个隐层向量
        last_hidden = self.dropout2(outputs[0])

        logits_label = self.linear1(pooled_cls)
        logits_tag = self.linear2(last_hidden)

        return logits_label, logits_tag
