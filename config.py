# -*- coding: utf-8 -*-
import torch



config = {
    "max_seq_len": 128,
    "batch_size": 32,
    "lr": 5e-5,
    "epochs": 5,
    "class_num": 2,  # 事件类别数
    "tag_num": 6,    # 序列标注标签数
    "alpha1": 1.,    # 分类损失权重
    "alpha2": 1.,    # 序列标注损失权重
    "model_name": "hfl/chinese-bert-wwm-ext",  # 预训练模型名称
    "save_model_path": "./model/base.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}