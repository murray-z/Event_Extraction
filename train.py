# -*- coding: utf-8 -*-


import os
import argparse
from data_helper import *
from model import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report
from seqeval.metrics import accuracy_score as seq_accuracy_score
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import classification_report as seq_classification_report


device = config["device"]

def criterion(label_logits, tag_logits, true_labels, true_tags):
    label_loss = F.cross_entropy(label_logits, true_labels)
    tag_loss = F.cross_entropy(tag_logits.view(-1, tag_logits.size(2)),
                               true_tags.view(-1), ignore_index=pad_idx)
    return config["alpha1"]*label_loss + config["alpha2"]*tag_loss


def calculate(true_label_all, pred_label_all, true_tag_all, pred_tag_all):
    try:
        acc_label = accuracy_score(true_label_all, pred_label_all)
        f1_label = f1_score(true_label_all, pred_label_all, average="micro")
        acc_tag = seq_accuracy_score(true_tag_all, pred_tag_all)
        f1_tag = seq_f1_score(true_tag_all, pred_tag_all, average="micro")
        table_label = classification_report(true_label_all, pred_label_all)
        table_tag = seq_classification_report(true_tag_all, pred_tag_all)
    except Exception as e:
        print("Error: ", str(e))
        return 0, 0, "", 0, 0, ""
    return acc_label, f1_label, table_label, acc_tag, f1_tag, table_tag


def dev(model, dev_data_loader):
    model.eval()
    dev_loss = 0.
    num_step = 0.
    true_label_all = []
    pred_label_all = []
    true_tag_all = []
    pred_tag_all = []
    with torch.no_grad():
        for i, batch in enumerate(dev_data_loader, start=1):
            batch = [d.to(device) for d in batch]
            input_ids = batch[0]
            attention_mask = batch[1]
            token_type_ids = batch[2]
            true_tags = batch[3]
            true_labels = batch[4]
            label_logits, tag_logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(label_logits, tag_logits, true_labels, true_tags)
            dev_loss += loss.item()
            num_step += 1

            # 数据转移到cpu
            true_tags = true_tags.detach().cpu()
            true_labels = true_labels.detach().cpu()
            label_logits = label_logits.detach().cpu()
            tag_logits = tag_logits.detach().cpu()

            # 标签
            label_logits = torch.argmax(label_logits, dim=1)
            pred_label_all.extend([idx2label[idx.item()] for idx in label_logits])
            true_label_all.extend([idx2label[idx.item()] for idx in true_labels])

            # 处理tag
            tag_logits = tag_logits.view(-1, tag_logits.size(2))
            true_tags = true_tags.view(-1)
            tag_logits = torch.argmax(tag_logits, 1)
            filter_idx = true_tags != pad_idx
            true_tags = true_tags[filter_idx]
            tag_logits = tag_logits[filter_idx]
            true_tag_all.extend([idx2tag[idx.item()] for idx in true_tags])
            pred_tag_all.extend([idx2tag[idx.item()] for idx in tag_logits])

    loss = dev_loss/num_step
    acc_label, f1_label, table_label, acc_tag, f1_tag, table_tag = \
    calculate(true_label_all, pred_label_all, true_tag_all, pred_tag_all)
    return loss, f1_label, acc_label, f1_tag, acc_tag, table_label, table_tag


def final_test(model, model_path, data_loader):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    loss, f1_label, acc_label, f1_tag, acc_tag, table_label, table_tag = dev(model, data_loader)
    return loss, f1_label, acc_label, f1_tag, acc_tag, table_label, table_tag


def main():
    # 参数
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]

    # 初始化模型
    model = EventExtraction(config)
    if fine_tuning and os.path.exists(config["save_model_path"]):
        model.load_state_dict(torch.load(config["save_model_path"]))
    model.to(device)

    # 加载数据
    train_data_loader = DataLoader(EventDataSet(train_data_path, config), batch_size=batch_size, shuffle=True,
                               num_workers=10)
    dev_data_loader = DataLoader(EventDataSet(dev_data_path, config), batch_size=batch_size, shuffle=False,
                             num_workers=10)
    test_data_loader = DataLoader(EventDataSet(test_data_path, config), batch_size=batch_size, shuffle=False,
                              num_workers=10)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # 开始训练
    best_dev_f1 = float('-inf')

    for epoch in range(1, epochs+1):
        model.train()
        num_step = 0.
        train_loss = 0.

        for i, batch in enumerate(train_data_loader, start=1):
            optimizer.zero_grad()
            batch = [d.to(device) for d in batch]
            input_ids = batch[0]
            attention_mask = batch[1]
            token_type_ids = batch[2]
            true_tags = batch[3]
            true_labels = batch[4]
            label_logits, tag_logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(label_logits, tag_logits, true_labels, true_tags)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_step += 1

            # 数据转移到cpu
            true_tags = true_tags.detach().cpu()
            true_labels = true_labels.detach().cpu()
            label_logits = label_logits.detach().cpu()
            tag_logits = tag_logits.detach().cpu()


            # 标签
            label_logits = torch.argmax(label_logits, dim=1)
            pred_label_all = [idx2label[idx.item()] for idx in label_logits]
            true_label_all = [idx2label[idx.item()] for idx in true_labels]

            # 处理tag
            tag_logits = tag_logits.view(-1, tag_logits.size(2))
            true_tags = true_tags.view(-1)
            tag_logits = torch.argmax(tag_logits, 1)
            filter_idx = true_tags != pad_idx
            true_tags = true_tags[filter_idx]
            tag_logits = tag_logits[filter_idx]

            true_tag_all = [idx2tag[idx.item()] for idx in true_tags]
            pred_tag_all = [idx2tag[idx.item()] for idx in tag_logits]

            if i % 100 == 0:
                acc_label, f1_label, table_label, acc_tag, f1_tag, table_tag = \
                    calculate(true_label_all, pred_label_all, true_tag_all, pred_tag_all)
                print("TRAIN: Epoch:{} step:{:0>4d} label_acc:{:.4f} label_f1:{:.4f}"
                      " tag_acc:{:.4f} tag_f1:{:.4f} loss:{:.4f}".format(
                    epoch, i, acc_label, f1_label, acc_tag, f1_tag, train_loss/num_step))

        # DEV
        loss, f1_label, acc_label, f1_tag, acc_tag, table_label, table_tag = dev(model, dev_data_loader)
        print("DEV: Epoch:{} label_acc:{:.4f} label_f1:{:.4f} tag_acc:{:.4f} tag_f1:{:.4f} loss:{:.4f}".format(
            epoch, acc_label, f1_label, acc_tag, f1_tag, loss))

        print(table_label)
        print(table_tag)

        if f1_tag > best_dev_f1:
            best_dev_f1 = f1_tag
            torch.save(model.state_dict(), config["save_model_path"])

        scheduler.step()

    # TEST
    loss, f1_label, acc_label, f1_tag, acc_tag, table_label, table_tag = \
        final_test(model, config["save_model_path"], test_data_loader)
    print("====TEST====")
    print(table_label)
    print(table_tag)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--fine_tuning", default=False, type=bool, dest="如果已经有模型，是否继续训练")
    args = argparser.parse_args()
    fine_tuning = args.fine_tuning
    main()