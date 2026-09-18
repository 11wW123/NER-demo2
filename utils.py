import json
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 把一个Python对象保存到指定的JSON文件里
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

"""
把模型输出的数字标签转换回BIO标签
"""
def decode_predictions(logits, labels, id2label):
    pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
    label_ids = labels.detach().cpu().tolist()

    y_pred, y_true = [], []

    for pred_row, label_row in zip(pred_ids, label_ids):
        p_seq, t_seq = [], []
        for u, v in zip(pred_row, label_row):
            if v == -100:
                continue
            p_seq.append(id2label[u])
            t_seq.append(id2label[v])
        y_pred.append(p_seq)
        y_true.append(t_seq)

    return y_true, y_pred