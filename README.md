# Demo2：中文命名实体识别（NER）

## 📌 项目介绍

本项目基于 PyTorch 和 Hugging Face Transformers 实现中文命名实体识别（Named Entity Recognition, NER）任务。

实验使用：

- 预训练模型：
  - bert-base-chinese
  - hfl/chinese-bert-wwm

- 数据集：
  - Weibo 中文微博实体识别数据集
  - MSRA 中文命名实体识别数据集

通过 BIO 序列标注方式完成中文实体识别任务，并分析不同模型以及参数设置对 NER 性能的影响。


## 🎯 实验目标

- 理解 BERT 在序列标注任务中的应用流程；
- 掌握 BIO 标签处理、Token 与标签对齐、动态 Padding 等数据处理方法；
- 实现完整的 NER 模型训练与评估流程；
- 对比不同预训练模型在不同数据集上的表现。


## 📂 项目结构

```
demo2/

├── configs/
│   ├── bert_weibo.json
│   ├── bert_msra.json
│   ├── wwm_weibo.json
│   └── wwm_msra.json
│
├── data/
│   ├── msra
│   └── weibo
│
├── config.py
├── dataset.py
├── model.py
├── evaluate.py
├── utils.py
├── train.py
└── README.md
```


## 🚀 实验运行

### bert-base-chinese + Weibo

```bash
python train.py --config configs/bert_weibo.json
```

### bert-base-chinese + MSRA

```bash
python train.py --config configs/bert_msra.json
```

### chinese-bert-wwm + Weibo

```bash
python train.py --config configs/wwm_weibo.json
```

### chinese-bert-wwm + MSRA

```bash
python train.py --config configs/wwm_msra.json
```


## 🧠 模型结构

```text
Input Text

    ↓

Tokenizer

    ↓

Token IDs + Attention Mask

    ↓

BERT Encoder

    ↓

Token-level Hidden Representations

    ↓

Dropout

    ↓

Linear Classifier

    ↓

BIO Tag Prediction

    ↓

Cross Entropy Loss
```

模型对每个 Token 进行实体类别预测


## 📊 评价指标

采用实体级评价指标：

- Precision
- Recall
- F1-score

其中：

```
Precision =
预测正确实体数量 / 预测实体总数量

Recall =
预测正确实体数量 / 标注实体总数量

F1 =
2 × Precision × Recall / (Precision + Recall)
```


# 📈 实验结果

## 1. 不同模型与数据集实验结果

| Model | Dataset | Precision | Recall | F1     |
|---|---|-----------|--------|--------|
| bert-base-chinese | Weibo | 0.6699    | 0.6796 | 0.6747 |
| bert-base-chinese | MSRA | 0.9000    | 0.9136 | 0.9067 |
| chinese-bert-wwm | Weibo | 0.6524    | 0.6650 | 0.6587 |
| chinese-bert-wwm | MSRA | 0.9035    | 0.9128 | 0.9081 |


## 2. SwanLab训练过程可视化

本项目使用 SwanLab 记录：

- Train Loss
- Dev Loss
- Precision
- Recall
- F1-score
- Learning Rate

完整实验曲线见：

results_image/

包含：

- bert-base-chinese + Weibo
- bert-base-chinese + MSRA
- chinese-bert-wwm + Weibo
- chinese-bert-wwm + MSRA


## 💾 模型保存

训练过程中，根据 Dev 集 F1 保存最佳模型：

```
outputs/
└── dataset_model/
    ├── best_model.pt
    └── history.json
```


## 📝 实验总结

通过本实验：

- 掌握了 BERT 在中文 NER 任务中的应用方法；
- 理解了 BIO 序列标注方式；
- 学习了 Token 与标签对齐处理流程；
- 实现了完整的 NER 模型训练与评估流程；
- 分析了不同预训练模型以及参数设置对实体识别性能的影响。
