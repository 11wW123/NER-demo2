"""
将BIO格式标签转换为实体集合
每个实体表示为 (实体类型, 起始位置, 结束位置)
"""
def find_entities(labels):
    entities = set()
    start = None
    entity_type = None

    for i, label in enumerate(labels):
        if label == "O":
            if start is not None:
                entities.add((entity_type, start, i - 1))
                start = None
                entity_type = None

        elif label.startswith("B-"):
            # 可能两个实体连续出现
            if start is not None:
                entities.add((entity_type, start, i - 1))
            start = i
            entity_type = label[2:]

        elif label.startswith("I-"):
            if start is None:
                start = i
                entity_type = label[2:]

    # 处理句子末尾的实体
    if start is not None:
        entities.add((entity_type, start, len(labels) - 1))

    return entities


def evaluate_predictions(y_true, y_pred):
    tp = 0
    fp = 0
    fn = 0

    for true_labels, pred_labels in zip(y_true, y_pred):
        true_entities = find_entities(true_labels)
        pred_entities = find_entities(pred_labels)

        tp += len(true_entities & pred_entities)
        fp += len(pred_entities - true_entities)
        fn += len(true_entities - pred_entities)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}