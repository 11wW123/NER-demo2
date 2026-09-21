import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

PAD_LABEL = -100

def read_bio_file(path):
    samples = [] # 保存所有句子
    chars, labels = [], [] # 保存当前句子的字符、BIO标签

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()

            if not line:
                if chars:
                    samples.append((chars, labels))
                    chars, labels = [], []
                continue

            parts = line.split()
            if len(parts) < 2:
                continue
            ch, la = parts[0], parts[-1]
            chars.append(ch)
            labels.append(la)

    # 为了防止文件最后没有空行补一句
    if chars:
        samples.append((chars, labels))

    return samples

"""
收集数据集中出现的全部 BIO 标签
"""
def collect_labels(path):
    labels = set()

    for _, tags in read_bio_file(path):
        labels.update(tags)

    labels = sorted(labels)
    if "O" in labels:
        labels.remove("O")
    # 让O固定为0
    labels = ["O"] + labels
    return labels

class NERDataset(Dataset):
    def __init__(self, path, tokenizer, label2id, max_length=128):
        self.samples = read_bio_file(path)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chars, tags = self.samples[idx]

        enc = self.tokenizer(
            chars,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt"
        )

        word_ids = enc.word_ids()

        label_ids = []

        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(PAD_LABEL)
            # 一个原始字符第一次出现时才放标签
            elif word_id != previous_word_id:
                if word_id < len(tags):
                    label_ids.append(self.label2id[tags[word_id]])
                else:
                    label_ids.append(PAD_LABEL)
            # 一个原始字符对应多个token时 只给第一个token标签
            else:
                label_ids.append(PAD_LABEL)
            previous_word_id = word_id

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }

        return item

def collate_fn(batch, tokenizer):

    input_ids = []
    attention_masks = []
    labels = []
    for item in batch:
        input_ids.append(item["input_ids"])
        attention_masks.append(item["attention_mask"])
        labels.append(item["labels"])

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    labels = pad_sequence(labels, batch_first=True, padding_value=PAD_LABEL)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels
    }