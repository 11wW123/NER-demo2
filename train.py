import argparse
import torch
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, get_linear_schedule_with_warmup)
from tqdm import tqdm
import swanlab
from config import Config
from dataset import (NERDataset, collect_labels, collate_fn)
from model import BertForNER
from evaluate import (evaluate_predictions)
from utils import (set_seed, save_json, decode_predictions)


def evaluate(model, loader, device, id2label):
    model.eval()

    total_loss = 0.0
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, logits = model(input_ids, attention_mask, labels)

            total_loss += loss.item()

            y_true, y_pred = decode_predictions(logits, labels, id2label)
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)

    metrics = evaluate_predictions(y_true_all, y_pred_all)

    metrics["loss"] = (total_loss / max(len(loader), 1))

    return metrics, y_true_all, y_pred_all


def main(args):
    cfg = Config(args.config)

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Model: {cfg.model_name}")
    print(f"Dataset: {cfg.dataset}")

    swanlab.init(
        project="NER-demo2",
        experiment_name=(
            f"{cfg.dataset}-"f"{cfg.model_name.split('/')[-1]}"
        ),
        description="BERT Chinese NER experiment",
        config={
            "model_name": cfg.model_name,
            "dataset": cfg.dataset,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "warmup_ratio": cfg.warmup_ratio,
            "max_length": cfg.max_length,
            "seed": cfg.seed
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    labels = collect_labels(
        [
            cfg.train_path,
            cfg.dev_path,
            cfg.test_path,
        ]
    )

    label2id = {
        label: i
        for i, label in enumerate(labels)
    }

    id2label = {
        i: label
        for label, i in label2id.items()
    }

    print(f"Labels ({len(labels)}): {labels}")

    train_set = NERDataset(cfg.train_path, tokenizer, label2id, cfg.max_length)
    dev_set = NERDataset(cfg.dev_path, tokenizer, label2id, cfg.max_length)
    test_set = NERDataset(cfg.test_path, tokenizer, label2id, cfg.max_length)

    collate = lambda batch: collate_fn(batch, tokenizer)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate
    )

    model = BertForNER(
        model_name=cfg.model_name,
        num_labels=len(labels)
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    total_steps = (len(train_loader) * cfg.epochs)

    warmup_steps = int(total_steps * cfg.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    save_dir = cfg.save_dir

    history = {
        "train_loss": [],
        "dev_loss": [],
        "dev_precision": [],
        "dev_recall": [],
        "dev_f1": []
    }

    best_f1 = -1.0

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0

        progress = tqdm(
            train_loader,
            desc = f"Epoch "f"{epoch + 1}"
        )

        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_tensor = batch["labels"].to(device)

            optimizer.zero_grad()

            loss, _ = model(input_ids, attention_mask, labels_tensor)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            scheduler.step()

            total_loss += loss.item()

            current_lr = scheduler.get_last_lr()[0]
            swanlab.log(
                {
                    "train/learning_rate": float(current_lr),
                }
            )

        train_loss = (total_loss / max(len(train_loader), 1))

        dev_metrics, _, _ = evaluate(model, dev_loader, device, id2label)

        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_metrics["loss"])
        history["dev_precision"].append(dev_metrics["precision"])
        history["dev_recall"].append(dev_metrics["recall"])
        history["dev_f1"].append(dev_metrics["f1"])

        swanlab.log(
            {
                "train/loss": float(train_loss),
                "dev/loss": float(dev_metrics["loss"]),
                "dev/precision": float(dev_metrics["precision"]),
                "dev/recall": float(dev_metrics["recall"]),
                "dev/f1": float(dev_metrics["f1"])
            },
            step=epoch + 1
        )

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"dev_loss={dev_metrics['loss']:.4f}, "
            f"dev_precision={dev_metrics['precision']:.4f}, "
            f"dev_recall={dev_metrics['recall']:.4f}, "
            f"dev_f1={dev_metrics['f1']:.4f}\n"
        )

        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            torch.save(
                model.state_dict(),
                save_dir / "best_model.pt"
            )

    save_json(
        history,
        save_dir / "history.json"
    )

    model.load_state_dict(
        torch.load(
            save_dir / "best_model.pt",
            map_location=device
        )
    )

    test_metrics, y_true, y_pred = evaluate(
        model,
        test_loader,
        device,
        id2label
    )

    print(
        f"Precision: {test_metrics['precision']:.4f}\n"
        f"Recall:    {test_metrics['recall']:.4f}\n"
        f"F1:        {test_metrics['f1']:.4f}\n"
        f"Loss:      {test_metrics['loss']:.4f}"
    )

    swanlab.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=[
            "configs/bert_weibo.json",
            "configs/bert_msra.json",
            "configs/wwm_weibo.json",
            "configs/wwm_msra.json",
        ],
        required=True
    )
    args = parser.parse_args()
    main(args)