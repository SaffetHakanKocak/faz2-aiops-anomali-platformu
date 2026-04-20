import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from logbert_prepare_dataset import main as prepare_dataset


DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
DATASET_CSV = os.path.join(DATA_DIR, "logbert_dataset.csv")
MODEL_DIR = os.path.join(DATA_DIR, "logbert_model")
METRICS_JSON = os.path.join(DATA_DIR, "logbert_metrics.json")

BASE_MODEL = os.environ.get("LOGBERT_BASE_MODEL", "prajjwal1/bert-tiny")
EPOCHS = float(os.environ.get("LOGBERT_EPOCHS", "1"))
BATCH_SIZE = int(os.environ.get("LOGBERT_BATCH_SIZE", "16"))
MAX_ROWS = int(os.environ.get("LOGBERT_MAX_ROWS", "1200"))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    prepare_dataset()

    if not os.path.exists(DATASET_CSV):
        raise RuntimeError(f"Dataset bulunamadı: {DATASET_CSV}")

    df = pd.read_csv(DATASET_CSV)
    if len(df) > MAX_ROWS:
        # Hızlı deneme modu: veri çok büyükse örnekleyip süreyi düşür.
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
    if df.empty or len(df) < 20:
        # Çok az veri varsa sadece bilgi bırakıp çık.
        payload = {
            "status": "insufficient_data",
            "message": "Eğitim için en az 20 satır önerilir",
            "rows": int(len(df)),
        }
        with open(METRICS_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(json.dumps(payload, ensure_ascii=False))
        return

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    except Exception:
        # Bazı ortamlarda fast tokenizer backend'i açılamazsa yavaş tokenizer'a düş.
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True)).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True)).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    args = TrainingArguments(
        output_dir=MODEL_DIR,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=5e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=20,
        report_to=[],
    )

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": test_ds,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    # transformers sürümleri arasında tokenizer/processing_class parametresi değişebiliyor.
    try:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)
    except TypeError:
        try:
            trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
        except TypeError:
            trainer = Trainer(**trainer_kwargs)

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()

