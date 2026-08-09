"""
Fine-tunes DistilBERT on the Rotten Tomatoes sentiment dataset.
Logs metrics/params/model to MLflow.
Run: python src/train.py
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/sentiment-distilbert"


def load_data():
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }


def main():
    print("Loading data...")
    train_dataset, val_dataset = load_data()

    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir="./training_output",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",  # we'll log manually to mlflow
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    mlflow.set_experiment("sentiment-classifier")

    with mlflow.start_run():
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "learning_rate": training_args.learning_rate,
            "epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
        })

        print("Starting training...")
        trainer.train()

        print("Evaluating...")
        eval_results = trainer.evaluate()
        print(eval_results)

        mlflow.log_metrics({
            "eval_accuracy": eval_results["eval_accuracy"],
            "eval_f1": eval_results["eval_f1"],
            "eval_precision": eval_results["eval_precision"],
            "eval_recall": eval_results["eval_recall"],
        })

        print(f"Saving model to {OUTPUT_DIR}")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        mlflow.log_artifacts(OUTPUT_DIR, artifact_path="model")

    print("Done. Run `mlflow ui` to view results.")


if __name__ == "__main__":
    main()