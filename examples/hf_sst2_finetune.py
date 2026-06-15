"""
Fine-tune BERT-base on SST-2 with PsiLogic through the HuggingFace Trainer.

    pip install psilogic transformers datasets
    python examples/hf_sst2_finetune.py --max-steps 500
"""

from __future__ import annotations

import argparse

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

from psilogic.integrations.hf import psilogic_trainer_class


def main() -> None:
    parser = argparse.ArgumentParser(description="PsiLogic + HF Trainer SST-2 example")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", default="./results/hf_sst2")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    dataset = load_dataset("glue", "sst2")

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=128, padding="max_length")

    dataset = dataset.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    dataset = dataset.rename_column("label", "labels")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=1e-2,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="no",
        report_to=[],
        seed=42,
    )

    PsiLogicTrainer = psilogic_trainer_class()
    trainer = PsiLogicTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        psilogic_preset="nlp",
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"\nFinal validation metrics: {metrics}")


if __name__ == "__main__":
    main()
