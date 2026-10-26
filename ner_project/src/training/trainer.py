import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    AutoConfig,
    TrainerCallback
)
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import evaluate
from config import Config

class VisualizationCallback(TrainerCallback):
    """Her epoch sonunda grafikleri guncelleyen callback."""
    def __init__(self, chart_dir):
        self.chart_dir = Path(chart_dir)
        self.chart_dir.mkdir(parents=True, exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        self.f1_scores = []
        self.accuracies = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            self.val_losses.append(logs["eval_loss"])
            self.f1_scores.append(logs["eval_f1"])
            self.accuracies.append(logs["eval_accuracy"])
            self.plot_metrics()

    def plot_metrics(self):
        plt.figure(figsize=(15, 5))
        
        # Loss Plot
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.title("Loss Curve")
        plt.legend()
        
        # F1 Plot
        plt.subplot(1, 3, 2)
        plt.plot(self.f1_scores, label="F1 Score", color="green")
        plt.title("F1 Evolution")
        plt.legend()
        
        # Accuracy Plot
        plt.subplot(1, 3, 3)
        plt.plot(self.accuracies, label="Accuracy", color="orange")
        plt.title("Accuracy Evolution")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.chart_dir / "training_metrics.png")
        plt.close()

class ProfessionalNERit:
    def __init__(self):
        self.config = Config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        self.metric = evaluate.load("seqeval")
        
    def read_conll(self, path):
        sentences, tags = [], []
        with open(path, "r", encoding="utf-8") as f:
            words, labels_list = [], []
            for line in f:
                if line.strip() == "":
                    if words:
                        sentences.append(words); tags.append(labels_list)
                        words, labels_list = [], []
                else:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        words.append(parts[0]); labels_list.append(parts[1])
        return sentences, tags

    def tokenize_and_align_labels(self, sentences, tags):
        tokenized_inputs = self.tokenizer(sentences, truncation=True, is_split_into_words=True, padding=True)
        labels_encoded = []
        for i, label in enumerate(tags):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None: label_ids.append(-100)
                else: label_ids.append(self.config.LABEL2ID[label[word_idx]])
            labels_encoded.append(label_ids)
        tokenized_inputs["labels"] = labels_encoded
        return tokenized_inputs

    def compute_metrics(self, p):
        predictions, labels_ids = p
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = [[self.config.LABELS[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels_ids)]
        true_labels = [[self.config.LABELS[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels_ids)]
        
        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def train(self):
        train_s, train_t = self.read_conll(Path(self.config.DATA_PROCESSED) / "train.txt")
        val_s, val_t = self.read_conll(Path(self.config.DATA_PROCESSED) / "dev.txt")
        
        train_tokenized = self.tokenize_and_align_labels(train_s, train_t)
        val_tokenized = self.tokenize_and_align_labels(val_s, val_t)
        
        class NERDataset(torch.utils.data.Dataset):
            def __init__(self, data): self.data = data
            def __getitem__(self, idx): return {k: torch.tensor(v[idx]) for k, v in self.data.items()}
            def __len__(self): return len(self.data["input_ids"])

        model_config = AutoConfig.from_pretrained(self.config.MODEL_NAME, num_labels=len(self.config.LABELS), id2label=self.config.ID2LABEL, label2id=self.config.LABEL2ID)
        model_config.hidden_dropout_prob = self.config.DROPOUT_PROB
        
        model = AutoModelForTokenClassification.from_pretrained(self.config.MODEL_NAME, config=model_config)
        
        args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.config.LEARNING_RATE,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            num_train_epochs=self.config.NUM_EPOCHS,
            weight_decay=self.config.WEIGHT_DECAY,
            logging_dir="./logs",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=torch.cuda.is_available()
        )
        
        trainer = Trainer(
            model=model, args=args,
            train_dataset=NERDataset(train_tokenized),
            eval_dataset=NERDataset(val_tokenized),
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForTokenClassification(self.tokenizer),
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=self.config.EARLY_STOPPING_PATIENCE),
                VisualizationCallback(self.config.CHART_DIR)
            ]
        )
        
        print("Training starting...")
        trainer.train()
        
        model.save_pretrained(self.config.MODEL_SAVE_PATH)
        self.tokenizer.save_pretrained(self.config.MODEL_SAVE_PATH)
        print("Model saved to", self.config.MODEL_SAVE_PATH)

if __name__ == "__main__":
    ner = ProfessionalNERit()
    ner.train()
