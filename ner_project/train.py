import os
import shutil
import torch
import numpy as np
import pandas as pd
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
    AutoConfig
)
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import evaluate

# -------------------------
# 0. TEMİZLİK VE RAPOR
# -------------------------
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("model")
RESULTS_DIR = Path("results")
LOG_DIR = Path("logs")

def cleanup():
    print("Temizlik yapılıyor...")
    for d in [MODEL_DIR, RESULTS_DIR, LOG_DIR]:
        if os.path.exists(d): shutil.rmtree(d)
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".png") or f == "metrics.txt": os.remove(OUTPUT_DIR / f)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

cleanup()

# -------------------------
# 1. AYARLAR
# -------------------------
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
labels = [
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
    "B-DATE", "I-DATE", "B-MONEY", "I-MONEY", "B-PERCENT", "I-PERCENT"
]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

# -------------------------
# 2. VERİ YÜKLEME
# -------------------------
def read_conll(path):
    sentences, tags = [], []
    if not os.path.exists(path): raise FileNotFoundError(path)
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

train_s, train_t = read_conll("data/hybrid/train.txt")
val_s, val_t = read_conll("data/hybrid/dev.txt")

# -------------------------
# 3. TOKENIZATION
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(sentences, tags):
    tokenized_inputs = tokenizer(sentences, truncation=True, is_split_into_words=True, padding=True)
    labels_encoded = []
    for i, label in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None: label_ids.append(-100)
            else: label_ids.append(label2id[label[word_idx]])
            previous_word_idx = word_idx
        labels_encoded.append(label_ids)
    tokenized_inputs["labels"] = labels_encoded
    return tokenized_inputs

train_tokenized = tokenize_and_align_labels(train_s, train_t)
val_tokenized = tokenize_and_align_labels(val_s, val_t)

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, data): self.data = data
    def __getitem__(self, idx): return {k: torch.tensor(v[idx]) for k, v in self.data.items()}
    def __len__(self): return len(self.data["input_ids"])

# -------------------------
# 4. MODEL VE EĞİTİM (GÜÇLÜ REGULARİZASYON)
# -------------------------
config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id)
# Dropout artırarak overfitting önleyelim
config.hidden_dropout_prob = 0.2
config.attention_probs_dropout_prob = 0.2

model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=config)

metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels_ids = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [[labels[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels_ids)]
    true_labels = [[labels[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels_ids)]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    flat_preds = [item for sublist in true_predictions for item in sublist]
    flat_labels = [item for sublist in true_labels for item in sublist]
    return {
        "precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"],
        "weighted_f1": f1_score(flat_labels, flat_preds, average='weighted', zero_division=0)
    }

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16, # Batch size artırıldı
    num_train_epochs=15,
    weight_decay=0.1, # Güçlü weight decay
    warmup_ratio=0.1,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=NERDataset(train_tokenized),
    eval_dataset=NERDataset(val_tokenized),
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("Eğitim başlıyor...")
trainer.train()

# -------------------------
# 5. ANALİZ
# -------------------------
predictions, labels_ids, metrics = trainer.predict(NERDataset(val_tokenized))
predictions = np.argmax(predictions, axis=2)
y_true, y_pred = [], []
for i in range(len(labels_ids)):
    for j in range(len(labels_ids[i])):
        if labels_ids[i][j] != -100:
            y_true.append(labels_ids[i][j]); y_pred.append(predictions[i][j])

# Metrikleri Kaydet
with open(OUTPUT_DIR / "metrics.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {metrics['test_accuracy']:.4f}\nF1: {metrics['test_f1']:.4f}\nWeighted F1: {metrics['test_weighted_f1']:.4f}\n")

# Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
plt.savefig(OUTPUT_DIR / "confusion_matrix.png")

model.save_pretrained(MODEL_DIR); tokenizer.save_pretrained(MODEL_DIR)
print("İşlem tamamlandı.")
