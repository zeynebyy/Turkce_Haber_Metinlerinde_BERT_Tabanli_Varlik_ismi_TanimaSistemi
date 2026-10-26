from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# -------------------------
# OUTPUT KLASÖRÜ
# -------------------------
os.makedirs("outputs", exist_ok=True)

# -------------------------
# ETİKETLER
# -------------------------
labels = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC"
]

label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

# -------------------------
# VERİ OKUMA
# -------------------------
def read_data(path):
    sentences, tags = [], []
    with open(path, "r", encoding="utf-8") as f:
        w, t = [], []
        for line in f:
            if line.strip() == "":
                if w:
                    sentences.append(w)
                    tags.append(t)
                w, t = [], []
            else:
                word, tag = line.strip().split()
                w.append(word)
                t.append(tag)
    return sentences, tags

train_s, train_t = read_data("data/generated/train.txt")
val_s, val_t = read_data("data/generated/dev.txt")

# -------------------------
# TOKENIZER
# -------------------------
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

def tokenize(sentences, tags):
    enc = tokenizer(sentences, is_split_into_words=True, padding=True, truncation=True)
    labels_enc = []

    for i, label in enumerate(tags):
        word_ids = enc.word_ids(batch_index=i)
        label_ids = []
        prev = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(label2id[label[word_idx]])
            prev = word_idx

        labels_enc.append(label_ids)

    enc["labels"] = labels_enc
    return enc

train_enc = tokenize(train_s, train_t)
val_enc = tokenize(val_s, val_t)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, enc):
        self.enc = enc

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.enc.items()}

    def __len__(self):
        return len(self.enc["input_ids"])

train_dataset = Dataset(train_enc)
val_dataset = Dataset(val_enc)

# -------------------------
# MODEL
# -------------------------
model = AutoModelForTokenClassification.from_pretrained(
    "dbmdz/bert-base-turkish-cased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# -------------------------
# TRAINING (UYUMLU)
# -------------------------
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

# -------------------------
# LOSS GRAFİĞİ
# -------------------------
logs = trainer.state.log_history

loss, steps = [], []
for log in logs:
    if "loss" in log:
        loss.append(log["loss"])
        steps.append(log["step"])

plt.plot(steps, loss)
plt.title("Training Loss")
plt.savefig("outputs/loss.png")
plt.close()

# -------------------------
# CONFUSION + F1
# -------------------------
pred = trainer.predict(val_dataset)
preds = pred.predictions.argmax(-1)
labels_true = pred.label_ids

y_true, y_pred = [], []

for i in range(len(labels_true)):
    for j in range(len(labels_true[i])):
        if labels_true[i][j] != -100:
            y_true.append(labels_true[i][j])
            y_pred.append(preds[i][j])

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

f1 = f1_score(y_true, y_pred, average="weighted")

with open("outputs/metrics.txt", "w") as f:
    f.write(f"F1 Score: {f1}")

# -------------------------
# MODEL KAYDET
# -------------------------
model.save_pretrained("model")
tokenizer.save_pretrained("model")

print("✅ Eğitim tamamlandı, outputs klasörüne kaydedildi!")