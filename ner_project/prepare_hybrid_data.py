import pandas as pd
import json
import random
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import difflib

# Ayarlar
DATA_DIR = Path("data")
HYBRID_DIR = DATA_DIR / "hybrid"
CSV_PATH = DATA_DIR / "csv_dataset" / "turkish_news_ner_1000.csv"
SYNTHETIC_TRAIN = DATA_DIR / "generated" / "train.txt"
SYNTHETIC_DEV = DATA_DIR / "generated" / "dev.txt"
SYNTHETIC_TEST = DATA_DIR / "generated" / "test.txt"
OUTPUT_DIR = Path("outputs")
STATS_FILE = OUTPUT_DIR / "dataset_stats.txt"

os.makedirs(HYBRID_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_conll(path):
    sentences = []
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        current_sent = []
        for line in f:
            line = line.strip()
            if not line:
                if current_sent:
                    sentences.append(current_sent)
                    current_sent = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    current_sent.append((parts[0], parts[1]))
        if current_sent: sentences.append(current_sent)
    return sentences

def read_csv_ner(path):
    df = pd.read_csv(path)
    sentences = []
    for _, row in df.iterrows():
        try:
            tokens = json.loads(row['tokens'])
            tags = json.loads(row['tags'])
            sent = list(zip(tokens, tags))
            sentences.append(sent)
        except: continue
    return sentences

def get_text(sent):
    return " ".join([w for w, t in sent])

def check_leakage(train, test, threshold=0.85):
    """Train ve Test arasında benzerlik kontrolü yapar."""
    train_texts = [get_text(s) for s in train]
    test_texts = [get_text(s) for s in test]
    
    leaked_indices = []
    # Basit bir set kontrolü (Duplicate)
    train_set = set(train_texts)
    
    clean_test = []
    duplicates = 0
    for i, t in enumerate(test_texts):
        if t in train_set:
            duplicates += 1
        else:
            clean_test.append(test[i])
    
    return clean_test, duplicates

def save_conll(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for sent in data:
            for word, tag in sent:
                f.write(f"{word} {tag}\n")
            f.write("\n")

print("Veriler yukleniyor...")
synthetic_data = read_conll(SYNTHETIC_TRAIN) + read_conll(SYNTHETIC_DEV) + read_conll(SYNTHETIC_TEST)
real_data = read_csv_ner(CSV_PATH)

print(f"Toplam Sentetik: {len(synthetic_data)}")
print(f"Toplam Gercek: {len(real_data)}")

# Karistirma
random.shuffle(synthetic_data)
random.shuffle(real_data)

# Hibrit Havuzu (%70 Sentetik, %30 Gercek)
target_real = len(real_data)
target_synth = int(target_real * 7 / 3)
hybrid_pool = synthetic_data[:target_synth] + real_data

random.shuffle(hybrid_pool)

# Split
n = len(hybrid_pool)
train_raw = hybrid_pool[:int(n*0.8)]
dev_raw = hybrid_pool[int(n*0.8):int(n*0.9)]
test_raw = hybrid_pool[int(n*0.9):]

# Leakage Kontrolü
print("Leakage kontrolü yapiliyor...")
dev_clean, dev_dups = check_leakage(train_raw, dev_raw)
test_clean, test_dups = check_leakage(train_raw, test_raw)

print(f"Temizlenen Duplicate (Dev): {dev_dups}")
print(f"Temizlenen Duplicate (Test): {test_dups}")

save_conll(HYBRID_DIR / "train.txt", train_raw)
save_conll(HYBRID_DIR / "dev.txt", dev_clean)
save_conll(HYBRID_DIR / "test.txt", test_clean)

# Analiz
all_tags = [tag for sent in hybrid_pool for _, tag in sent if tag != "O"]
tag_counts = Counter(all_tags)
all_words = [word.lower() for sent in hybrid_pool for word, _ in sent if len(word) > 2]
word_counts = Counter(all_words)

with open(STATS_FILE, "w", encoding="utf-8") as f:
    f.write("=== PROFESYONEL HIBRIT DATASET ANALIZI ===\n")
    f.write(f"Toplam Ornek: {len(hybrid_pool)}\n")
    f.write(f"Train: {len(train_raw)} | Dev: {len(dev_clean)} | Test: {len(test_clean)}\n")
    f.write(f"Duplicate/Leakage Silinen: {dev_dups + test_dups}\n")
    f.write("\n--- Etiket Dagilimi ---\n")
    for tag, count in tag_counts.most_common():
        f.write(f"{tag}: {count}\n")
    f.write("\n--- En Sık Gecen Kelimeler (Top 20) ---\n")
    for word, count in word_counts.most_common(20):
        f.write(f"{word}: {count}\n")

# Grafikler
plt.figure(figsize=(10, 6))
sns.barplot(x=list(tag_counts.keys()), y=list(tag_counts.values()))
plt.title("Etiket Dağılımı")
plt.xticks(rotation=45)
plt.savefig(OUTPUT_DIR / "label_distribution.png")

print("Veri hazırlama tamamlandı.")
