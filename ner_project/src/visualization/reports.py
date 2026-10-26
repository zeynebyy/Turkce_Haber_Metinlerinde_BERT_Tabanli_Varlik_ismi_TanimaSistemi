import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from pathlib import Path
from config import Config

def generate_dataset_reports(conll_path, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    words, tags = [], []
    with open(conll_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                words.append(parts[0])
                tags.append(parts[1])
    
    # 1. Label Imbalance
    tag_counts = Counter([t for t in tags if t != "O"])
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(tag_counts.keys()), y=list(tag_counts.values()), palette="viridis")
    plt.title("Entity Label Distribution")
    plt.savefig(out / "label_distribution.png")
    
    # 2. Token Diversity (Top words)
    word_counts = Counter([w.lower() for w in words if len(w) > 2])
    top_words = dict(word_counts.most_common(20))
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(top_words.keys()), y=list(top_words.values()), palette="magma")
    plt.xticks(rotation=45)
    plt.title("Top 20 Tokens (Diversity)")
    plt.savefig(out / "token_diversity.png")
    
    print(f"Reports generated in {output_dir}")

if __name__ == "__main__":
    generate_dataset_reports("data/processed/train.txt", "outputs/charts")
