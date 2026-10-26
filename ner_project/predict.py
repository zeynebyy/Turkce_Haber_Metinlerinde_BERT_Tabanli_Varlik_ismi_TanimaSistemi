import sys
import os
from transformers import pipeline
from colorama import init, Fore, Style
from collections import defaultdict

# Colorama'yı başlat
init(autoreset=True)

# Türkçe etiket haritası
LABEL_MAP = {
    "PER": ("Kişi", Fore.CYAN),
    "ORG": ("Kurum", Fore.YELLOW),
    "LOC": ("Yer", Fore.GREEN),
    "DATE": ("Tarih", Fore.MAGENTA),
    "MONEY": ("Para", Fore.RED),
    "PERCENT": ("Yüzde", Fore.BLUE)
}

def load_model():
    model_path = "model"
    if not os.path.exists(model_path):
        print(f"{Fore.RED}Hata: Model klasörü bulunamadı. Lütfen önce 'train.py' çalıştırın.")
        return None
        
    print(f"{Fore.BLUE}Model yükleniyor, lütfen bekleyin...")
    try:
        nlp = pipeline(
            "ner",
            model=model_path,
            tokenizer=model_path,
            aggregation_strategy="simple"
        )
        return nlp
    except Exception as e:
        print(f"{Fore.RED}Hata: Model yüklenemedi: {e}")
        return None

def display_results(results, text):
    if not results:
        print(f"{Fore.YELLOW}Herhangi bir varlık tespit edilemedi.")
        return

    # 1. Renkli Metin Görünümü
    print("\n" + "="*60)
    print(f"{Fore.WHITE}{Style.BRIGHT}      TÜRKÇE HABER NER - ANALİZ SONUÇLARI")
    print("="*60 + "\n")

    output_text = text
    # Sondan başa doğru değiştiriyoruz ki indisler kaymasın
    for r in sorted(results, key=lambda x: x['start'], reverse=True):
        label_key = r['entity_group']
        label_info = LABEL_MAP.get(label_key, (label_key, Fore.WHITE))
        label_name = label_info[0]
        color = label_info[1]
        
        start, end = r['start'], r['end']
        entity_text = text[start:end]
        
        replacement = f"{color}[{entity_text} ({label_name})]{Style.RESET_ALL}"
        output_text = output_text[:start] + replacement + output_text[end:]

    print(output_text)
    print("\n" + "-"*60)

    # 2. Gruplanmış Liste Görünümü
    grouped = defaultdict(set)
    for r in results:
        label_key = r['entity_group']
        label_name = LABEL_MAP.get(label_key, (label_key, ""))[0]
        grouped[label_name].add(r['word'])

    print(f"{Fore.WHITE}{Style.BRIGHT}Tespit Edilen Varlıklar (Gruplandırılmış):")
    for label, items in grouped.items():
        color = next((v[1] for k, v in LABEL_MAP.items() if v[0] == label), Fore.WHITE)
        print(f"\n{color}{label}:")
        for item in sorted(items):
            print(f" - {item}")
    
    print("\n" + "="*60)

def main():
    nlp = load_model()
    if not nlp:
        return

    print(f"\n{Fore.GREEN}Sistem Hazır! Analiz edilecek haber metnini girin.")
    print(f"{Fore.YELLOW}(Çıkmak için 'q' yazın veya Ctrl+C yapın)")

    try:
        while True:
            text = input(f"\n{Fore.WHITE}{Style.BRIGHT}Haber Metni > {Style.RESET_ALL}").strip()
            if text.lower() == 'q':
                break
            if not text:
                continue
            
            # Uzun metinleri parçalayarak işleme (BERT sınırı 512 token)
            # Pipeline aggregation_strategy="simple" bunu genelde yönetir ama 
            # çok uzun metinlerde dikkatli olunmalı.
            results = nlp(text)
            display_results(results, text)
            
    except KeyboardInterrupt:
        print(f"\n{Fore.BLUE}Program kapatılıyor...")

if __name__ == "__main__":
    main()