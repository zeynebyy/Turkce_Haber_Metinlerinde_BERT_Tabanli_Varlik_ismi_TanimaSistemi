import re
import torch
from transformers import pipeline
from config import Config
import os


# ==========================================
# TÜRKÇE NER POST-PROCESSING VE REGEX SİSTEMİ
# ==========================================

# 1. PARA REGEX SİSTEMİ (Kullanıcı Talebi)
MONEY_PATTERNS = [
    re.compile(r"\d+\s*dolar", re.IGNORECASE),
    re.compile(r"\d+\s*TL", re.IGNORECASE),
    re.compile(r"\d+\s*euro", re.IGNORECASE),
    re.compile(r"\d+\s*milyon", re.IGNORECASE),
    re.compile(r"\d+\s*milyar", re.IGNORECASE),
    re.compile(r"\d+[\.,]?\d*\s*(dolar|euro|TL|lira|sterlin|milyon|milyar|trilyon)", re.IGNORECASE),
]

# 2. TARİH REGEX SİSTEMİ (Kullanıcı Talebi - Ay İsimleri)
DATE_MONTHS = [
    "ocak", "şubat", "mart", "nisan", "mayıs", "haziran", 
    "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık"
]
DATE_MONTH_PATTERN = re.compile(r"\b(" + "|".join(DATE_MONTHS) + r")\b", re.IGNORECASE)

DATE_PATTERNS = [
    re.compile(r"\d{4}['']?(da|de|ta|te|dan|den|tan|ten|daki|deki)?", re.IGNORECASE),
    re.compile(r"\d{4}\s*yılında?", re.IGNORECASE),
    re.compile(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}", re.IGNORECASE),
    DATE_MONTH_PATTERN
]

# 3. SÜRE KALIPLARI
DURATION_PATTERNS = [
    re.compile(r"\d+\s*(yıl|ay|gün|saat|dakika|hafta|içinde)", re.IGNORECASE),
]

# 4. ÜRÜN/TEKNOLOJİ ANAHTAR KELİMELERİ (Kullanıcı Talebi: iPhone vb.)
PRODUCT_KEYWORDS = {"iphone", "ipad", "macbook", "samsung", "huawei", "modeli", "cihazı"}

# 5. UNVANLAR
TITLE_WORDS = {
    "başkan", "başkanı", "başkanlığı", "cumhurbaşkanı", "başbakan", "bakan", "bakanı", "bakanlığı",
    "genel müdür", "müdür", "direktör", "vali", "kaymakam", "belediye başkanı", "rektör", "dekan",
    "komutan", "general", "amiral", "büyükelçi", "sözcü", "sekreter", "şef", "yetkili"
}

def apply_rule_based_fixes(entity_label: str, word: str) -> str:
    w = word.strip()
    w_lower = w.lower()

    # PARA KONTROLÜ
    if any(p.search(w) for p in MONEY_PATTERNS):
        return "MONEY"
    
    # TARİH KONTROLÜ
    if any(p.search(w) for p in DATE_PATTERNS):
        return "DATE"

    # ÜRÜN KONTROLÜ (Kullanıcı Talebi: iPhone -> URUN)
    if any(pk in w_lower for pk in PRODUCT_KEYWORDS):
        return "URUN"

    # SÜRE KONTROLÜ
    if any(p.search(w) for p in DURATION_PATTERNS):
        return "DURATION"

    # UNVAN FİLTRELEME
    if entity_label in ("ORG", "PER"):
        if w_lower in TITLE_WORDS:
            return "OTHER"

    return entity_label


class AdvancedNERPipeline:
    def __init__(self, model_path="model"):
        self.config = Config()
        # KESİN KURAL: Sadece "ner" pipeline ve simple aggregation
        self.nlp = pipeline(
            "ner",
            model=model_path,
            tokenizer=model_path,
            aggregation_strategy="simple"
        )

    def post_process(self, results, text):
        if not results:
            return []

        # 1. SMART MERGE (Gerekirse bitişik aynı tipleri birleştir)
        pre_merged = []
        if not results: return []
        current = results[0].copy()

        for next_res in results[1:]:
            gap = text[current['end']:next_res['start']]
            same_type = next_res['entity_group'] == current['entity_group']
            close_enough = len(gap) <= 2 
            if same_type and close_enough:
                current['end'] = next_res['end']
                current['score'] = (current['score'] + next_res['score']) / 2
            else:
                pre_merged.append(current)
                current = next_res.copy()
        pre_merged.append(current)

        # 2. CONFIDENCE VE KURAL TABANLI FİLTRELEME
        final_results = []
        seen_indices = set()

        for r in pre_merged:
            # Skoru kontrol et (Config'den gelir)
            if r['score'] < self.config.CONFIDENCE_THRESHOLD:
                # Regex'e takılıyorsa skor düşük olsa da kabul et (özellikle PARA/TARİH için)
                if not any(p.search(text[r['start']:r['end']]) for p in (MONEY_PATTERNS + DATE_PATTERNS)):
                    continue

            # Orijinal metinden kelimeyi tam al ve ekleri temizle
            start, end = r['start'], r['end']
            while end < len(text) and not text[end].isspace() and text[end] not in [".", ",", "!", "?", ";", ":"]:
                end += 1
            
            r['end'] = end
            r['word'] = text[start:end]

            # Kural tabanlı etiket düzeltme
            r['entity_group'] = apply_rule_based_fixes(r['entity_group'], r['word'])

            if r['entity_group'] == "OTHER":
                continue

            # Index çakışmasını önle
            idx_range = tuple(range(r['start'], r['end']))
            if any(i in seen_indices for i in idx_range):
                continue
            for i in idx_range: seen_indices.add(i)

            final_results.append(r)

        return final_results

    def predict(self, text):
        # KESİN KURAL: Metni ASLA değiştirme (no paraphrase)
        raw_results = self.nlp(text)
        processed = self.post_process(raw_results, text)
        return processed
