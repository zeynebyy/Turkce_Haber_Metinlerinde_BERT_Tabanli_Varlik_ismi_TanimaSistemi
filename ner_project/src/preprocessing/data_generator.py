import random
import os
import json
from pathlib import Path
from collections import defaultdict

class AdvancedDataGenerator:
    def __init__(self, config):
        self.config = config
        self.initialize_data()
        
    def initialize_data(self):
        # 1. Gelişmiş Varlık Listeleri (Multi-token ve Tekli)
        self.persons = [
            "Recep Tayyip Erdoğan", "Özgür Özel", "Ekrem İmamoğlu", "Mansur Yavaş", "Ali Koç",
            "Meral Akşener", "Devlet Bahçeli", "Hakan Fidan", "Mehmet Şimşek", "Ali Yerlikaya",
            "Aziz Sancar", "Orhan Pamuk", "Elon Musk", "Joe Biden", "Vladimir Putin",
            "İlber Ortaylı", "Celal Şengör", "Fatih Terim", "Arda Güler", "Sultan Süleyman"
        ]
        
        self.orgs = [
            "İstanbul Teknik Üniversitesi", "Orta Doğu Teknik Üniversitesi", "NATO", 
            "Avrupa Birliği Komisyonu", "Türkiye Büyük Millet Meclisi", "TBMM",
            "Türk Hava Yolları", "Merkez Bankası", "TCMB", "ASELSAN", "Baykar Teknoloji",
            "Dünya Sağlık Örgütü", "Birleşmiş Milletler", "HaberTürk", "Anadolu Ajansı",
            "Koç Holding", "Sabancı Vakfı", "Borsa İstanbul", "BIST 100", "Trendyol"
        ]
        
        self.locations = [
            "Ankara", "İstanbul", "İzmir", "Diyarbakır", "Trabzon", "Gaziantep", 
            "Ege Denizi", "Marmara Bölgesi", "Kızılay Meydanı", "Taksim", "Londra", 
            "Washington DC", "Orta Doğu", "Güneydoğu Anadolu", "Kıbrıs", "Azerbaycan"
        ]
        
        # Gerçek Tarihler vs. Belirsiz İfadeler
        self.real_dates = [
            "12 Mayıs 2024", "önümüzdeki Pazartesi", "geçen hafta", "29 Ekim", 
            "15 Temmuz akşamı", "saat 14:30'da", "2025 yılı başında", "yarın sabah"
        ]
        self.temporal_noise = [
            "uzun vadede", "ilerleyen süreçte", "genel olarak", "kısa sürede", 
            "vakti geldiğinde", "zaman zaman", "her zaman", "hiçbir zaman"
        ]
        
        # Finansal Veriler
        self.money = [
            "100 milyon TL", "50 milyar dolar", "250 bin euro", "yüklü miktarda para",
            "rekor ceza", "15 TL", "0.50 kuruş", "trilyon dolarlık bütçe"
        ]
        
        self.percentages = [
            "%45 faiz", "yüzde 20", "yüzde onluk artış", "%0.5", "yüzde yüze yakın"
        ]

    # --- Türkçe Ek Motoru ---
    def inflect(self, word, case):
        back_vowels = "aıou"
        front_vowels = "eiöü"
        hard_consonants = "fstkçşhp"
        
        lv = "a"
        for ch in reversed(word.lower()):
            if ch in back_vowels + front_vowels:
                lv = ch
                break
        
        is_hard = word[-1].lower() in hard_consonants
        
        if case == "locative": # -da/de
            d = "t" if is_hard else "d"
            v = "a" if lv in back_vowels else "e"
            return f"{word}'{d}{v}"
        elif case == "genitive": # -ın/in/un/ün
            if lv in "aı": v = "ı"
            elif lv in "ei": v = "i"
            elif lv in "ou": v = "u"
            else: v = "ü"
            if word[-1].lower() in back_vowels + front_vowels: return f"{word}'n{v}n"
            return f"{word}'{v}n"
        elif case == "dative": # -a/e
            v = "a" if lv in back_vowels else "e"
            if word[-1].lower() in back_vowels + front_vowels: return f"{word}'y{v}"
            return f"{word}'{v}"
        return word

    def get_template_sentence(self):
        p = random.choice(self.persons)
        o = random.choice(self.orgs)
        l = random.choice(self.locations)
        d = random.choice(self.real_dates)
        tn = random.choice(self.temporal_noise)
        m = random.choice(self.money)
        pct = random.choice(self.percentages)
        
        templates = [
            # Siyaset/Diplomasi
            (f"{p}, {self.inflect(l, 'locative')} yaptığı basın açıklamasında {o} ile iş birliğinin süreceğini belirtti.", [(p, "PER"), (l, "LOC"), (o, "ORG")]),
            (f"{o} heyeti, {self.inflect(p, 'genitive')} davetiyle {d} tarihinde toplandı.", [(o, "ORG"), (p, "PER"), (d, "DATE")]),
            
            # Ekonomi/Finans
            (f"{o}, {d} %15 büyüme beklerken, {m} değerinde yeni bir fon ayırdı.", [(o, "ORG"), (d, "DATE"), (m, "MONEY")]),
            (f"Dolar kuru ve altın fiyatları {tn} stabil kalacak ancak faizler {pct} arttı.", [(pct, "PERCENT")]),
            
            # Savunma/Teknoloji
            (f"{o} mühendisleri {self.inflect(l, 'locative')} {d} yeni bir sistem test etti.", [(o, "ORG"), (l, "LOC"), (d, "DATE")]),
            (f"{p}, savunma sanayi alanındaki gelişmeleri {tn} takip edeceklerini söyledi.", [(p, "PER")]),
            
            # Karışık (Daha karmaşık yapılar)
            (f"{d} {self.inflect(l, 'dative')} gitmesi planlanan {p}, {o} tarafından düzenlenen {m} bütçeli törene katılamayacak.", [(d, "DATE"), (l, "LOC"), (p, "PER"), (o, "ORG"), (m, "MONEY")])
        ]
        
        # Template sayısını artırmak için burada döngülerle 100+ varyasyon üretilebilir
        # Ancak mantık olarak bu yapıyı kullanacağız.
        return random.choice(templates)

    def generate_negatives(self):
        neutrals = [
            "Bugün hava oldukça güzel ve güneşli.",
            "Yemekten sonra biraz yürüyüş yapmak iyi gelir.",
            "Kitap okumanın zihinsel gelişime faydası çoktur.",
            "Yeni bir hobi edinmek hayat kalitesini artırır.",
            "Teknolojik gelişmeleri yakından takip etmek gerekiyor."
        ]
        return random.choice(neutrals), []

    def tokenize_and_label(self, sentence, entities):
        # Suffix handling: Kelimeyi ekinden ayırmadan tokenize et
        # Ama etiketlerken sadece kökü değil, tüm kelimeyi (Ankara'da) etiketle
        for ch in [".", ",", "!", "?", ";", ":"]:
            sentence = sentence.replace(ch, f" {ch}")
            
        tokens = sentence.split()
        tags = ["O"] * len(tokens)
        
        for entity_text, entity_type in entities:
            # Entity'nin ek almış hallerini de kontrol et
            # Örn: "Ankara" entities içindeyse, tokens içindeki "Ankara'da" yı bulmalı
            entity_tokens = entity_text.split()
            
            for i in range(len(tokens) - len(entity_tokens) + 1):
                match = True
                for j in range(len(entity_tokens)):
                    # Kısmi eşleşme (Ekleri hesaba katmak için)
                    if not tokens[i+j].startswith(entity_tokens[j]):
                        match = False
                        break
                
                if match:
                    tags[i] = f"B-{entity_type}"
                    for j in range(1, len(entity_tokens)):
                        tags[i+j] = f"I-{entity_type}"
        
        return list(zip(tokens, tags))

    def run(self, count=10000):
        dataset = []
        for _ in range(count):
            if random.random() < self.config.NEGATIVE_RATIO:
                sent, ents = self.generate_negatives()
            else:
                sent, ents = self.get_template_sentence()
            
            labeled = self.tokenize_and_label(sent, ents)
            dataset.append(labeled)
        return dataset

def save_conll(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for sent in data:
            for word, tag in sent:
                f.write(f"{word} {tag}\n")
            f.write("\n")
