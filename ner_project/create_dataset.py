import random
import os
from pathlib import Path
from collections import defaultdict

random.seed(42)

# -------------------------
# 1. DEVASA VARLIK LİSTELERİ
# -------------------------
persons = [
    "Recep Tayyip Erdoğan", "Özgür Özel", "Ekrem İmamoğlu", "Mansur Yavaş", "Ali Koç",
    "Elon Musk", "Selçuk Bayraktar", "Meral Akşener", "Devlet Bahçeli", "Hakan Fidan",
    "Mehmet Şimşek", "Ali Yerlikaya", "Fatih Terim", "Arda Güler", "Hakan Çalhanoğlu",
    "Aziz Sancar", "Orhan Pamuk", "Sezen Aksu", "Tarkan", "Haluk Bilginer",
    "Bill Gates", "Vladimir Putin", "Joe Biden", "Emmanuel Macron", "Zelenskiy",
    "Fatma Şahin", "Tunç Soyer", "Yılmaz Büyükerşen", "Murat Kurum", "Süleyman Soylu",
    "Kemal Kılıçdaroğlu", "Ümit Özdağ", "Ahmet Davutoğlu", "Ali Babacan",
    "İlker Başbuğ", "Hulusi Akar", "Yaşar Güler", "Canan Karatay", "Celal Şengör",
    "İlber Ortaylı", "Acun Ilıcalı", "Müge Anlı", "Cem Yılmaz", "Şener Şen"
]

locations = [
    "Ankara", "İstanbul", "İzmir", "Bursa", "Antalya", "Diyarbakır", "Trabzon", "Konya",
    "Adana", "Gaziantep", "Eskişehir", "Samsun", "Muğla", "Aydın", "Denizli",
    "Washington", "Londra", "Paris", "Berlin", "Moskova", "Pekin", "Tokyo", "Bakü",
    "Brüksel", "Roma", "Madrid", "Atina", "Kahire", "Kandil", "Erbil", "Gazze",
    "Kızılay", "Kadıköy", "Alsancak", "Çankaya", "Beşiktaş", "Üsküdar", "Esenboğa",
    "Atatürk Havalimanı", "Sabiha Gökçen", "Boğaziçi Köprüsü", "Taksim Meydanı",
    "Ege Bölgesi", "Marmara", "Güneydoğu Anadolu", "Karadeniz", "Akdeniz"
]

orgs = [
    "NATO", "Avrupa Birliği", "Birleşmiş Milletler", "Merkez Bankası", "TCMB", "AFAD",
    "Türk Hava Yolları", "THY", "ASELSAN", "TUSAŞ", "HAVELSAN", "ROKETSAN", "Baykar",
    "Google", "Microsoft", "Apple", "Amazon", "Tesla", "SpaceX", "Twitter",
    "Fenerbahçe", "Galatasaray", "Beşiktaş", "Trabzonspor", "TFF", "UNESCO",
    "WHO", "IMF", "Dünya Bankası", "TÜBİTAK", "TRT", "Anadolu Ajansı", "AA",
    "Kızılay", "Yeşilay", "Koç Holding", "Sabancı Holding", "Eczacıbaşı", "Ziraat Bankası",
    "Halkbank", "Vakıfbank", "Trendyol", "Getir", "Hepsiburada", "Yemeksepeti"
]

dates = [
    "12 Mayıs 2024", "29 Ekim", "15 Temmuz", "30 Ağustos", "2025 yılı", "önümüzdeki hafta",
    "geçen ay", "dün", "yarın", "bugün", "Pazartesi günü", "Salı akşamı", "Hafta sonu",
    "Aralık ayında", "2023'ün ilk çeyreğinde", "Kurban Bayramı'nda", "Ramazan'da",
    "saat 10:00'da", "gece yarısı", "yüzyılın sonunda", "Cumhuriyetin 100. yılında"
]

money = [
    "100 milyon TL", "50 milyar dolar", "10 bin Euro", "5 bin lira", "2 milyon sterlin",
    "milyarlarca lira", "trilyon dolarlık", "sembolik bir ücret", "rekor ceza",
    "150 TL", "0.50 kuruş", "yüklü miktarda para", "asgari ücret"
]

percentages = [
    "%20", "%50", "%0.5", "yüzde 10", "yüzde yüze yakın", "oranında", "%1.5 artış",
    "yüzde 3'lük dilim", "yüzde seksen", "tamamı", "yarı yarıya"
]

# -------------------------
# 2. TÜRKÇE SES VE EK KURALLARI (GELİŞMİŞ)
# -------------------------
back_vowels = "aıou"
front_vowels = "eiöü"
hard_consonants = "fstkçşhp"

def last_vowel(word):
    for ch in reversed(word.lower()):
        if ch in back_vowels + front_vowels: return ch
    return "a"

def is_hard_consonant(word):
    return word[-1].lower() in hard_consonants

def inflect(word, case):
    lv = last_vowel(word)
    if case == "locative": # -da/de
        d = "t" if is_hard_consonant(word) else "d"
        v = "a" if lv in back_vowels else "e"
        return f"{word}'{d}{v}"
    elif case == "ablative": # -dan/den
        d = "t" if is_hard_consonant(word) else "d"
        v = "a" if lv in back_vowels else "e"
        return f"{word}'{d}{v}n"
    elif case == "dative": # -a/e
        v = "a" if lv in back_vowels else "e"
        if word[-1].lower() in back_vowels + front_vowels: return f"{word}'y{v}"
        return f"{word}'{v}"
    elif case == "genitive": # -ın/in/un/ün
        if lv in "aı": v = "ı"
        elif lv in "ei": v = "i"
        elif lv in "ou": v = "u"
        else: v = "ü"
        if word[-1].lower() in back_vowels + front_vowels: return f"{word}'n{v}n"
        return f"{word}'{v}n"
    elif case == "accusative": # -ı/i/u/ü
        if lv in "aı": v = "ı"
        elif lv in "ei": v = "i"
        elif lv in "ou": v = "u"
        else: v = "ü"
        if word[-1].lower() in back_vowels + front_vowels: return f"{word}'y{v}"
        return f"{word}'{v}"
    return word

# -------------------------
# 3. 100+ ŞABLON SİSTEMİ
# -------------------------
def generate_complex_sentence():
    p = random.choice(persons)
    loc = random.choice(locations)
    org = random.choice(orgs)
    d = random.choice(dates)
    m = random.choice(money)
    pct = random.choice(percentages)

    templates = [
        # Siyaset & Diplomasi
        (f"{p}, {inflect(loc, 'locative')} yaptığı açıklamada {org} ile ilişkilerin {d} güçleneceğini belirtti.", [(p, "PER"), (loc, "LOC"), (org, "ORG"), (d, "DATE")]),
        (f"{org} heyeti, {inflect(p, 'accusative')} ziyaret etmek üzere {inflect(loc, 'dative')} doğru yola çıktı.", [(org, "ORG"), (p, "PER"), (loc, "LOC")]),
        (f"{d} tarihinde {inflect(loc, 'ablative')} düzenlenen zirveye {p} damga vurdu.", [(d, "DATE"), (loc, "LOC"), (p, "PER")]),
        (f"{org}'in {d} yayınladığı rapor, {inflect(loc, 'genitive')} güvenlik durumunu ele alıyor.", [(org, "ORG"), (d, "DATE"), (loc, "LOC")]),
        
        # Ekonomi
        (f"{org}, bu yıl {inflect(loc, 'locative')} {m} değerinde yeni bir yatırım yapacak.", [(org, "ORG"), (loc, "LOC"), (m, "MONEY")]),
        (f"Enflasyon rakamları {d} açıklandı; {pct} seviyesinde bir artış gözlendi.", [(d, "DATE"), (pct, "PERCENT")]),
        (f"{p}, {org} CEO'su olarak {m} maaşla göreve başladı.", [(p, "PER"), (org, "ORG"), (m, "MONEY")]),
        (f"{inflect(loc, 'locative')} döviz kurları {d} %5 değer kaybetti.", [(loc, "LOC"), (d, "DATE"), ("%5", "PERCENT")]),
        
        # Teknoloji & Sağlık
        (f"{org} mühendisleri {inflect(loc, 'locative')} {pct} daha hızlı bir çip geliştirdi.", [(org, "ORG"), (loc, "LOC"), (pct, "PERCENT")]),
        (f"{p}, {d} tarihinde {inflect(org, 'genitive')} yeni kanser aşısını duyurdu.", [(p, "PER"), (d, "DATE"), (org, "ORG")]),
        (f"Yapay zeka alanında {inflect(loc, 'locative')} yapılan çalışmalar {m} hibe desteği aldı.", [(loc, "LOC"), (m, "MONEY")]),
        
        # Spor
        (f"{org}, {p} ile {d} için {m} karşılığında anlaştı.", [(org, "ORG"), (p, "PER"), (d, "DATE"), (m, "MONEY")]),
        (f"{inflect(loc, 'locative')} oynanan dev derbide {org} galip geldi.", [(loc, "LOC"), (org, "ORG")]),
        (f"{p}, sakatlığı nedeniyle {d} sahalardan uzak kalacak.", [(p, "PER"), (d, "DATE")]),
        
        # Karmaşık / Uzun Yapılar
        (f"Bugün {inflect(loc, 'locative')} basın toplantısı düzenleyen {p}, {org} tarafından kendisine yöneltilen suçlamaları reddederek {d} tarihini işaret etti.", [(loc, "LOC"), (p, "PER"), (org, "ORG"), (d, "DATE")]),
        (f"Dünya genelinde {pct} pazar payına sahip olan {org}, {inflect(loc, 'genitive')} ardından {d} yeni şubesini açıyor.", [(pct, "PERCENT"), (org, "ORG"), (loc, "LOC"), (d, "DATE")]),
    ]

    # Daha fazla şablon eklemek için burada döngüler veya varyasyonlar kullanılabilir
    # (Burada temsili olarak 16 temel yapı var, ama listelerle binlerce varyasyon oluşur)
    
    sentence, entities = random.choice(templates)
    return sentence, entities

def generate_negative_sentence():
    neutrals = [
        "Hava bugün oldukça güneşli ve sıcak geçecek.",
        "Yeni çıkan yemek tarifleri kitabını mutlaka okumalısınız.",
        "Spor yapmanın düzenli uykuya büyük katkısı olduğu biliniyor.",
        "Teknoloji dünyasındaki gelişmeler bazen takibi zorlaştırıyor.",
        "Bahçedeki çiçekler baharın gelmesiyle birlikte açmaya başladı.",
        "Yarın akşam için güzel bir film planı yaptık.",
        "Kitap okumak, kelime dağarcığını geliştiren en önemli alışkanlıktır.",
        "Okulların tatil olmasına daha çok vakit var.",
        "Sağlıklı beslenmek için sebze tüketimine dikkat edilmeli.",
        "Müzik dinlemek stresi azaltan doğal bir yöntemdir.",
        "Deniz kenarında yürüyüş yapmak insana huzur veriyor.",
        "Akşam trafiği her zamanki gibi yoğundu.",
        "Yemekten sonra çay içmek bir Türk geleneğidir.",
        "Gökyüzünde yıldızlar bu gece çok parlak görünüyor.",
        "Yorgun bir günün ardından dinlenmek gibisi yok."
    ]
    return random.choice(neutrals), []

# -------------------------
# 4. TOKENIZE VE ETİKETLEME
# -------------------------
def tokenize(text):
    for ch in [".", ",", "!", "?", ":", ";", "(", ")", "\"", "'"]:
        text = text.replace(ch, f" {ch} ")
    return text.split()

def label_tokens(tokens, entities):
    tags = ["O"] * len(tokens)
    for entity, etype in entities:
        # Entity içinde çekim eki varsa (Ankara'da gibi) temizleyip bulalım
        clean_entity = entity.split("'")[0] if "'" in entity else entity
        etoks = tokenize(entity)
        
        for i in range(len(tokens) - len(etoks) + 1):
            match = True
            for j in range(len(etoks)):
                if tokens[i+j] != etoks[j]:
                    match = False
                    break
            if match:
                tags[i] = f"B-{etype}"
                for j in range(1, len(etoks)):
                    tags[i+j] = f"I-{etype}"
    return list(zip(tokens, tags))

# -------------------------
# 5. DATASET ÜRETİMİ (LEAKAGE ÖNLEME)
# -------------------------
def generate_dataset(n, neg_ratio=0.2):
    data = []
    seen = set()
    while len(data) < n:
        if random.random() < neg_ratio:
            sent, ents = generate_negative_sentence()
        else:
            sent, ents = generate_complex_sentence()
            
        if sent not in seen:
            # Data Augmentation: %15 ihtimalle küçük harf / gürültü
            orig_sent = sent
            if random.random() < 0.15:
                sent = sent.lower()
                ents = [(e.lower(), t) for e, t in ents]
                
            tokens = tokenize(sent)
            labeled = label_tokens(tokens, ents)
            data.append(labeled)
            seen.add(orig_sent)
    return data

def save_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for sent in data:
            for word, tag in sent:
                f.write(f"{word} {tag}\n")
            f.write("\n")

def main():
    out = Path("data/generated")
    out.mkdir(parents=True, exist_ok=True)

    print("Gerçekçi veri seti üretiliyor...")
    # Toplam 20 bin cümle üretelim (Daha fazla çeşitlilik için)
    full_data = generate_dataset(20000)
    
    # Karıştır ve böl
    random.shuffle(full_data)
    
    # Train / Dev / Test Split
    n = len(full_data)
    train_data = full_data[:int(n*0.8)]
    dev_data = full_data[int(n*0.8):int(n*0.9)]
    test_data = full_data[int(n*0.9):]

    save_file(out / "train.txt", train_data)
    save_file(out / "dev.txt", dev_data)
    save_file(out / "test.txt", test_data)

    print(f"Başarıyla üretildi: {len(train_data)} Train, {len(dev_data)} Dev, {len(test_data)} Test.")

if __name__ == "__main__":
    main()