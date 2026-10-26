import streamlit as st
import os
import pandas as pd
import html

# ==========================================
# 1. SAYFA YAPILANDIRMASI
# ==========================================
st.set_page_config(
    page_title="Türkçe Haber NER | Production-Ready",
    page_icon="🇹🇷",
    layout="wide"
)

# ==========================================
# 2. CSS - MODERN & RESPONSIVE (Kullanıcı Talebi)
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

body { font-family: 'Outfit', sans-serif; }

.main { background-color: #f8fafc; }

/* Header Panel */
.header-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    padding: 2.5rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
}

/* Analiz Sonuç Kutusu - KESİN KURAL: Metni bozmaz */
.analysis-result {
    background: white;
    padding: 2rem;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    line-height: 2;
    color: #334155;
    font-size: 1.1rem;
    word-wrap: break-word;
    overflow-wrap: break-word;
    white-space: pre-wrap;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
}

/* Entity Tag Tasarımı */
.entity {
    display: inline-block;
    padding: 2px 8px;
    margin: 2px 4px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.9em;
    border: 1px solid rgba(0,0,0,0.05);
    transition: transform 0.2s;
}
.entity:hover { transform: scale(1.05); }

/* Renk Paleti */
.per { background-color: #dbeafe; color: #1e40af; } /* KİŞİ */
.org { background-color: #fef3c7; color: #92400e; } /* KURUM */
.loc { background-color: #d1fae5; color: #065f46; } /* YER */
.date { background-color: #fee2e2; color: #991b1b; } /* TARİH */
.para { background-color: #f3e8ff; color: #6b21a8; } /* PARA */
.urun { background-color: #e0f2fe; color: #0369a1; } /* ÜRÜN */
.dur { background-color: #f1f5f9; color: #475569; }  /* SÜRE */

.type-label {
    font-size: 0.7em;
    text-transform: uppercase;
    margin-left: 5px;
    opacity: 0.7;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. MODEL YÜKLEME
# ==========================================
from src.inference.pipeline import AdvancedNERPipeline

@st.cache_resource
def load_model():
    model_path = "model"
    if not os.path.exists(model_path):
        model_path = "results/checkpoint-6250"
    return AdvancedNERPipeline(model_path=model_path) if os.path.exists(model_path) else None

nlp = load_model()

# ==========================================
# 4. YARDIMCI FONKSİYONLAR
# ==========================================
def escape_html(text):
    """XSS Koruması (Kullanıcı Talebi)"""
    return html.escape(text)

def get_entity_class(label):
    mapping = {
        "PER": "per", "ORG": "org", "LOC": "loc", 
        "DATE": "date", "MONEY": "para", "URUN": "urun", 
        "DURATION": "dur", "PERCENT": "para"
    }
    return mapping.get(label, "dur")

def get_type_name(label):
    mapping = {
        "PER": "KİŞİ", "ORG": "KURUM", "LOC": "YER", 
        "DATE": "TARİH", "MONEY": "PARA", "URUN": "ÜRÜN", 
        "DURATION": "SÜRE", "PERCENT": "YÜZDE"
    }
    return mapping.get(label, label)

# ==========================================
# 5. UI - HEADER
# ==========================================
st.markdown("""
<div class="header-container">
    <h1 style='margin:0;'>🚀 Profesyonel Türkçe Haber NER</h1>
    <p style='opacity:0.8; margin-top:5px;'>Hızlı, Güvenli ve Kesin Varlık Tanıma Sistemi</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 6. SESSION STATE YÖNETİMİ
# ==========================================
if "analyzed_text" not in st.session_state:
    st.session_state.analyzed_text = ""
if "ner_results" not in st.session_state:
    st.session_state.ner_results = []
if "show_results" not in st.session_state:
    st.session_state.show_results = False

# ==========================================
# 7. SIFIRLA VE ANALİZ CALLBACK'LERİ
# ==========================================
def reset_analysis():
    """Sıfırla Butonu (Kullanıcı Talebi - Tam Temizlik)"""
    st.session_state.news_input = ""
    st.session_state.ner_results = []
    st.session_state.analyzed_text = ""
    st.session_state.show_results = False

def run_analysis():
    """Yeni analizde eski sonuçları temizle ve çalıştır"""
    text = st.session_state.news_input.strip()
    if not text:
        st.warning("⚠️ Lütfen analiz için metin girin.")
        return
    
    if nlp:
        with st.spinner("🤖 Analiz ediliyor..."):
            # 1. Eski verileri temizle
            st.session_state.ner_results = []
            
            # 2. Tahmin al (Model metni asla değiştirmez)
            results = nlp.predict(text)
            
            # 3. Sonuçları kaydet
            st.session_state.ner_results = results
            st.session_state.analyzed_text = text
            st.session_state.show_results = True
    else:
        st.error("❌ Model yüklenemedi!")

# ==========================================
# 8. ANA ARAYÜZ
# ==========================================
col_in, col_out = st.columns([1.2, 1], gap="large")

with col_in:
    st.markdown("### 📝 Haber Metni")
    input_text = st.text_area(
        "Giriş",
        height=250,
        placeholder="Haber metnini buraya yapıştırın...",
        key="news_input",
        label_visibility="collapsed"
    )
    
    c1, c2 = st.columns(2)
    with c1:
        st.button("⚡ Analiz Et", type="primary", use_container_width=True, on_click=run_analysis)
    with c2:
        st.button("🗑️ Sıfırla", use_container_width=True, on_click=reset_analysis)

# ==========================================
# 9. RENDERING (Kullanıcı Talebi: Sondan Başa Slice)
# ==========================================
if st.session_state.show_results:
    with col_out:
        st.markdown("### 🎯 Analiz Sonucu")
        
        original_text = st.session_state.analyzed_text
        results = st.session_state.ner_results
        
        # 1. XSS Koruması: Metni escape et
        escaped_text = escape_html(original_text)
        
        # 2. Rendering: Sondan başa doğru (Index kaymasını önler)
        # ÖNEMLİ: entities.sort((a, b) => b.start - a.start)
        sorted_ents = sorted(results, key=lambda x: x["start"], reverse=True)
        
        rendered_html = escaped_text
        for ent in sorted_ents:
            s, e = ent["start"], ent["end"]
            label = ent["entity_group"]
            css_class = get_entity_class(label)
            type_name = get_type_name(label)
            score_percent = ent["score"] * 100
            
            # Entity text'i escape et (içerik bozulmaması için)
            ent_word = escape_html(original_text[s:e])
            
            # Highlight Span Yapısı (Skor eklendi)
            span = (
                f'<span class="entity {css_class}">'
                f'{ent_word}'
                f'<span class="type-label">{type_name} %{score_percent:.1f}</span>'
                f'</span>'
            )

            
            # Slice ile yerleştirme (Kullanıcı talebi: KESİN YÖNTEM)
            rendered_html = rendered_html[:s] + span + rendered_html[e:]

        st.markdown(f'<div class="analysis-result">{rendered_html}</div>', unsafe_allow_html=True)
        
        # 📊 Grafikler ve İstatistikler
        if results:
            st.divider()
            df = pd.DataFrame(results)
            counts = df["entity_group"].value_counts().reset_index()
            counts.columns = ["Tür", "Adet"]
            counts["Tür"] = counts["Tür"].map(get_type_name)
            
            # Grafik Bug Fix: Streamlit st.bar_chart her render'da yenilenir
            st.markdown("#### 📊 Varlık Dağılımı")
            st.bar_chart(counts.set_index("Tür"), color="#6366f1")
            
            # Varlık Listesi
            st.markdown("#### 📋 Tespit Edilenler")
            for r in results:
                st.write(f"**{get_type_name(r['entity_group'])}:** {r['word']}")
        else:
            st.info("Bu metinde herhangi bir varlık tespit edilemedi.")

else:
    with col_out:
        st.info("🔎 Analiz sonuçları burada görünecek.")
