# ============================================================
#  SMART CAMPUS FEEDBACK SYSTEM — LIGHTWEIGHT VERSION
#  NLP & Sentiment Analysis via HuggingFace Inference API
#  EAS Kecerdasan Buatan 2025/2026 | Kelas 6A STI
#  Universitas Muhammadiyah Prof. Dr. Hamka
# ============================================================
#  Model      : IndoBERT via HF Inference API (cloud)
#  Deploy     : Hugging Face Spaces (gratis)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re, time, requests
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Campus Feedback | AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{ font-family:'Syne',sans-serif !important; }
.stApp{ background:#050810; }
[data-testid="stSidebar"]{ background:#080d1a !important; border-right:1px solid rgba(255,255,255,.06); }
.block-container{ padding:1.5rem 2rem; }
h1,h2,h3{ font-family:'Syne',sans-serif !important; font-weight:800 !important; }
.stTabs [data-baseweb="tab-list"]{ background:#0d1120; border-radius:12px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"]{ background:transparent; color:rgba(255,255,255,.4); border-radius:8px; font-weight:700; font-size:13px; }
.stTabs [aria-selected="true"]{ background:linear-gradient(135deg,#00f5ff22,#39ff8222) !important; color:white !important; }
.stButton>button{ background:linear-gradient(135deg,#00f5ff,#39ff82); color:#050810 !important; font-weight:800; border:none; border-radius:10px; }
.stDownloadButton button{ background:linear-gradient(135deg,#00f5ff,#39ff82) !important; color:#050810 !important; font-weight:800 !important; border:none !important; border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────
API_URL   = "https://api-inference.huggingface.co/models/mdhugol/indonesia-bert-sentiment-classifier"
LABEL_MAP = {'LABEL_0':'Positif','LABEL_1':'Netral','LABEL_2':'Negatif'}
COLOR_MAP = {'Positif':'#39ff82','Netral':'#ffe600','Negatif':'#ff2d78'}
EMOJI_MAP = {'Positif':'😊','Netral':'😐','Negatif':'😞'}
STOPWORDS_ID = {
    'yang','dan','di','ke','dari','untuk','dengan','ini','itu','ada','pada',
    'adalah','dalam','tidak','juga','lebih','sudah','saya','kami','kita',
    'mereka','dia','kamu','anda','atau','tapi','namun','karena','oleh','akan',
    'bisa','dapat','harus','jika','maka','seperti','sangat','sekali','pun',
    'pula','nya','lah','kah','sih','aja','yg','dgn','telah','agar','setelah',
    'sebelum','bagi','hal','satu','dua','tiga','empat','lima','enam','tujuh',
    'delapan','sembilan','setiap','sama','sering','baru','saja','masih','belum',
    'jadi','lagi','kalau','mau','perlu','punya','saat','selalu','semua'
}

# ── HF Token ─────────────────────────────────────────────────
def get_token():
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return None

# ── HF Inference API ─────────────────────────────────────────
def call_api(text, token, retries=3):
    """Panggil HF Inference API dengan retry untuk cold start model."""
    headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(retries):
        try:
            res = requests.post(
                API_URL,
                headers=headers,
                json={"inputs": text[:512]},
                timeout=30
            )
            data = res.json()
            # Model masih loading (cold start) → tunggu
            if isinstance(data, dict) and "error" in data:
                if "loading" in data.get("error","").lower():
                    wait = data.get("estimated_time", 20)
                    time.sleep(min(wait, 25))
                    continue
                return None, 0.0
            # Ambil prediksi terbaik
            if isinstance(data, list) and len(data) > 0:
                preds = data[0] if isinstance(data[0], list) else data
                best  = max(preds, key=lambda x: x['score'])
                label = LABEL_MAP.get(best['label'], 'Netral')
                score = round(best['score'] * 100, 1)
                return label, score
        except Exception:
            time.sleep(2)
    return 'Netral', 0.0

# ── Predict Batch ─────────────────────────────────────────────
def predict_batch(texts, token):
    results = []
    prog = st.progress(0, text="Menganalisis sentimen...")
    for i, text in enumerate(texts):
        label, score = call_api(preprocess(text), token)
        results.append({'sentimen': label or 'Netral', 'confidence': score})
        prog.progress((i+1)/len(texts),
                      text=f"Menganalisis... {i+1}/{len(texts)}")
    prog.empty()
    return pd.DataFrame(results)

# ── Utils ─────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def clean_words(texts):
    out = []
    for t in texts:
        for w in preprocess(t).split():
            if w not in STOPWORDS_ID and len(w) > 3:
                out.append(w)
    return out

def make_wc(words, cmap):
    if not words: return None
    return WordCloud(width=520, height=270, background_color='#0d1120',
                     colormap=cmap, min_font_size=10, max_font_size=74,
                     collocations=False).generate(' '.join(words))

def metric_card(label, value, color):
    st.markdown(f"""
    <div style='background
