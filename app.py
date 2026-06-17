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
def call_api(text, token, retries=5):
    """Panggil HF Inference API dengan optimasi handling untuk cold start."""
    if not token:
        return 'Netral', 0.0
        
    headers = {"Authorization": f"Bearer {token}"}
    
    for attempt in range(retries):
        try:
            res = requests.post(
                API_URL,
                headers=headers,
                json={"inputs": text[:512]},
                timeout=30
            )
            
            # Jika terkena Rate Limit, tunggu sebentar lalu coba lagi
            if res.status_code == 429:
                time.sleep(3)
                continue
                
            data = res.json()
            
            # Model masih loading (cold start) → tunggu sesuai estimasi server
            if isinstance(data, dict) and "error" in data:
                error_msg = data.get("error", "").lower()
                if "loading" in error_msg:
                    wait_time = data.get("estimated_time", 20)
                    time.sleep(min(wait_time, 20))
                    continue
                return 'Netral', 0.0
                
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
        results.append({'sentimen': label, 'confidence': score})
        # Beri jeda kecil antar baris agar tidak memicu deteksi spam/rate-limit di server HF
        time.sleep(0.1)
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
    <div style='background:#0d1120;border:1px solid rgba(255,255,255,0.07);
         border-radius:16px;padding:20px;text-align:center'>
      <div style='font-size:10px;text-transform:uppercase;letter-spacing:0.1em;
           color:rgba(255,255,255,0.35);margin-bottom:8px;font-family:DM Mono,monospace'>{label}</div>
      <div style='font-size:38px;font-weight:800;line-height:1;color:{color}'>{value}</div>
    </div>""", unsafe_allow_html=True)

# ── Sample Dataset ────────────────────────────────────────────
@st.cache_data
def get_dataset():
    komentar = [
        "Dosen mata kuliah ini sangat membantu dan menjelaskan materi dengan sangat jelas",
        "Fasilitas laboratorium komputer sangat baik, lengkap, dan selalu terawat",
        "Perpustakaan kampus sangat nyaman dan koleksi bukunya sangat lengkap",
        "Sistem informasi akademik sangat mudah digunakan dan responsif sekali",
        "Kantin kampus menyediakan makanan yang bervariasi dengan harga terjangkau",
        "Kegiatan seminar kampus sangat bermanfaat dan informatif bagi mahasiswa",
        "Dosen pembimbing sangat kooperatif dan cepat memberikan feedback",
        "WiFi kampus sudah jauh lebih cepat dan stabil dari semester sebelumnya",
        "Ruang kuliah sangat nyaman dengan AC yang berfungsi sangat baik",
        "Pelayanan administrasi kini jauh lebih cepat dan efisien",
        "Kegiatan UKM sangat beragam dan mendukung pengembangan diri mahasiswa",
        "Dosen pengampu sangat sabar menjelaskan materi yang sulit dipahami",
        "Lingkungan kampus bersih dan terawat oleh petugas kebersihan",
        "Perpustakaan digital sangat membantu mengakses jurnal ilmiah internasional",
        "Proses pendaftaran KRS sangat mudah dan sistem tidak pernah error",
        "Beasiswa yang disediakan sangat membantu mahasiswa kurang mampu",
        "Laboratorium bahasa sangat modern dan mendukung pembelajaran efektif",
        "Dosen selalu hadir tepat waktu dan materi disampaikan sangat relevan",
        "Fasilitas olahraga sangat lengkap dan selalu terawat dengan baik",
        "Sistem absensi digital memudahkan monitoring kehadiran kuliah",
        "Koneksi WiFi kampus sangat lambat dan sering mengalami gangguan",
        "Proses administrasi sangat berbelit-belit dan memakan waktu lama",
        "Parkiran kampus sangat sempit dan tidak mencukupi untuk mahasiswa",
        "Kantin sering kehabisan makanan padahal jam makan belum selesai",
        "Toilet kampus kurang bersih dan sering tidak tersedia air bersih",
        "Dosen sering tidak hadir tanpa memberikan pemberitahuan apapun",
        "AC di ruang kuliah sering rusak dan membuat belajar tidak nyaman",
        "Buku di perpustakaan sudah sangat usang dan tidak pernah diperbarui",
        "Birokrasi mengurus surat keterangan sangat panjang dan melelahkan",
        "Jadwal ujian sering berubah mendadak tanpa pemberitahuan cukup",
        "Kursi di ruang kuliah banyak yang rusak dan tidak segera diperbaiki",
        "Sistem informasi akademik sering error dan tidak bisa diakses",
        "Dosen memberikan tugas terlalu banyak tanpa mempertimbangkan waktu",
        "Harga kantin terlalu mahal dibandingkan kualitas makanan disajikan",
        "Sidang skripsi sangat sulit dijadwalkan karena dosen susah ditemui",
        "Laboratorium komputer sering penuh dan mahasiswa harus antri lama",
        "Kebisingan di sekitar kampus mengganggu proses belajar mengajar",
        "Lift kampus sering tidak berfungsi dan harus naik tangga terus",
        "Informasi beasiswa sangat kurang dipublikasikan secara luas",
        "Jadwal kuliah terlalu padat membuat mahasiswa kelelahan dan stres",
        "Kampus memiliki tiga gedung utama untuk kegiatan perkuliahan",
        "Perpustakaan buka setiap hari mulai pukul delapan hingga delapan malam",
        "Terdapat dua kantin di area kampus dengan berbagai pilihan makanan",
        "Jadwal kuliah semester ini dimulai Februari dan berakhir Juni",
        "Kampus menyediakan layanan shuttle bus dari stasiun terdekat",
        "Ada empat program studi di Fakultas Teknologi Industri dan Informatika",
        "Gedung perpustakaan selesai direnovasi pada awal tahun ini",
        "Semester ini terdapat empat belas mata kuliah yang ditawarkan",
        "Kampus memiliki kebijakan baru terkait penggunaan laptop di kelas",
        "Pendaftaran wisuda dilakukan secara online melalui portal akademik",
        "Kampus menyediakan ruang diskusi yang dapat dipinjam mahasiswa",
        "Mata kuliah Kecerdasan Buatan merupakan mata kuliah wajib semester ini",
        "Jam operasional kantor administrasi pukul delapan sampai empat",
        "Terdapat area parkir khusus mahasiswa di sisi timur gedung kampus",
        "Kampus mengadakan wisuda dua kali dalam satu tahun akademik",
        "Laboratorium komputer berkapasitas empat puluh mahasiswa per sesi",
        "Program beasiswa KIP-K tersedia untuk mahasiswa yang memenuhi kriteria",
        "Mata kuliah pilihan dapat dipilih mahasiswa mulai semester lima",
        "Kampus memiliki dua puluh unit kegiatan mahasiswa aktif",
        "Proses registrasi ulang semester dilakukan setiap awal periode baru",
    ]
    kat   = ['Fasilitas','Akademik','Dosen','Administrasi','Kantin','Perpustakaan','WiFi','Kegiatan']
    dates = pd.date_range('2025-01-05', periods=60, freq='5D')
    return pd.DataFrame({
        'id'      : range(1, 61),
        'tanggal' : dates.strftime('%Y-%m-%d'),
        'bulan'   : dates.strftime('%b %Y'),
        'kategori': [kat[i % 8] for i in range(60)],
        'komentar': komentar
    })

# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:24px 0 12px'>
      <div style='font-size:46px'>🎓</div>
      <div style='font-size:16px;font-weight:800;
           background:linear-gradient(135deg,#00f5ff,#39ff82);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-top:8px'>
        Smart Campus</div>
      <div style='font-size:11px;color:rgba(255,255,255,0.3)'>Feedback System · IndoBERT AI</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    page = st.radio("📍 Navigasi", [
        "🏠  Dashboard",
        "📈  Tren & Kategori",
        "☁️   Word Cloud",
        "💬  Detail Komentar",
        "🔍  Uji Teks Baru",
        "📋  Tabel Data",
    ])
    st.divider()
    st.markdown("""
    <div style='font-size:10px;color:rgba(255,255,255,0.22);font-family:DM Mono,monospace;line-height:2.4'>
    🤖 Model · IndoBERT (API)<br>
    ⚡ RAM  · ~150MB (ringan!)<br>
    📊 Data · 60 komentar<br>
    🏫 Kelas · 6A STI UHAMKA<br>
    📅 EAS · 2025/2026
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#0d1120,#111827);
     border:1px solid rgba(255,255,255,0.08);border-radius:22px;
     padding:30px 36px;margin-bottom:24px'>
  <div style='font-size:11px;letter-spacing:0.14em;color:#00f5ff;
       text-transform:uppercase;margin-bottom:10px;font-family:DM Mono,monospace'>
    ✦ NLP &amp; Sentiment Analysis · EAS Kecerdasan Buatan 2025/2026
  </div>
  <h1 style='font-size:2rem;font-weight:800;margin:0 0 8px;
       background:linear-gradient(135deg,#00f5ff,#39ff82);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
    Smart Campus Feedback System
  </h1>
  <p style='color:rgba(255,255,255,0.4);font-size:13px;margin:0;line-height:1.6'>
    Analisis sentimen komentar mahasiswa · <b style='color:rgba(255,255,255,0.7)'>IndoBERT via HF API</b> ·
    Universitas Muhammadiyah Prof. Dr. Hamka · Kelas 6A STI
  </p>
</div>
""", unsafe_allow_html=True)

# ── Token Check ───────────────────────────────────────────────
token = get_token()
if not token:
    st.error("""
    ⚠️ **HuggingFace Token belum dikonfigurasi!**

    **Cara setup di HF Spaces:**
    1. Buka Settings → Secrets
    2. Tambahkan secret: `HF_TOKEN` = token kamu
    3. Dapatkan token gratis di: https://huggingface.co/settings/tokens
    """)
    st.stop()

# ── Load & Predict ─────────────────────────────────────────────
df_raw = get_dataset()
uploaded = st.file_uploader("📁 Upload CSV (opsional) — atau pakai 60 data sampel:", type=['csv'])
if uploaded:
    df_raw = pd.read_csv(uploaded)
    st.success(f"✅ Dataset dimuat: {len(df_raw)} baris")

# Cache hasil prediksi agar tidak call API berulang
cache_key = "pred_cache"
if cache_key not in st.session_state or len(st.session_state[cache_key]) != len(df_raw):
    with st.spinner("🤖 Menganalisis sentimen dengan IndoBERT... (30–60 detik pertama kali)"):
        pred = predict_batch(df_raw['komentar'].tolist(), token)
    st.session_state[cache_key] = pred
    st.success("✅ Analisis selesai!")

pred = st.session_state[cache_key]
df   = pd.concat([df_raw.reset_index(drop=True), pred], axis=1)

counts  = df['sentimen'].value_counts()
n_pos   = int(counts.get('Positif', 0))
n_neg   = int(counts.get('Negatif', 0))
n_neu   = int(counts.get('Netral',  0))
n_total = len(df)
score   = round((n_pos - n_neg) / n_total * 100)

# ════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,lbl,val,color in [
        (c1,"Total Komentar",n_total,"#00f5ff"),
        (c2,"😊 Positif",n_pos,"#39ff82"),
        (c3,"😐 Netral",n_neu,"#ffe600"),
        (c4,"😞 Negatif",n_neg,"#ff2d78"),
        (c5,"💯 Skor",f"{score:+}","#bf5aff"),
    ]:
        with col: metric_card(lbl, val, color)
    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.1, 0.9])
    with col_l:
        fig_pie = go.Figure(go.Pie(
            labels=['Positif','Netral','Negatif'],
            values=[n_pos, n_neu, n_neg], hole=0.6,
            marker_colors=['#39ff82','#ffe600','#ff2d78'], textfont_size=13))
        fig_pie.add_annotation(
            text=f"{round(n_pos/n_total*100)}%<br><span style='font-size:12px'>Positif</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=22, color='#39ff82'))
        fig_pie.update_layout(
            title=dict(text='Distribusi Sentimen', font=dict(color='#ffffff',size=14)),
            paper_bgcolor='#0d1120', plot_bgcolor='#0d1120',
            font=dict(color='rgba(255,255,255,0.7)'),
            legend=dict(orientation='h', y=-0.1),
            margin=dict(t=50,b=30,l=20,r=20), height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            gauge={'axis':{'range':[-100,100],'tickcolor':'rgba(255,255,255,0.3)'},
                   'bar':{'color':'#00f5ff'},'bgcolor':'#0d1120',
                   'steps':[
                       {'range':[-100,-33],'color':'rgba(255,45,120,0.2)'},
                       {'range':[-33,33],'color':'rgba(255,230,0,0.15)'},
                       {'range':[33,100],'color':'rgba(57,255,130,0.2)'}]},
            title={'text':"Skor Sentimen",'font':{'color':'#ffffff','size':13}},
            number={'font':{'color':'#00f5ff','size':40}}))
        fig_gauge.update_layout(
            paper_bgcolor='#0d1120', font=dict(color='rgba(255,255,255,0.7)'),
            margin=dict(t=60,b=20,l=30,r=30), height=320)
        st.plotly_chart(fig_gauge, use_container_width=True)

    cat_df = df.groupby(['kategori','sentimen']).size().reset_index(name='count')
    fig_bar = px.bar(cat_df, x='kategori', y='count', color='sentimen',
                     color_discrete_map=COLOR_MAP, barmode='group',
                     title='Sentimen per Kategori')
    fig_bar.update_layout(
        paper_bgcolor='#0d1120', plot_bgcolor='#0d1120',
        font=dict(color='rgba(255,255,255,0.7)'),
        title_font=dict(color='#ffffff',size=14),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)',title=''),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)',title='Jumlah'),
        margin=dict(t=50,b=20,l=20,r=20), height=300)
    st.plotly_chart(fig_bar, use_container_width=True)

    dominant  = counts.idxmax()
    color_d   = COLOR_MAP[dominant]
    worst_kat = df[df['sentimen']=='Negatif']['kategori'].value_counts().idxmax() if n_neg > 0 else '-'
    best_kat  = df[df['sentimen']=='Positif']['kategori'].value_counts().idxmax() if n_pos > 0 else '-'
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0d1120,#111827);
         border:1px solid rgba(255,255,255,0.08);border-radius:18px;padding:24px'>
      <div style='font-size:11px;color:rgba(255,255,255,0.35);text-transform:uppercase;
           letter-spacing:0.08em;margin-bottom:12px;font-family:DM Mono,monospace'>💡 Insight Otomatis</div>
      <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px'>
        <div style='background:rgba(57,255,130,0.07);border:1px solid rgba(57,255,130,0.2);
             border-radius:12px;padding:14px'>
          <div style='font-size:10px;color:#39ff82;margin-bottom:4px;font-family:DM Mono,monospace'>TERBAIK</div>
          <div style='font-size:18px;font-weight:800;color:white'>{best_kat}</div>
        </div>
        <div style='background:rgba(255,45,120,0.07);border:1px solid rgba(255,45,120,0.2);
             border-radius:12px;padding:14px'>
          <div style='font-size:10px;color:#ff2d78;margin-bottom:4px;font-family:DM Mono,monospace'>PERLU PERBAIKAN</div>
          <div style='font-size:18px;font-weight:800;color:white'>{worst_kat}</div>
        </div>
        <div style='background:rgba(0,245,255,0.07);border:1px solid rgba(0,245,255,0.2);
             border-radius:12px;padding:14px'>
          <div style='font-size:10px;color:#00f5ff;margin-bottom:4px;font-family:DM Mono,monospace'>KEPUASAN</div>
          <div style='font-size:18px;font-weight:800;color:white'>{round(n_pos/n_total*100)}%</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  PAGE: TREN & KATEGORI
# ════════════════════════════════════════════════════════════
elif page == "📈  Tren & Kategori":
    st.markdown("### 📈 Tren Sentimen & Heatmap Kategori")
    if 'bulan' in df.columns:
        tren = df.groupby(['bulan','sentimen']).size().reset_index(name='count')
        fig_line = px.line(tren, x='bulan', y='count', color='sentimen',
                           color_discrete_map=COLOR_MAP, markers=True,
                           title='Tren Sentimen per Bulan')
        fig_line.update_traces(line=dict(width=2.5))
        fig_line.update_layout(
            paper_bgcolor='#0d1120', plot_bgcolor='#0d1120',
            font=dict(color='rgba(255,255,255,0.7)'),
            title_font=dict(color='#ffffff',size=14),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)',title=''),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)',title='Jumlah'),
            margin=dict(t=50,b=20,l=20,r=20), height=300)
        st.plotly_chart(fig_line, use_container_width=True)

    pivot = df.groupby(['kategori','sentimen']).size().unstack(fill_value=0)
    fig_heat = px.imshow(pivot, text_auto=True, aspect='auto',
                         color_continuous_scale='Viridis',
                         title='Heatmap Kategori vs Sentimen')
    fig_heat.update_layout(
        paper_bgcolor='#0d1120', plot_bgcolor='#0d1120',
        font=dict(color='rgba(255,255,255,0.7)'),
        title_font=dict(color='#ffffff',size=14),
        margin=dict(t=50,b=20,l=20,r=20), height=320)
    st.plotly_chart(fig_heat, use_container_width=True)

    fig_box = px.box(df, x='kategori', y='confidence', color='sentimen',
                     color_discrete_map=COLOR_MAP,
                     title='Distribusi Confidence per Kategori')
    fig_box.update_layout(
        paper_bgcolor='#0d1120', plot_bgcolor='#0d1120',
        font=dict(color='rgba(255,255,255,0.7)'),
        title_font=dict(color='#ffffff',size=14),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)',title=''),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)',title='Confidence (%)'),
        margin=dict(t=50,b=20,l=20,r=20), height=320)
    st.plotly_chart(fig_box, use_container_width=True)

# ════════════════════════════════════════════════════════════
#  PAGE: WORD CLOUD
# ════════════════════════════════════════════════════════════
elif page == "☁️   Word Cloud":
    st.markdown("### ☁️ Word Cloud & Kata Kunci per Sentimen")
    cmaps = {'Positif':'Greens','Netral':'cool','Negatif':'Reds'}
    cols  = st.columns(3)
    for col, sent in zip(cols, ['Positif','Netral','Negatif']):
        with col:
            color = COLOR_MAP[sent]
            words = clean_words(df[df['sentimen']==sent]['komentar'].tolist())
            st.markdown(f"""
            <div style='text-align:center;padding:10px 0 8px;font-weight:800;
                 font-size:16px;color:{color}'>{EMOJI_MAP[sent]} {sent}
              <span style='font-size:11px;color:rgba(255,255,255,0.3);font-weight:400;
                   font-family:DM Mono,monospace;margin-left:6px'>
                {len(df[df['sentimen']==sent])} komentar
              </span>
            </div>""", unsafe_allow_html=True)
            wc = make_wc(words, cmaps[sent])
            if wc:
                fig, ax = plt.subplots(figsize=(5, 2.7), facecolor='#0d1120')
                ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
                st.pyplot(fig, use_container_width=True); plt.close()
            freq = Counter(words).most_common(8)
            if freq:
                wf   = pd.DataFrame(freq, columns=['Kata','Frekuensi'])
                fig2 = px.bar(wf, x='Frekuensi', y='Kata', orientation='h',
                              color_discrete_sequence=[color])
                fig2.update_layout(
                    paper_bgcolor='#0d1120', plot_bgcolor='#0d1120',
                    font=dict(color='rgba(255,255,255,0.6)',size=11),
                    margin=dict(t=6,b=6,l=6,r=6), height=230, showlegend=False,
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
                st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════
#  PAGE: DETAIL KOMENTAR
# ════════════════════════════════════════════════════════════
elif page == "💬  Detail Komentar":
    st.markdown("### 💬 Detail Komentar Mahasiswa")
    col_s, col_k = st.columns(2)
    with col_s: sel_sent = st.selectbox("Filter Sentimen:", ['Semua','Positif','Netral','Negatif'])
    with col_k: sel_kat  = st.selectbox("Filter Kategori:", ['Semua']+sorted(df['kategori'].unique().tolist()))
    df_show = df.copy()
    if sel_sent != 'Semua': df_show = df_show[df_show['sentimen']==sel_sent]
    if sel_kat  != 'Semua': df_show = df_show[df_show['kategori']==sel_kat]
    st.caption(f"Menampilkan {len(df_show)} komentar")
    for _, row in df_show.iterrows():
        c  = COLOR_MAP[row['sentimen']]; em = EMOJI_MAP[row['sentimen']]
        conf_bar = int(row['confidence'])
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.06);
             border-left:3px solid {c};border-radius:14px;padding:16px 18px;margin-bottom:12px'>
          <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px'>
            <span style='font-size:10px;color:rgba(255,255,255,0.3);font-family:DM Mono,monospace'>
              #{row['id']} · {row['kategori']} · {row['tanggal']}
            </span>
            <span style='background:{c}22;color:{c};padding:3px 12px;
                  border-radius:20px;font-size:11px;font-weight:700'>
              {em} {row['sentimen']} · {row['confidence']}%
            </span>
          </div>
          <div style='font-size:14px;color:rgba(255,255,255,0.82);line-height:1.65;margin-bottom:10px'>
            {row['komentar']}
          </div>
          <div style='background:rgba(255,255,255,0.05);border-radius:20px;height:4px'>
            <div style='background:{c};border-radius:20px;height:4px;width:{conf_bar}%'></div>
          </div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  PAGE: UJI TEKS BARU
# ════════════════════════════════════════════════════════════
elif page == "🔍  Uji Teks Baru":
    st.markdown("### 🔍 Analisis Komentar Baru secara Real-Time")
    tab1, tab2 = st.tabs(["✏️ Satu Komentar", "📋 Banyak Komentar"])

    with tab1:
        user_text = st.text_area("Masukkan komentar:", height=130,
            placeholder="Contoh: Fasilitas kampus sangat bagus dan lengkap...")
        if st.button("🚀 Analisis", key="single"):
            if user_text.strip():
                with st.spinner("Menganalisis..."):
                    label, score_val = call_api(preprocess(user_text), token)
                    color = COLOR_MAP[label]; emoji = EMOJI_MAP[label]
                col_res, col_info = st.columns(2)
                with col_res:
                    st.markdown(f"""
                    <div style='background:#0d1120;border:2px solid {color};
                         border-radius:20px;padding:32px;text-align:center'>
                      <div style='font-size:54px;margin-bottom:14px'>{emoji}</div>
                      <div style='font-size:24px;font-weight:800;color:{color};margin-bottom:6px'>{label}</div>
                      <div style='font-size:44px;font-weight:800;color:white'>{score_val}%</div>
                      <div style='font-size:11px;color:rgba(255,255,255,0.35);font-family:DM Mono,monospace'>
                        Confidence Score
                      </div>
                    </div>""", unsafe_allow_html=True)
                with col_info:
                    st.markdown("**Preprocessing:**")
                    st.code(f"Original:\n{user_text[:150]}\n\nCleaned:\n{preprocess(user_text)[:150]}", language='text')
            else:
                st.warning("⚠️ Masukkan teks terlebih dahulu!")

    with tab2:
        bulk_text = st.text_area("Komentar (satu per baris):",
            placeholder="Dosen sangat baik\nWifi sangat lambat\nKampus memiliki dua kantin", height=180)
        if st.button("🚀 Analisis Semua", key="bulk"):
            lines = [l.strip() for l in bulk_text.strip().split('\n') if l.strip()]
            if lines:
                with st.spinner(f"Menganalisis {len(lines)} komentar..."):
                    results = predict_batch(lines, token)
                    results['komentar'] = lines
                for _, row in results.iterrows():
                    c = COLOR_MAP[row['sentimen']]; em = EMOJI_MAP[row['sentimen']]
                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.025);border-left:3px solid {c};
                         border-radius:10px;padding:12px 16px;margin-bottom:8px;
                         display:flex;justify-content:space-between;align-items:center'>
                      <span style='font-size:13px;color:rgba(255,255,255,0.8)'>{row['komentar']}</span>
                      <span style='background:{c}22;color:{c};padding:3px 12px;
                            border-radius:20px;font-size:11px;font-weight:700;white-space:nowrap;margin-left:12px'>
                        {em} {row['sentimen']} {row['confidence']}%
                      </span>
                    </div>""", unsafe_allow_html=True)
                rc = results['sentimen'].value_counts()
                st.markdown(f"""<br><div style='background:#0d1120;border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;padding:16px;display:flex;gap:20px;justify-content:center'>
                  <span style='color:#39ff82;font-weight:800'>😊 {rc.get("Positif",0)}</span>
                  <span style='color:#ffe600;font-weight:800'>😐 {rc.get("Netral",0)}</span>
                  <span style='color:#ff2d78;font-weight:800'>😞 {rc.get("Negatif",0)}</span>
                </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  PAGE: TABEL DATA
# ════════════════════════════════════════════════════════════
elif page == "📋  Tabel Data":
    st.markdown("### 📋 Tabel Lengkap Hasil Analisis")
    c1,c2,c3 = st.columns(3)
    with c1: f_sent = st.multiselect("Sentimen:",['Positif','Netral','Negatif'],default=['Positif','Netral','Negatif'])
    with c2: f_kat  = st.multiselect("Kategori:", sorted(df['kategori'].unique()), default=sorted(df['kategori'].unique()))
    with c3: min_conf = st.slider("Min. Confidence (%):", 0, 100, 0)
    df_f = df[df['sentimen'].isin(f_sent) & df['kategori'].isin(f_kat) & (df['confidence']>=min_conf)]
    st.caption(f"{len(df_f)} dari {n_total} komentar · Rata-rata confidence: {df_f['confidence'].mean():.1f}%")
    def color_sent(val): return f"color:{COLOR_MAP.get(val,'white')};font-weight:700"
    st.dataframe(
        df_f[['id','tanggal','kategori','komentar','sentimen','confidence']]
            .style.map(color_sent, subset=['sentimen']),
        use_container_width=True, height=480)
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("⬇️ Download CSV",
            df_f.to_csv(index=False).encode('utf-8'),
            "hasil_sentimen.csv","text/csv")
    with col_dl2:
        st.download_button("⬇️ Download JSON",
            df_f.to_json(orient='records',force_ascii=False).encode('utf-8'),
            "hasil_sentimen.json","application/json")

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:32px 0 16px;
     font-size:11px;color:rgba(255,255,255,0.18);font-family:DM Mono,monospace;line-height:2'>
  Implementasi Analisis Sentimen Komentar Mahasiswa Berbasis IndoBERT<br>
  Smart Campus Feedback System · EAS Kecerdasan Buatan 2025/2026 · UHAMKA 6A STI
</div>""", unsafe_allow_html=True)
