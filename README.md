---
title: Smart Campus Feedback System
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# 🎓 Smart Campus Feedback System
## Implementasi Analisis Sentimen Komentar Mahasiswa Berbasis IndoBERT

> EAS Kecerdasan Buatan 2025/2026 | Kelas 6A STI  
> Universitas Muhammadiyah Prof. Dr. Hamka

---

## 🚀 Setup HuggingFace Token

1. Daftar/login di https://huggingface.co
2. Buka https://huggingface.co/settings/tokens
3. Buat token baru (Read access)
4. Di HF Spaces → **Settings → Secrets**
5. Tambahkan: `HF_TOKEN` = token kamu

## 🤖 Teknologi
- **Model AI**: IndoBERT via HuggingFace Inference API
- **Framework**: Streamlit
- **Visualisasi**: Plotly, Matplotlib, WordCloud
- **RAM Usage**: ~150MB (ringan!)

## ✨ Fitur
- Dashboard sentimen interaktif
- Tren & heatmap kategori  
- Word Cloud per sentimen
- Uji teks real-time
- Download CSV & JSON
