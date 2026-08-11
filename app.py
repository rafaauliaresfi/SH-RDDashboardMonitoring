"""
Dashboard Monitoring Sentimen & Topic Modeling — SH-RD: Protein Cream
======================================================================
Satu halaman panjang (single-page scroll) — menu di sidebar berupa
tautan yang langsung meloncat/scroll ke bagian terkait:
Ringkasan, Analisis Sentimen, Topic Modeling, Tren Ulasan, Insight & Rekomendasi.
Data dibaca otomatis dari MyDrive/skripsi/hasil_prediksi.csv (tanpa upload manual).

Jalankan lokal:
    pip install -r requirements.txt
    streamlit run app.py
Lalu buka http://localhost:8501 di browser.
"""

import ast
import os

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(page_title="SH-RD | Monitoring Ulasan", page_icon="📈", layout="wide")

# Palet sentimen (pola "traffic-light" umum dipakai di dashboard e-commerce
# agar Positif/Netral/Negatif langsung dikenali tanpa perlu membaca label)
WARNA = {
    "Positif": "#16A34A",
    "Netral": "#F59E0B",
    "Negatif": "#DC2626",
    "bg": "#F7F8FA",
    "card": "#FFFFFF",
    "border": "#E5E7EB",
    "text": "#111827",
    "muted": "#6B7280",
    "accent": "#0E7C86",
}
PALET_TOPIK = ["#0E7C86", "#8E5CD9", "#E8A33D", "#3457D5", "#C2609C", "#1AA6B7", "#D9534F", "#6B8E23"]

# Warna khas tiap marketplace, dipakai untuk grafik sebaran platform
# supaya batangnya langsung dikenali (mis. oranye = Shopee, hijau = Tokopedia)
WARNA_PLATFORM = {
    "shopee": "#EE4D2D",
    "tokopedia": "#42B549",
    "lazada": "#0F146D",
    "tiktok": "#010101",
    "blibli": "#0072CE",
    "bukalapak": "#D2273C",
    "jd.id": "#E3232D",
    "zalora": "#000000",
}
PALET_PLATFORM_FALLBACK = ["#0E7C86", "#8E5CD9", "#3457D5", "#E8A33D", "#C2609C", "#1AA6B7"]


def warna_platform(nama):
    kunci = str(nama).strip().lower()
    for cocok, warna in WARNA_PLATFORM.items():
        if cocok in kunci:
            return warna
    idx = abs(hash(kunci)) % len(PALET_PLATFORM_FALLBACK)
    return PALET_PLATFORM_FALLBACK[idx]


if "topik_cache" not in st.session_state:
    st.session_state.topik_cache = {}

# =========================================================
# CSS — TAMPILAN KUSTOM (bukan tema default Streamlit)
# =========================================================
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,600&display=swap');

    html {{ scroll-behavior: smooth; }}
    html, body, [class*="css"]  {{
        font-family: 'Inter', sans-serif;
    }}
    .stApp {{
        background-color: {WARNA['bg']};
    }}
   #MainMenu, footer {{ visibility: hidden; }}

    /* offset agar judul bagian tidak tertutup saat discroll via anchor */
    .anchor {{ position: relative; top: -12px; visibility: hidden; }}

    section[data-testid="stSidebar"] {{
        background-color: #111827;
    }}
    section[data-testid="stSidebar"] * {{
        color: #E7EAF3 !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div > div {{
        background-color: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] * {{
        color: #0B0F1A !important;
        -webkit-text-fill-color: #0B0F1A !important;
        font-weight: 500 !important;
    }}

    .side-nav a {{
        display: block;
        padding: 8px 12px;
        margin-bottom: 2px;
        border-radius: 8px;
        color: #C9D3E8 !important;
        text-decoration: none !important;
        font-size: 13px;
        font-weight: 500;
        transition: background 0.15s ease;
    }}
    .side-nav a:hover {{
        background: rgba(255,255,255,0.08);
        color: #FFFFFF !important;
    }}

    h1, h2, h3 {{
        font-family: 'Source Serif 4', serif;
        color: {WARNA['text']};
    }}
    .eyebrow {{
        font-size: 11px;
        letter-spacing: 0.08em;
        font-weight: 600;
        text-transform: uppercase;
        color: {WARNA['muted']};
        margin-bottom: 2px;
    }}
    .big-title {{
        font-family: 'Source Serif 4', serif;
        font-size: 26px;
        font-weight: 600;
        color: {WARNA['text']};
        margin: 34px 0 18px 0;
        border-bottom: 2px solid {WARNA['border']};
        padding-bottom: 10px;
    }}
    .big-title:first-of-type {{ margin-top: 0; }}
    .sub-caption {{
        font-size: 13px;
        color: {WARNA['muted']};
        margin-top: -10px;
        margin-bottom: 16px;
    }}

    div[data-testid="stMetric"] {{
        background-color: {WARNA['card']};
        border: 1px solid {WARNA['border']};
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 11px !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: {WARNA['muted']} !important;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {WARNA['card']};
        border-radius: 12px;
        border: 1px solid {WARNA['border']} !important;
    }}

    .quote-card {{
        background: #FAFBFC;
        border: 1px solid {WARNA['border']};
        border-left: 4px solid {WARNA['accent']};
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }}
    .quote-meta {{
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        color: {WARNA['muted']};
        margin-bottom: 6px;
    }}
    .quote-user {{ font-weight: 600; color: {WARNA['text']}; }}
    .quote-badge {{
        display: inline-block;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        padding: 2px 8px;
        border-radius: 4px;
        background: #EEF1F8;
        color: {WARNA['accent']};
    }}
    .quote-text {{
        font-size: 14px;
        color: {WARNA['text']};
        font-style: italic;
        line-height: 1.5;
    }}
    .quote-text::before {{ content: '“'; }}
    .quote-text::after {{ content: '”'; }}

    .legend-row {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 4px;
        border-bottom: 1px solid #F0F1F4;
        font-size: 13px;
    }}
    .legend-dot {{
        width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
    }}
    .legend-label {{ flex: 1; color: {WARNA['text']}; }}
    .legend-value {{ font-weight: 600; color: {WARNA['text']}; }}

    .k-banner {{
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        color: #065F46;
        margin-bottom: 10px;
    }}
    .k-banner b {{ color: #065F46; }}

    div.stButton > button {{
        border-radius: 6px;
        border: 1px solid {WARNA['border']};
        font-size: 12px;
        padding: 2px 10px;
        background: white;
        color: {WARNA['accent']};
    }}
    div.stButton > button:hover {{
        border-color: {WARNA['accent']};
        color: {WARNA['accent']};
    }}
    div.stDownloadButton > button {{
        border-radius: 6px;
        border: 1px solid {WARNA['accent']};
        font-size: 13px;
        background: {WARNA['accent']};
        color: white;
    }}
    div.stDownloadButton > button:hover {{
        opacity: 0.9;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def anchor(id_):
    st.markdown(f'<div class="anchor" id="{id_}"></div>', unsafe_allow_html=True)


def donat(labels, values, colors, center_number, center_label, height=320):
    fig = go.Figure(
        go.Pie(
            labels=labels, values=values, hole=0.62,
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value} ulasan (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(
        showlegend=False, height=height, margin=dict(t=10, b=10, l=10, r=10),
        annotations=[dict(
            text=f"<b style='font-size:26px'>{center_number}</b><br><span style='font-size:11px;color:{WARNA['muted']}'>{center_label}</span>",
            showarrow=False, font=dict(family="Inter"),
        )],
    )
    return fig


def bintang(rating):
    try:
        n = int(round(float(rating)))
    except (ValueError, TypeError):
        return ""
    return "★" * n + "☆" * (5 - n)


# =========================================================
# SUMBER DATA — otomatis coba beberapa lokasi umum
# (Google Drive di Colab, atau satu folder dengan app.py di lokal)
# =========================================================
KANDIDAT_PATH = [
    "/content/drive/MyDrive/skripsi/hasil_prediksi.csv",  # Colab + Google Drive
    "hasil_prediksi.csv",                                   # lokal, satu folder dengan app.py
    "data/hasil_prediksi.csv",                              # lokal, dalam subfolder data/
]
FILE_PATH = next((p for p in KANDIDAT_PATH if os.path.exists(p)), None)

if FILE_PATH is None:
    st.title("📈 SH-RD — Dashboard Monitoring Ulasan")
    daftar = "\n".join(f"- `{p}`" for p in KANDIDAT_PATH)
    st.error(
        f"File `hasil_prediksi.csv` tidak ditemukan. Sudah dicoba di lokasi berikut:\n{daftar}\n\n"
        "Taruh file CSV di salah satu lokasi di atas (di Colab: pastikan Google Drive sudah di-mount; "
        "di lokal: taruh satu folder dengan app.py)."
    )
    st.stop()

df = pd.read_csv(FILE_PATH)
kolom = df.columns.tolist()
opsi_kolom = ["(tidak ada)"] + kolom


def idx_default(nama_list, *targets):
    for target in targets:
        if target in nama_list:
            return nama_list.index(target)
    return 0


st.sidebar.markdown("### 📈 SH-RD Monitoring")
st.sidebar.caption(f"{len(df):,} ulasan dimuat")
st.sidebar.caption(f"Sumber: {FILE_PATH}")
st.sidebar.markdown("---")
with st.sidebar.expander("Pemetaan kolom", expanded=False):
    col_teks = st.selectbox("Teks ulasan", kolom, index=idx_default(kolom, "content"))
    col_label = st.selectbox("Label sentimen", kolom, index=idx_default(kolom, "sentimen_pred"))
    col_rating = st.selectbox("Rating", opsi_kolom, index=idx_default(opsi_kolom, "rating"))
    col_tanggal = st.selectbox("Tanggal", opsi_kolom, index=idx_default(opsi_kolom, "tanggal"))
    col_platform = st.selectbox("Platform", opsi_kolom, index=idx_default(opsi_kolom, "platform"))
    col_username = st.selectbox("Username", opsi_kolom, index=idx_default(opsi_kolom, "username"))
    col_token = st.selectbox("Token stemming", opsi_kolom, index=idx_default(opsi_kolom, "stemming"))

df[col_label] = df[col_label].astype(str).str.strip().str.capitalize()

st.sidebar.markdown("### Navigasi")
st.sidebar.markdown(
    """
    <div class="side-nav">
        <a href="#ringkasan"> Ringkasan</a>
        <a href="#sentimen"> Analisis Sentimen</a>
        <a href="#topik"> Topic Modeling</a>
        <a href="#tren"> Tren Ulasan</a>
        <a href="#insight"> Insight &amp; Rekomendasi</a>
    </div>
    """,
    unsafe_allow_html=True,
)


def ambil_token(teks, token_str):
    if isinstance(token_str, str):
        try:
            parsed = ast.literal_eval(token_str)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    if isinstance(teks, str):
        return [w for w in teks.lower().split() if len(w) > 2]
    return []


def _fit_lda(docs, k):
    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dictionary.doc2bow(d) for d in docs]
    lda = LdaModel(corpus, id2word=dictionary, num_topics=k, random_state=42,
                    passes=10, iterations=150, alpha="auto", eta="auto")
    coherence = CoherenceModel(model=lda, texts=docs, dictionary=dictionary, coherence="c_v").get_coherence()
    return lda, dictionary, corpus, coherence


def cari_topik_otomatis(kelas, k_min=2, k_max=8):
    """Coba k=k_min..k_max, pilih k dengan coherence (c_v) tertinggi,
    lalu latih ulang model final di k terbaik dengan jumlah passes lebih tinggi."""
    key = f"{kelas}_{col_teks}_{col_token}"
    if key in st.session_state.topik_cache:
        return st.session_state.topik_cache[key]

    subset = df[df[col_label] == kelas].copy().reset_index(drop=True)
    subset["_token"] = [ambil_token(t, tok) for t, tok in zip(subset[col_teks], subset[col_token])]
    mask = subset["_token"].apply(len) > 0
    subset = subset[mask].reset_index(drop=True)
    if len(subset) < 20:
        return None
    docs = subset["_token"].tolist()

    percobaan = []
    terbaik = None
    for k in range(k_min, k_max + 1):
        lda_k, dic_k, corpus_k, coh_k = _fit_lda(docs, k)
        percobaan.append({"k": k, "coherence": coh_k})
        if terbaik is None or coh_k > terbaik["coherence"]:
            terbaik = {"k": k, "coherence": coh_k}

    k_terbaik = terbaik["k"]
    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dictionary.doc2bow(d) for d in docs]
    lda_final = LdaModel(corpus, id2word=dictionary, num_topics=k_terbaik, random_state=42,
                          passes=20, iterations=300, alpha="auto", eta="auto")
    coherence_final = CoherenceModel(model=lda_final, texts=docs, dictionary=dictionary,
                                      coherence="c_v").get_coherence()

    topik_dominan = []
    for bow in corpus:
        dist = lda_final.get_document_topics(bow)
        topik_dominan.append(max(dist, key=lambda x: x[1])[0] if dist else -1)
    subset["topik"] = topik_dominan

    hasil = {
        "lda": lda_final, "dictionary": dictionary, "subset": subset,
        "k": k_terbaik, "coherence": coherence_final, "percobaan": percobaan,
    }
    st.session_state.topik_cache[key] = hasil
    return hasil


def render_quote_cards(subset_df):
    for _, row in subset_df.iterrows():
        platform = row[col_platform] if col_platform != "(tidak ada)" else ""
        username = row[col_username] if col_username != "(tidak ada)" else "Pengguna"
        rating = row[col_rating] if col_rating != "(tidak ada)" else None
        st.markdown(
            f"""
            <div class="quote-card">
                <div class="quote-meta">
                    <span class="quote-user">{username}</span>
                    <span class="quote-badge">{platform}</span>
                </div>
                <div style="color:#E8A33D;font-size:13px;margin-bottom:4px;">{bintang(rating)}</div>
                <div class="quote-text">{row[col_teks]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def filter_platform_rating(subset, key_prefix):
    """Tampilkan filter platform & rating, kembalikan subset yang sudah difilter."""
    fcol1, fcol2 = st.columns(2)
    hasil = subset
    if col_platform != "(tidak ada)":
        opsi_platform = ["Semua"] + sorted(subset[col_platform].dropna().unique().tolist())
        pilih_platform = fcol1.selectbox("Platform", opsi_platform, key=f"{key_prefix}_platform")
        if pilih_platform != "Semua":
            hasil = hasil[hasil[col_platform] == pilih_platform]
    if col_rating != "(tidak ada)":
        opsi_rating = ["Semua"] + sorted(subset[col_rating].dropna().unique().tolist(), reverse=True)
        pilih_rating = fcol2.selectbox("Rating", opsi_rating, key=f"{key_prefix}_rating")
        if pilih_rating != "Semua":
            hasil = hasil[hasil[col_rating] == pilih_rating]
    return hasil


@st.dialog("Ulasan Pengguna", width="large")
def popup_ulasan_kelas(kelas, lingkup_df=None):
    st.markdown(f"**Kelas Sentimen: {kelas}**")
    sumber = lingkup_df if lingkup_df is not None else df
    subset = sumber[sumber[col_label] == kelas]
    terfilter = filter_platform_rating(subset, key_prefix=f"kelas_{kelas}")
    st.caption(f"Menampilkan {len(terfilter):,} dari {len(subset):,} ulasan")
    with st.container(height=440):
        render_quote_cards(terfilter)


@st.dialog("Ulasan pada Topik Ini", width="large")
def popup_ulasan_topik(kelas, topik_idx, kata_kunci, subset):
    st.markdown(f"**{kelas} — Topik {topik_idx + 1}:** {kata_kunci}")
    ulasan_topik = subset[subset["topik"] == topik_idx].reset_index(drop=True)
    terfilter = filter_platform_rating(ulasan_topik, key_prefix=f"topik_{kelas}_{topik_idx}")
    st.caption(f"Menampilkan {len(terfilter):,} dari {len(ulasan_topik):,} ulasan pada topik ini")
    with st.container(height=440):
        render_quote_cards(terfilter)


# =========================================================
# BAGIAN 1 — RINGKASAN
# =========================================================
# logo + judul
LOGO_SVG = """
<svg width="52" height="52" viewBox="0 0 52 52" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="52" height="52" rx="14" fill="#0E7C86"/>
  <path d="M12 18c0-2.2 1.8-4 4-4h20c2.2 0 4 1.8 4 4v10c0 2.2-1.8 4-4 4H24l-6 5v-5h-2c-2.2 0-4-1.8-4-4V18z"
        fill="white" opacity="0.12"/>
  <path d="M12 18c0-2.2 1.8-4 4-4h20c2.2 0 4 1.8 4 4v10c0 2.2-1.8 4-4 4H24l-6 5v-5h-2c-2.2 0-4-1.8-4-4V18z"
        stroke="white" stroke-width="1.6"/>
  <rect x="18" y="22" width="3.4" height="6" rx="1" fill="#16A34A"/>
  <rect x="23.5" y="19" width="3.4" height="9" rx="1" fill="#F59E0B"/>
  <rect x="29" y="16" width="3.4" height="12" rx="1" fill="#FCA5A5"/>
</svg>
"""

st.markdown(
    f"""<div style="display:flex; align-items:center; gap:16px; padding:4px 0 22px 0;">
<div>{LOGO_SVG}</div>
<div>
<div style="font-family:'Source Serif 4', serif; font-size:35px; font-weight:700; color:{WARNA['text']}; line-height:1.2;">SH-RD Monitoring</div>
<div style="font-size:13px; color:{WARNA['muted']};">Dashboard Monitoring Sentimen &amp; Topic Modeling Ulasan Marketplace</div>
</div>
</div>""",
    unsafe_allow_html=True,
)

anchor("ringkasan")
st.markdown('<div class="big-title">Ringkasan</div>', unsafe_allow_html=True)


total = len(df)
counts = df[col_label].value_counts()
urutan = [k for k in ["Positif", "Netral", "Negatif"] if k in counts.index]

colR1, colR2 = st.columns([1, 2])
with colR1:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Total Ulasan</div>', unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:40px;font-weight:700;color:{WARNA['text']};'>{total:,}</div>", unsafe_allow_html=True)
        if col_tanggal != "(tidak ada)":
            tgl_valid = pd.to_datetime(df[col_tanggal], errors="coerce").dropna()
            if len(tgl_valid):
                st.caption(f"Periode {tgl_valid.min().strftime('%d %b %Y')} – {tgl_valid.max().strftime('%d %b %Y')}")
with colR2:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Sebaran Ulasan per Platform</div>', unsafe_allow_html=True)
        if col_platform != "(tidak ada)":
            vc_platform = df[col_platform].value_counts()
            df_plat = pd.DataFrame({
                "Platform": vc_platform.index.astype(str),
                "Jumlah": vc_platform.values,
            })
            fig_plat = px.bar(
                df_plat, x="Jumlah", y="Platform", orientation="h",
                color="Platform",
                color_discrete_map={p: warna_platform(p) for p in df_plat["Platform"]},
            )
            fig_plat.update_traces(
                hovertemplate="<b>%{y}</b><br>Jumlah ulasan: %{x:,}<extra></extra>",
            )
            fig_plat.update_layout(
                showlegend=False, height=max(180, 60 * len(df_plat)),
                margin=dict(t=10, b=10, l=10, r=20),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title="Jumlah ulasan", yaxis_title="",
            )
            fig_plat.update_xaxes(showgrid=True, gridcolor="#F0F1F4")
            st.plotly_chart(fig_plat, use_container_width=True)
        else:
            st.caption("Kolom platform tidak dipetakan.")

if col_rating != "(tidak ada)":
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Distribusi Rating</div>', unsafe_allow_html=True)
        rc = df[col_rating].value_counts().sort_index(ascending=False)
        df_rating = pd.DataFrame({
            "Rating": rc.index.astype(str),
            "Jumlah": rc.values,
        })
        fig2 = px.bar(
            df_rating, x="Rating", y="Jumlah",
            color="Jumlah", color_continuous_scale=["#0E7C86", "#E6F4F3", WARNA["accent"]],
        )
        fig2.update_traces(
            hovertemplate="<b>Rating %{x}</b><br>Jumlah ulasan: %{y:,}<extra></extra>",
        )
        fig2.update_layout(
            showlegend=False, coloraxis_showscale=False, height=280,
            margin=dict(t=10, b=10, l=10, r=10),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Rating", yaxis_title="Jumlah",
        )
        fig2.update_yaxes(showgrid=True, gridcolor="#F0F1F4")
        st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# BAGIAN 2 — ANALISIS SENTIMEN
# =========================================================
anchor("sentimen")
st.markdown('<div class="big-title">Analisis Sentimen</div>', unsafe_allow_html=True)


if col_platform != "(tidak ada)":
    opsi_plat_sentimen = ["Semua Platform"] + sorted(df[col_platform].dropna().unique().tolist())
    plat_pilihan = st.selectbox("Filter platform", opsi_plat_sentimen, key="filter_sentimen_platform")
    df_sentimen = df if plat_pilihan == "Semua Platform" else df[df[col_platform] == plat_pilihan]
else:
    plat_pilihan = "Semua Platform"
    df_sentimen = df

total_s = len(df_sentimen)
counts_s = df_sentimen[col_label].value_counts()
urutan_s = [k for k in ["Positif", "Netral", "Negatif"] if k in counts_s.index]

colA, colB, colC = st.columns([1, 2, 1])
with colB:
    with st.container(border=True):
        st.markdown(f'<div class="eyebrow">Distribusi Sentimen — {plat_pilihan}</div>', unsafe_allow_html=True)
        if total_s == 0:
            st.info("Tidak ada ulasan pada platform ini.")
        else:
            fig = donat(urutan_s, [counts_s[k] for k in urutan_s], [WARNA[k] for k in urutan_s], f"{total_s:,}", "ULASAN")
            st.plotly_chart(fig, use_container_width=True)
            for k in urutan_s:
                pct = counts_s[k] / total_s * 100
                rowL, rowR = st.columns([5, 1])
                with rowL:
                    st.markdown(
                        f"""<div class="legend-row" style="border-bottom:none;">
                            <span class="legend-dot" style="background:{WARNA[k]}"></span>
                            <span class="legend-label">{k}</span>
                            <span class="legend-value">{counts_s[k]:,} ({pct:.1f}%)</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                with rowR:
                    if st.button("Lihat", key=f"lihat_{k}_{plat_pilihan}"):
                        popup_ulasan_kelas(k, lingkup_df=df_sentimen)

# =========================================================
# BAGIAN 3 — TOPIC MODELING
# =========================================================
anchor("topik")
st.markdown('<div class="big-title">Topic Modeling</div>', unsafe_allow_html=True)

kelas_tersedia = [k for k in ["Positif", "Netral", "Negatif"] if k in df[col_label].unique()]
sudah_ada_cache = any(f"{kelas}_{col_teks}_{col_token}" in st.session_state.topik_cache for kelas in kelas_tersedia)

jalankan = st.button(
    "🔍 Jalankan Topic Modeling Otomatis (semua kelas)" if not sudah_ada_cache else "🔁 Jalankan Ulang Topic Modeling",
    key="btn_topik_semua",
)

if jalankan:
    for kelas in kelas_tersedia:
        key = f"{kelas}_{col_teks}_{col_token}"
        st.session_state.topik_cache.pop(key, None)

if jalankan or sudah_ada_cache:
    for kelas in kelas_tersedia:
        with st.spinner(f"Mencari k terbaik untuk kelas {kelas}..."):
            hasil = cari_topik_otomatis(kelas)
        with st.container(border=True):
            st.markdown(f'<div class="eyebrow">Skema — {kelas}</div>', unsafe_allow_html=True)
            if hasil is None:
                st.warning(f"Data terlalu sedikit untuk kelas {kelas} (minimal 20 ulasan bertoken).")
                continue
            lda, subset, k = hasil["lda"], hasil["subset"], hasil["k"]
            st.markdown(
                f"""<div class="k-banner">Ditemukan <b>{k} topik utama</b> dari
                <b>{len(subset):,} ulasan</b> yang dianalisis pada kelas {kelas}</div>""",
                unsafe_allow_html=True,
            )

            n_col = min(k, 4)
            cols = st.columns(n_col)
            for t in range(k):
                kata_kunci = ", ".join(w for w, _ in lda.show_topic(t, topn=5))
                with cols[t % n_col]:
                    wc = WordCloud(width=300, height=200, background_color="white",
                                   color_func=lambda *a, **kw: PALET_TOPIK[t % len(PALET_TOPIK)]
                                   ).generate_from_frequencies({w: p for w, p in lda.show_topic(t, topn=20)})
                    fig, ax = plt.subplots(figsize=(2.4, 1.6))
                    ax.imshow(wc); ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption(f"**Topik {t+1}** · {kata_kunci}")
                    if st.button("Lihat ulasan", key=f"btn_{kelas}_{t}", use_container_width=True):
                        popup_ulasan_topik(kelas, t, kata_kunci, subset)
else:
    st.info("Klik tombol di atas untuk memulai pencarian k otomatis (bisa memakan waktu beberapa menit).")

# =========================================================
# BAGIAN 4 — TREN ULASAN
# =========================================================
anchor("tren")
st.markdown('<div class="big-title">Tren Ulasan</div>', unsafe_allow_html=True)
if col_tanggal == "(tidak ada)":
    st.warning("Kolom tanggal belum dipetakan.")
else:
    tgl_semua = pd.to_datetime(df[col_tanggal], errors="coerce").dropna()
    tgl_min, tgl_max = tgl_semua.min().date(), tgl_semua.max().date()

    colF1, colF2 = st.columns([2, 1])
    with colF1:
        rentang_waktu = st.slider(
            "Pilih rentang waktu (bulan-tahun)",
            min_value=tgl_min, max_value=tgl_max,
            value=(tgl_min, tgl_max),
            format="MMM YYYY",
            key="rentang_waktu_tren",
        )
    with colF2:
        granularitas = st.radio("Tampilkan per", ["Bulan", "Minggu", "Hari"], horizontal=True, index=0)

    freq_map = {"Hari": "D", "Minggu": "W", "Bulan": "M"}

    df_tren = df.copy()
    df_tren[col_tanggal] = pd.to_datetime(df_tren[col_tanggal], errors="coerce")
    df_tren = df_tren.dropna(subset=[col_tanggal])

    mask_periode = (df_tren[col_tanggal].dt.date >= rentang_waktu[0]) & (df_tren[col_tanggal].dt.date <= rentang_waktu[1])
    df_periode = df_tren[mask_periode]

    panjang_hari = (rentang_waktu[1] - rentang_waktu[0]).days
    batas_sebelumnya_akhir = rentang_waktu[0] - pd.Timedelta(days=1)
    batas_sebelumnya_awal = batas_sebelumnya_akhir - pd.Timedelta(days=panjang_hari)
    mask_sebelumnya = (df_tren[col_tanggal].dt.date >= batas_sebelumnya_awal) & (df_tren[col_tanggal].dt.date <= batas_sebelumnya_akhir)
    df_sebelumnya = df_tren[mask_sebelumnya]

    def hitung_ringkasan(subset):
        total = len(subset)
        if total == 0:
            return {"total": 0, "pct_pos": 0, "pct_neg": 0}
        cnt = subset[col_label].value_counts()
        return {
            "total": total,
            "pct_pos": cnt.get("Positif", 0) / total * 100,
            "pct_neg": cnt.get("Negatif", 0) / total * 100,
        }

    ring_sekarang = hitung_ringkasan(df_periode)
    ring_sebelumnya = hitung_ringkasan(df_sebelumnya)

    label_periode = f"{rentang_waktu[0].strftime('%b %Y')} \u2013 {rentang_waktu[1].strftime('%b %Y')}"
    st.caption(f"Menampilkan periode **{label_periode}**, dibandingkan otomatis dengan periode sebelumnya yang panjangnya sama.")

    c1, c2, c3 = st.columns(3)
    with c1:
        delta_total = ring_sekarang["total"] - ring_sebelumnya["total"]
        st.metric("Total Ulasan pada Periode Ini", f"{ring_sekarang['total']:,}",
                   delta=f"{delta_total:+,} ulasan" if ring_sebelumnya["total"] else None)
    with c2:
        st.metric("Persentase Positif", f"{ring_sekarang['pct_pos']:.1f}%",
                   delta=f"{ring_sekarang['pct_pos'] - ring_sebelumnya['pct_pos']:+.1f} poin" if ring_sebelumnya["total"] else None)
    with c3:
        st.metric("Persentase Negatif", f"{ring_sekarang['pct_neg']:.1f}%",
                   delta=f"{ring_sekarang['pct_neg'] - ring_sebelumnya['pct_neg']:+.1f} poin" if ring_sebelumnya["total"] else None,
                   delta_color="inverse")

    df_periode = df_periode.copy()
    df_periode["periode"] = df_periode[col_tanggal].dt.to_period(freq_map[granularitas]).dt.to_timestamp()
    tren = df_periode.groupby(["periode", col_label]).size().reset_index(name="jumlah")

    with st.container(border=True):
        st.markdown(f'<div class="eyebrow">Tren Ulasan \u2014 {label_periode}</div>', unsafe_allow_html=True)
        if tren.empty:
            st.info("Tidak ada ulasan pada rentang waktu ini.")
        else:
            fig = px.line(tren, x="periode", y="jumlah", color=col_label, markers=True,
                          color_discrete_map=WARNA)
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", legend_title=None, height=420)
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Cara membaca grafik ini"):
        st.markdown(
            "- Setiap **garis** mewakili satu kategori sentimen: hijau (Positif), kuning (Netral), merah (Negatif).\n"
            "- **Sumbu mendatar** menunjukkan waktu, **sumbu tegak** menunjukkan jumlah ulasan.\n"
            "- **Titik yang naik tajam** menandakan lonjakan jumlah ulasan pada periode tersebut \u2014 bisa jadi efek promo, campaign, atau viral di media sosial.\n"
            "- **Garis merah yang ikut naik** adalah sinyal untuk segera ditindaklanjuti \u2014 gunakan tombol \"Lihat\" pada halaman Analisis Sentimen untuk membaca ulasan negatif pada periode tersebut secara langsung.\n"
            "- Kartu angka di atas grafik membandingkan periode yang sedang dipilih dengan periode sebelumnya secara otomatis, jadi tidak perlu membaca grafik untuk tahu tren naik atau turun."
        )

# =========================================================
# BAGIAN 5 — INSIGHT & REKOMENDASI
# =========================================================
anchor("insight")
st.markdown('<div class ="big-title">Insight & Rekomendasi</div>', unsafe_allow_html=True)

if col_platform != "(tidak ada)":
    opsi_plat_insight = ["Semua Platform"] + sorted(df[col_platform].dropna().unique().tolist())
    plat_insight = st.selectbox("Filter Platform", opsi_plat_insight, key="filter_insight_platform")
    df_insight = df if plat_insight == "Semua Platform" else df[df[col_platform] == plat_insight]
else:
    plat_insight = "Semua Platform"
    df_insight = df

total_i = len(df_insight)
counts_i = df_insight[col_label].value_counts()

c1, c2 = st.columns(2)
c1.metric("Sentimen Positif", f"{counts_i.get('Positif', 0) / total_i * 100:.1f}%" if total_i else "-")
c2.metric("Sentimen Negatif", f"{counts_i.get('Negatif', 0) / total_i * 100:.1f}%" if total_i else "-")

with st.container(border=True):
    st.markdown('<div class="eyebrow">Insight Otomatis</div>', unsafe_allow_html=True)
    key_neg = f"Negatif_{col_teks}_{col_token}"
    key_pos = f"Positif_{col_teks}_{col_token}"
    hn = st.session_state.topik_cache.get(key_neg)
    hp = st.session_state.topik_cache.get(key_pos)
    baris_insight = []
    if hn and hp:
        kn = [w for t in range(hn["k"]) for w, _ in hn["lda"].show_topic(t, topn=5)]
        kp = [w for t in range(hp["k"]) for w, _ in hp["lda"].show_topic(t, topn=5)]
        kn_unik = list(dict.fromkeys(kn))
        kp_unik = list(dict.fromkeys(kp))
        baris_insight = [
            f"Cakupan: {plat_insight} ({total_i:,} ulasan)",
            f"Sentimen Positif didominasi kata: {', '.join(kp_unik[:8]) if kp_unik else '-'}",
            f"Sentimen Negatif didominasi kata: {', '.join(kn_unik[:8]) if kn_unik else '-'}",
            "Kepuasan pelanggan banyak ditopang aspek pengiriman & sensorik, sedangkan keluhan berpusat pada efikasi inti produk.",
        ]
        st.markdown("\n".join(f"- {b}" for b in baris_insight))
    else:
        st.info(
            "Jalankan Topic Modeling untuk kelas Positif dan Negatif terlebih dahulu di bagian Topic Modeling.",
            icon="ℹ️",
        )

with st.container(border=True):
    st.markdown('<div class="eyebrow">Catatan Tim</div>', unsafe_allow_html=True)
    catatan = st.text_area("Rekomendasi tindak lanjut", height=140, label_visibility="collapsed",
                            key="catatan_tim")

with st.container(border=True):
    st.markdown('<div class="eyebrow">Unduh Hasil</div>', unsafe_allow_html=True)
    st.caption(f"Berlaku untuk cakupan filter induk saat ini: **{plat_insight}**")
    dl1, dl2 = st.columns(2)
    with dl1:
        baris_laporan = ["LAPORAN INSIGHT & REKOMENDASI — SH-RD", f"Cakupan: {plat_insight}",
                          f"Total ulasan: {total_i:,}"]
        if total_i:
            baris_laporan.append(f"Sentimen Positif: {counts_i.get('Positif', 0) / total_i * 100:.1f}%")
            baris_laporan.append(f"Sentimen Negatif: {counts_i.get('Negatif', 0) / total_i * 100:.1f}%")
        laporan = "\n".join(baris_laporan)
        if baris_insight:
            laporan += "\n\n" + "\n".join(f"- {b}" for b in baris_insight)
        if catatan:
            laporan += f"\n\nCatatan Tim:\n{catatan}"
        st.download_button(
            "⬇️ Unduh Insight (.txt)",
            data=laporan.encode("utf-8"),
            file_name=f"insight_{plat_insight.replace(' ', '_').lower()}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "⬇️ Unduh Data Ulasan (.csv)",
            data=df_insight.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"ulasan_{plat_insight.replace(' ', '_').lower()}.csv",
            mime="text/csv",
            use_container_width=True,
        )
