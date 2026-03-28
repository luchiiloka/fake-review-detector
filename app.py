"""
Fake Review Detector — Streamlit Web App
Run with: streamlit run app.py
"""

import re
import warnings
import os
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Colourful Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

  /* ── Base ── */
  .stApp {
    background: linear-gradient(135deg, #0d0221 0%, #0a0a2e 40%, #1a0a2e 70%, #0d1a0a 100%);
    color: #e0e0e0;
    font-family: 'Inter', sans-serif;
  }
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0221 0%, #0a0a2e 100%);
    border-right: 1px solid #2a1a4a;
  }

  /* ── Metric cards ── */
  div[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(100,0,200,0.15), rgba(0,150,255,0.1));
    border: 1px solid rgba(150,80,255,0.4);
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 4px 20px rgba(120,0,255,0.15);
  }
  div[data-testid="metric-container"] label {
    color: #aa88ff !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 900 !important;
  }

  /* ── Headers ── */
  h1 {
    background: linear-gradient(90deg, #ff4dc4, #a855f7, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 900 !important;
  }
  h2 {
    color: #ffffff !important;
    border-left: 4px solid #a855f7;
    padding-left: 12px;
    background: linear-gradient(90deg, rgba(168,85,247,0.1), transparent);
    border-radius: 0 8px 8px 0;
    padding: 6px 12px;
  }
  h3 { color: #c4b5fd !important; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    color: #888 !important;
    border-radius: 8px;
    font-weight: 600;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: #ffffff !important;
    border-radius: 8px;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 10px 28px;
    box-shadow: 0 4px 15px rgba(124,58,237,0.4);
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #9333ea, #3b82f6);
    box-shadow: 0 6px 20px rgba(124,58,237,0.6);
    transform: translateY(-1px);
  }

  /* ── Sliders ── */
  .stSlider [data-baseweb="slider"] { color: #a855f7; }

  /* ── Score badges ── */
  .badge-genuine  { background: linear-gradient(135deg,#064e3b,#065f46); color:#34d399; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; border:1px solid #34d399; }
  .badge-low      { background: linear-gradient(135deg,#713f12,#78350f); color:#fbbf24; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; border:1px solid #fbbf24; }
  .badge-moderate { background: linear-gradient(135deg,#7c2d12,#9a3412); color:#fb923c; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; border:1px solid #fb923c; }
  .badge-high     { background: linear-gradient(135deg,#7f1d1d,#991b1b); color:#f87171; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; border:1px solid #f87171; }
  .badge-fake     { background: linear-gradient(135deg,#4c0519,#881337); color:#fb7185; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; border:1px solid #fb7185; }

  /* ── Review cards ── */
  .review-card {
    background: linear-gradient(135deg, rgba(15,10,40,0.9), rgba(10,15,40,0.95));
    border: 1px solid #2a1a4a;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 14px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .review-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(120,0,255,0.2);
  }

  /* ── Sidebar elements ── */
  .stCheckbox label { color: #c4b5fd !important; }
  .stMultiSelect [data-baseweb="tag"] {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
  }

  /* ── Divider ── */
  hr { border-color: rgba(168,85,247,0.3) !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0a0a2e; }
  ::-webkit-scrollbar-thumb { background: #7c3aed; border-radius: 3px; }

  /* ── Info boxes ── */
  .stAlert { background: rgba(124,58,237,0.1) !important; border-color: #7c3aed !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# WORD LISTS
# ─────────────────────────────────────────────
VAGUE_WORDS    = {"great","good","bad","nice","terrible","amazing","worst","best",
                  "awesome","horrible","loved","hated","perfect","recommend"}
SPECIFIC_WORDS = {"room","floor","lobby","parking","breakfast","pool","shower","bed",
                  "wifi","staff","neighborhood","elevator","checkout","restaurant",
                  "view","bathroom","towel","pillow","concierge","reception","suite"}

# ─────────────────────────────────────────────
# CORE ANALYSIS (cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_analysis(file_bytes: bytes, filename: str):
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))

    # --- Clean ---
    df["text"]     = df.get("reviews.text", df.get("text", pd.Series([""] * len(df)))).fillna("").astype(str)
    df["rating"]   = pd.to_numeric(df.get("reviews.rating", df.get("rating", 0)), errors="coerce").fillna(0)
    df["username"] = df.get("reviews.username", df.get("username", pd.Series(["anon"] * len(df)))).fillna("anonymous").astype(str)
    df["hotel"]    = df.get("name", df.get("hotel", pd.Series(["Unknown"] * len(df)))).fillna("Unknown Hotel").astype(str)
    df["date_raw"] = df.get("reviews.date", df.get("date", pd.Series([""] * len(df)))).fillna("")

    # Normalize rating to 1-5
    max_rating = df["rating"].max()
    if max_rating > 5:
        df["rating5"] = (df["rating"] / (max_rating / 5)).clip(1, 5).round()
    else:
        df["rating5"] = df["rating"].clip(1, 5)

    # Parse dates
    def parse_date(s):
        try: return pd.to_datetime(s, infer_datetime_format=True)
        except: return pd.NaT
    df["date"]      = df["date_raw"].apply(parse_date)
    df["date_only"] = df["date"].dt.date
    df["word_count"] = df["text"].apply(lambda t: len(t.split()))

    scores  = np.zeros(len(df), dtype=float)
    signals = {f"S{i}": np.zeros(len(df), dtype=bool) for i in range(1, 9)}

    # S1 — Length
    short = df["word_count"] < 15
    s1 = short & ((df["rating5"] == 5) | (df["rating5"] == 1))
    signals["S1"] = s1; scores[s1] += 20

    # S2 — Vague vs Specific
    def vague_ratio(text):
        words = re.findall(r'\b\w+\b', text.lower())
        v = sum(1 for w in words if w in VAGUE_WORDS)
        s = sum(1 for w in words if w in SPECIFIC_WORDS)
        return v / max(s, 1)
    df["vague_ratio"] = df["text"].apply(vague_ratio)
    s2 = df["vague_ratio"] > 3
    signals["S2"] = s2; scores[s2] += 15

    # S3 — Extreme Sentiment
    def polarity(text):
        try: return TextBlob(text).sentiment.polarity
        except: return 0.0
    df["polarity"] = df["text"].apply(polarity)
    s3 = (df["polarity"] > 0.8) | (df["polarity"] < -0.8)
    signals["S3"] = s3; scores[s3] += 15

    # S4 — Duplicates
    texts = df["text"].tolist()
    dup_flags = np.zeros(len(df), dtype=bool)
    dup_cluster = np.full(len(df), -1, dtype=int)
    duplicate_clusters = []
    visited = set()
    cluster_id = 0
    CHUNK = 500
    try:
        vec = TfidfVectorizer(min_df=2, max_features=5000, ngram_range=(1, 2))
        tfidf = vec.fit_transform(texts)
        n = len(df)
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            sims = cosine_similarity(tfidf[start:end], tfidf)
            for i, row_sim in enumerate(sims):
                gi = start + i
                if gi in visited: continue
                matches = [m for m in np.where(row_sim >= 0.80)[0].tolist() if m != gi]
                if matches:
                    members = [gi] + matches
                    duplicate_clusters.append(members)
                    for m in members:
                        dup_flags[m] = True
                        dup_cluster[m] = cluster_id
                        visited.add(m)
                    cluster_id += 1
    except Exception:
        pass
    signals["S4"] = dup_flags; scores[dup_flags] += 20
    df["dup_cluster"] = dup_cluster

    # S5 — Reviewer Behavior
    rdh = df.groupby(["username","date_only"])["hotel"].nunique().reset_index()
    rdh.columns = ["username","date_only","hotels_per_day"]
    multi = set(rdh[rdh["hotels_per_day"] > 1]["username"])
    s5a = df["username"].isin(multi)
    rr = df.groupby("username")["rating5"].agg(["min","max","count"])
    extreme = rr[((rr["min"]==5)&(rr["max"]==5)|(rr["min"]==1)&(rr["max"]==1))&(rr["count"]>=2)].index
    s5b = df["username"].isin(extreme)
    signals["S5"] = s5a | s5b
    scores[s5a] += 15; scores[s5b] += 10

    # S6 — Rating Anomaly
    h5pct = df.groupby("hotel")["rating5"].apply(lambda x: (x==5).sum()/max(len(x),1))
    manip = set(h5pct[h5pct >= 0.90].index)
    s6 = df["hotel"].isin(manip)
    signals["S6"] = s6; scores[s6] += 10

    # S7 — Timing Bursts
    hdc = df.groupby(["hotel","date_only"]).size().reset_index(name="count")
    burst_pairs = set(hdc[hdc["count"]>=5].apply(lambda r:(r["hotel"],r["date_only"]),axis=1))
    s7 = df.apply(lambda r:(r["hotel"],r["date_only"]) in burst_pairs, axis=1)
    signals["S7"] = s7; scores[s7] += 15

    # S8 — Exclamation
    df["excl_count"] = df["text"].apply(lambda t: t.count("!"))
    s8 = df["excl_count"] >= 3
    signals["S8"] = s8; scores[s8] += 10

    # Final score
    df["suspicion_score"] = np.clip(scores, 0, 100).astype(int)
    for k, v in signals.items():
        df[f"signal_{k}"] = v

    signal_cols = [f"signal_S{i}" for i in range(1, 9)]
    df["triggered_signals"] = df[signal_cols].apply(
        lambda r: [f"S{i}" for i in range(1,9) if r[f"signal_S{i}"]], axis=1
    )

    def label(s):
        if s <= 20:  return "Genuine"
        if s <= 40:  return "Low Suspicion"
        if s <= 60:  return "Moderate"
        if s <= 80:  return "High Suspicion"
        return "Likely Fake"
    df["label"] = df["suspicion_score"].apply(label)

    return df, duplicate_clusters

def score_color(s):
    if s <= 20:  return "#22cc66"
    if s <= 40:  return "#cccc22"
    if s <= 60:  return "#ff8800"
    if s <= 80:  return "#ff4400"
    return "#cc0000"

def label_badge(lbl):
    mapping = {
        "Genuine":       "badge-genuine",
        "Low Suspicion": "badge-low",
        "Moderate":      "badge-moderate",
        "High Suspicion":"badge-high",
        "Likely Fake":   "badge-fake",
    }
    css = mapping.get(lbl, "badge-genuine")
    return f'<span class="{css}">{lbl}</span>'

SIGNAL_NAMES = {
    "S1": "Length", "S2": "Vague", "S3": "Extreme", "S4": "Duplicate",
    "S5": "Behavior", "S6": "Rating", "S7": "Timing", "S8": "Exclamation"
}

# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="
      background:linear-gradient(135deg,rgba(124,58,237,0.3),rgba(37,99,235,0.2));
      border:1px solid rgba(168,85,247,0.4);
      border-radius:12px;padding:16px;margin-bottom:16px;text-align:center;
    ">
      <div style="font-size:1.4rem;font-weight:900;
        background:linear-gradient(90deg,#ff4dc4,#a855f7,#38bdf8);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
        🔍 Fake Review<br>Detector
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📂 Upload Reviews CSV")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"],
                                 help="Needs columns: reviews.text, reviews.rating, name, reviews.username, reviews.date")

    # Default to project data if no upload
    default_path = os.path.join(os.path.dirname(__file__), "data", "hotel_reviews.csv")
    use_default = False
    if not uploaded and os.path.exists(default_path):
        use_default = st.checkbox("Use project dataset (hotel_reviews.csv)", value=True)

    st.markdown("---")
    st.markdown("### ⚙️ Filters")
    min_score = st.slider("Min Suspicion Score", 0, 100, 0)
    selected_labels = st.multiselect(
        "Filter by Label",
        ["Genuine", "Low Suspicion", "Moderate", "High Suspicion", "Likely Fake"],
        default=["Genuine", "Low Suspicion", "Moderate", "High Suspicion", "Likely Fake"]
    )

    st.markdown("---")
    st.markdown("### 📖 Signal Guide")
    sig_colors = ["#6366f1","#8b5cf6","#a855f7","#ec4899","#f43f5e","#f97316","#fbbf24","#34d399"]
    sig_icons  = ["📏","🌫️","💥","📋","👤","⭐","⏱️","❗"]
    for i, (k, v) in enumerate(SIGNAL_NAMES.items()):
        c = sig_colors[i]
        ic = sig_icons[i]
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.05)">
          <span style="background:{c}33;border:1px solid {c};border-radius:6px;
                       padding:2px 7px;font-size:0.7rem;font-weight:700;color:{c}">{k}</span>
          <span style="font-size:0.85rem;color:#c4b5fd">{ic} {v}</span>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════
df = None
if uploaded:
    file_bytes = uploaded.read()
    with st.spinner("🔍 Analysing reviews…"):
        df, dup_clusters = run_analysis(file_bytes, uploaded.name)
elif use_default:
    with open(default_path, "rb") as f:
        file_bytes = f.read()
    with st.spinner("🔍 Analysing hotel_reviews.csv…"):
        df, dup_clusters = run_analysis(file_bytes, "hotel_reviews.csv")

# ═══════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div style="
  background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(37,99,235,0.15), rgba(236,72,153,0.1));
  border: 1px solid rgba(168,85,247,0.4);
  border-radius: 20px;
  padding: 36px 40px;
  margin-bottom: 24px;
  box-shadow: 0 8px 40px rgba(124,58,237,0.2);
  position: relative;
  overflow: hidden;
">
  <div style="
    position:absolute;top:-60px;right:-60px;
    width:250px;height:250px;
    background:radial-gradient(circle, rgba(168,85,247,0.15) 0%, transparent 70%);
    border-radius:50%;
  "></div>
  <div style="
    font-size:2.8rem;font-weight:900;
    background:linear-gradient(90deg,#ff4dc4,#a855f7,#38bdf8,#34d399);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    margin-bottom:10px;
  ">🔍 Fake Review Detector</div>
  <div style="color:#c4b5fd;font-size:1rem;font-weight:400;">
    AI-powered fraud intelligence · Detect suspicious, fake & manipulated hotel reviews instantly
  </div>
</div>
""", unsafe_allow_html=True)

if df is None:
    st.info("👈 Upload a CSV file or enable the project dataset in the sidebar to begin.")
    st.markdown("""
    ### Expected CSV Columns
    | Column | Description |
    |---|---|
    | `reviews.text` | Review body text |
    | `reviews.rating` | Numeric rating |
    | `name` | Hotel name |
    | `reviews.username` | Reviewer username |
    | `reviews.date` | Date of review |
    """)
    st.stop()

# ── Apply filters ──
filtered = df[
    (df["suspicion_score"] >= min_score) &
    (df["label"].isin(selected_labels))
]

total      = len(df)
n_genuine  = (df["suspicion_score"] <= 20).sum()
n_sus      = (total - n_genuine)
n_fake     = (df["suspicion_score"] > 80).sum()
n_high     = ((df["suspicion_score"] > 60) & (df["suspicion_score"] <= 80)).sum()

# ── Tabs ──
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview", "🚨 Suspicious Reviews", "🏨 Hotels",
    "📡 Signals", "🌡️ Heatmap", "📋 Duplicates", "🔎 Search"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📊 Dataset Overview")

    # Colourful stat cards
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "📋 Total Reviews",  f"{total:,}",             "",                             "#7c3aed","#2563eb"),
        (c2, "✅ Genuine",         f"{n_genuine:,}",          f"{100*n_genuine/total:.1f}%",  "#059669","#10b981"),
        (c3, "⚠️ Suspicious",     f"{n_sus:,}",              f"{100*n_sus/total:.1f}%",      "#d97706","#f59e0b"),
        (c4, "🔴 High Risk",      f"{n_high+n_fake:,}",      f"{100*(n_high+n_fake)/total:.1f}%","#dc2626","#ef4444"),
        (c5, "🚨 Likely Fake",    f"{n_fake:,}",             f"{100*n_fake/total:.1f}%",     "#9d174d","#ec4899"),
    ]
    for col_obj, label, value, delta, c1h, c2h in cards:
        with col_obj:
            st.markdown(f"""
            <div style="
              background:linear-gradient(135deg,{c1h}22,{c2h}11);
              border:1px solid {c1h}66;
              border-radius:14px;padding:18px;text-align:center;
              box-shadow:0 4px 20px {c1h}33;
            ">
              <div style="font-size:0.7rem;color:{c2h};text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">{label}</div>
              <div style="font-size:1.9rem;font-weight:900;color:#fff">{value}</div>
              <div style="font-size:0.8rem;color:{c2h};margin-top:4px">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # Score distribution donut — hover shows count + pct
    with col_a:
        st.markdown("### Suspicion Score Distribution")
        st.caption("Hover a segment for details · Click legend to hide/show")
        labels_order = ["Genuine", "Low Suspicion", "Moderate", "High Suspicion", "Likely Fake"]
        colors_order = ["#34d399", "#fbbf24", "#fb923c", "#f87171", "#ec4899"]
        counts = [df[df["label"]==l].shape[0] for l in labels_order]
        pcts   = [100*c/total for c in counts]
        fig_donut = go.Figure(go.Pie(
            labels=labels_order, values=counts,
            hole=0.55,
            marker=dict(colors=colors_order, line=dict(color="#0a0a2e", width=3)),
            textinfo="label+percent",
            textfont=dict(color="white", size=12),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
            pull=[0.04, 0, 0, 0.04, 0.08],   # pull out genuine + fake slices
        ))
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", showlegend=True,
            legend=dict(bgcolor="rgba(255,255,255,0.05)", bordercolor="#7c3aed",
                        font_color="white", orientation="v"),
            margin=dict(t=10, b=10, l=10, r=10), height=360,
            hoverlabel=dict(bgcolor="#1a1a2e", font_color="white", bordercolor="#7c3aed"),
        )
        st.plotly_chart(fig_donut, use_container_width=True, key="donut")

    # Rating distribution bar — hover shows exact count, zoom enabled
    with col_b:
        st.markdown("### Rating Distribution by Category")
        st.caption("Hover bars · Scroll or box-select to zoom · Double-click to reset")
        gen_r   = df[df["suspicion_score"] < 20]["rating5"].value_counts().sort_index()
        low_r   = df[(df["suspicion_score"]>=21)&(df["suspicion_score"]<=40)]["rating5"].value_counts().sort_index()
        fake_r  = df[df["suspicion_score"] > 50]["rating5"].value_counts().sort_index()
        fig_bar = go.Figure()
        for rating_data, name, color in [
            (gen_r,  "Genuine",    "#34d399"),
            (low_r,  "Low Suspicion","#fbbf24"),
            (fake_r, "Suspicious", "#ec4899"),
        ]:
            fig_bar.add_trace(go.Bar(
                x=rating_data.index, y=rating_data.values, name=name,
                marker=dict(color=color, opacity=0.85,
                            line=dict(color="rgba(255,255,255,0.2)", width=1)),
                hovertemplate=f"<b>{name}</b><br>Rating: %{{x}}★<br>Count: %{{y:,}}<extra></extra>",
            ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
            font_color="white", barmode="group",
            xaxis=dict(title="Rating (1–5)", gridcolor="rgba(255,255,255,0.05)",
                       tickvals=[1,2,3,4,5]),
            yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(bgcolor="rgba(255,255,255,0.05)", bordercolor="#7c3aed", font_color="white"),
            hoverlabel=dict(bgcolor="#1a1a2e", font_color="white", bordercolor="#7c3aed"),
            margin=dict(t=10, b=40), height=360,
            dragmode="zoom",
        )
        fig_bar.update_xaxes(fixedrange=False)
        st.plotly_chart(fig_bar, use_container_width=True, key="rating_bar")

    # Timeline — fully zoomable + hover shows hotel burst details
    st.markdown("### ⏱ Review Volume Timeline — Zoomable")
    st.caption("Scroll to zoom · Click and drag to pan · Click legend to toggle · Hover burst stars for hotel details")
    if df["date"].notna().sum() > 0:
        daily = df.groupby(df["date"].dt.date).size().reset_index(name="count")
        daily.columns = ["date", "count"]
        threshold = daily["count"].quantile(0.95)
        daily["is_burst"] = daily["count"] >= threshold

        # attach top hotel per day
        top_hotel_day = (
            df.dropna(subset=["date"])
            .assign(date_only=df["date"].dt.date)
            .groupby("date_only")
            .apply(lambda x: x["hotel"].value_counts().idxmax())
            .reset_index()
        )
        top_hotel_day.columns = ["date", "top_hotel"]
        daily = daily.merge(top_hotel_day, on="date", how="left")

        normal = daily[~daily["is_burst"]]
        bursts = daily[daily["is_burst"]]

        fig_tl = go.Figure()
        fig_tl.add_trace(go.Scatter(
            x=normal["date"], y=normal["count"], mode="lines+markers",
            name="Normal Day",
            marker=dict(color="#818cf8", size=5),
            line=dict(color="#6366f1", width=2),
            fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
            hovertemplate="<b>%{x}</b><br>Reviews: %{y}<extra></extra>",
        ))
        fig_tl.add_trace(go.Scatter(
            x=bursts["date"], y=bursts["count"], mode="markers+text",
            name="Burst Day (5+)",
            text=bursts["top_hotel"].str[:18],
            textposition="top center",
            textfont=dict(color="#fbbf24", size=9),
            marker=dict(color="#ec4899", size=16, symbol="star",
                        line=dict(color="#fff", width=1.5)),
            hovertemplate=(
                "<b>BURST DAY: %{x}</b><br>"
                "Reviews: %{y}<br>"
                "Top Hotel: %{text}<extra></extra>"
            ),
        ))
        # add threshold line
        fig_tl.add_hline(
            y=threshold, line_dash="dot", line_color="#f59e0b", line_width=1.5,
            annotation_text=f"Burst threshold ({threshold:.0f})",
            annotation_font_color="#f59e0b", annotation_position="top right"
        )
        fig_tl.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
            font_color="white",
            xaxis=dict(title="Date", gridcolor="rgba(255,255,255,0.05)",
                       rangeslider=dict(visible=True,
                                        bgcolor="rgba(124,58,237,0.1)",
                                        bordercolor="#7c3aed"),
                       rangeselector=dict(
                           buttons=[
                               dict(count=3, label="3m", step="month", stepmode="backward"),
                               dict(count=6, label="6m", step="month", stepmode="backward"),
                               dict(count=1, label="1y", step="year",  stepmode="backward"),
                               dict(step="all", label="All"),
                           ],
                           bgcolor="rgba(124,58,237,0.2)",
                           activecolor="#7c3aed",
                           font_color="white",
                       )),
            yaxis=dict(title="Reviews per Day", gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(bgcolor="rgba(255,255,255,0.05)", bordercolor="#7c3aed"),
            hoverlabel=dict(bgcolor="#1a1a2e", font_color="white", bordercolor="#ec4899"),
            height=400, margin=dict(t=30, b=60),
            dragmode="pan",
        )
        st.plotly_chart(fig_tl, use_container_width=True, key="timeline")
    else:
        st.info("No valid dates found for timeline.")

# ════════════════════════════════════════════════════════════
# TAB 2 — SUSPICIOUS REVIEWS
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🚨 Most Suspicious Reviews")

    top_n = st.slider("Show top N reviews", 5, 50, 15)
    top_reviews = filtered.nlargest(top_n, "suspicion_score").reset_index(drop=True)

    for _, row in top_reviews.iterrows():
        sc  = int(row["suspicion_score"])
        col = score_color(sc)
        lbl = row["label"]
        sigs = row["triggered_signals"] if isinstance(row["triggered_signals"], list) else []
        sig_html = " ".join(
            f'<span style="font-size:0.7rem;padding:2px 8px;border:1px solid {col};'
            f'border-radius:12px;color:{col};margin-right:4px">{SIGNAL_NAMES.get(s,s)}</span>'
            for s in sigs
        )
        hotel_safe = str(row["hotel"])[:60]
        text_safe  = str(row["text"])[:300]
        rating = int(row.get("rating5", 0))
        stars  = "★" * rating + "☆" * (5 - rating)

        st.markdown(f"""
        <div class="review-card" style="border-color:{col}">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <span style="font-weight:700;color:#fff;font-size:0.95rem">{hotel_safe}</span>
            <span style="color:#ffcc44">{stars}</span>
            <span style="background:{col};color:#fff;padding:3px 12px;border-radius:20px;
                         font-weight:900;font-size:0.85rem">{sc}/100</span>
          </div>
          <div style="font-size:0.82rem;color:#aaa;font-style:italic;margin:8px 0">
            "{text_safe}{'…' if len(str(row['text']))>300 else ''}"
          </div>
          <div style="background:#222;border-radius:4px;height:6px;margin:8px 0">
            <div style="width:{sc}%;height:6px;border-radius:4px;background:{col}"></div>
          </div>
          <div style="margin-top:8px">{sig_html}</div>
          <div style="font-size:0.75rem;color:#666;margin-top:6px">@{str(row['username'])[:30]}</div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — HOTELS
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🏨 Hotel Analysis")

    hotel_stats = df.groupby("hotel").agg(
        total=("suspicion_score", "count"),
        flagged=("suspicion_score", lambda x: (x > 40).sum()),
        avg_score=("suspicion_score", "mean"),
        avg_rating=("rating5", "mean")
    ).reset_index()
    hotel_stats["pct_flagged"] = 100 * hotel_stats["flagged"] / hotel_stats["total"]
    hotel_stats = hotel_stats[hotel_stats["total"] >= 3].sort_values("pct_flagged", ascending=False)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Top 15 Most Suspicious Hotels")
        st.caption("Hover for details · Click a bar to filter reviews below · Scroll to zoom")
        top15h = hotel_stats.head(15)
        fig_h = go.Figure(go.Bar(
            x=top15h["pct_flagged"],
            y=top15h["hotel"].str[:32],
            orientation="h",
            marker=dict(
                color=top15h["pct_flagged"],
                colorscale=[[0,"#6366f1"],[0.4,"#f59e0b"],[0.7,"#f87171"],[1,"#ec4899"]],
                showscale=True,
                colorbar=dict(
                    tickfont=dict(color="white"),
                    title=dict(text="% Flagged", font=dict(color="white")),
                    len=0.8,
                )
            ),
            text=top15h["pct_flagged"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            textfont=dict(color="white", size=11),
            customdata=np.stack([top15h["total"], top15h["flagged"], top15h["avg_score"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Flagged: %{x:.1f}%<br>"
                "Total reviews: %{customdata[0]:,}<br>"
                "Flagged count: %{customdata[1]:,}<br>"
                "Avg suspicion score: %{customdata[2]:.1f}"
                "<extra></extra>"
            ),
        ))
        fig_h.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
            font_color="white",
            xaxis=dict(title="% Flagged Reviews", gridcolor="rgba(255,255,255,0.05)",
                       fixedrange=False),
            yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.05)"),
            hoverlabel=dict(bgcolor="#1a1a2e", font_color="white", bordercolor="#a855f7"),
            height=480, margin=dict(t=10, b=40, l=10, r=80),
            dragmode="zoom",
        )
        event_h = st.plotly_chart(fig_h, use_container_width=True, key="hotel_bar",
                                   on_select="rerun", selection_mode="points")

    with col_b:
        st.markdown("### Suspicion Score vs Review Volume")
        st.caption("Bubble size = flagged count · Hover for full details · Zoom & pan freely")
        fig_scatter = px.scatter(
            hotel_stats.head(60),
            x="total", y="avg_score",
            size="flagged", color="pct_flagged",
            hover_name="hotel",
            color_continuous_scale=["#34d399","#fbbf24","#f87171","#ec4899"],
            labels={"total":"Total Reviews","avg_score":"Avg Suspicion Score",
                    "pct_flagged":"% Flagged","flagged":"Flagged Count"},
            custom_data=["flagged","pct_flagged","avg_rating"],
        )
        fig_scatter.update_traces(
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "Total reviews: %{x:,}<br>"
                "Avg score: %{y:.1f}/100<br>"
                "Flagged: %{customdata[0]:,} (%{customdata[1]:.1f}%)<br>"
                "Avg rating: %{customdata[2]:.1f}★"
                "<extra></extra>"
            ),
            marker=dict(opacity=0.85, line=dict(color="white", width=0.5)),
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
            font_color="white",
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", fixedrange=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", fixedrange=False),
            hoverlabel=dict(bgcolor="#1a1a2e", font_color="white", bordercolor="#ec4899"),
            height=480, margin=dict(t=10, b=40),
            dragmode="zoom",
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key="hotel_scatter")

    # Click-to-filter: if user clicked a bar, show that hotel's reviews
    selected_hotel = None
    try:
        pts = event_h.selection.get("points", [])
        if pts:
            selected_hotel = pts[0].get("label") or pts[0].get("y")
    except Exception:
        pass

    if selected_hotel:
        st.markdown(f"### Reviews for: **{selected_hotel}**")
        hotel_reviews = df[df["hotel"].str[:32] == selected_hotel].sort_values(
            "suspicion_score", ascending=False).head(10)
        for _, row in hotel_reviews.iterrows():
            sc  = int(row["suspicion_score"])
            col_c = score_color(sc)
            sigs = row["triggered_signals"] if isinstance(row["triggered_signals"], list) else []
            sig_html = " ".join(
                f'<span style="font-size:0.7rem;padding:2px 7px;border:1px solid {col_c};'
                f'border-radius:10px;color:{col_c}">{SIGNAL_NAMES.get(s,s)}</span>' for s in sigs)
            st.markdown(f"""
            <div class="review-card" style="border-color:{col_c}">
              <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                <span style="color:#ffcc44">{"★"*int(row.get("rating5",0))}{"☆"*(5-int(row.get("rating5",0)))}</span>
                <span style="background:{col_c};color:#fff;padding:2px 10px;border-radius:14px;font-weight:900;font-size:0.8rem">{sc}/100</span>
              </div>
              <div style="font-size:0.82rem;color:#aaa;font-style:italic">"{str(row['text'])[:250]}…"</div>
              <div style="margin-top:8px">{sig_html}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("### Full Hotel Table")
        st.caption("Click a bar above to filter · Sort columns by clicking headers")
        display_cols = ["hotel","total","flagged","pct_flagged","avg_score","avg_rating"]
        st.dataframe(
            hotel_stats[display_cols].rename(columns={
                "hotel":"Hotel","total":"Total","flagged":"Flagged",
                "pct_flagged":"% Flagged","avg_score":"Avg Score","avg_rating":"Avg Rating"
            }).style.format({"% Flagged":"{:.1f}%","Avg Score":"{:.1f}","Avg Rating":"{:.1f}"}),
            use_container_width=True, height=420
        )

# ════════════════════════════════════════════════════════════
# TAB 4 — SIGNALS
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 📡 Fraud Signal Analysis")

    signal_display = ["Length","Vague","Extreme","Duplicate","Behavior","Rating","Timing","Exclamation"]
    sig_counts     = [int(df[f"signal_S{i}"].sum()) for i in range(1,9)]
    sig_pcts       = [100*c/total for c in sig_counts]

    col_a, col_b = st.columns(2)

    sig_descriptions = [
        "Short review (<15 words) with extreme rating (1 or 5★)",
        "Too many vague words vs specific hotel details",
        "TextBlob polarity > 0.8 or < -0.8 (overly extreme)",
        "80%+ text overlap with another review (near-duplicate)",
        "Multi-hotel same-day poster or always-extreme rater",
        "Hotel has 90%+ five-star reviews (inflated rating pool)",
        "5+ reviews for one hotel on a single day (coordinated burst)",
        "3+ exclamation marks — artificially enthusiastic writing",
    ]
    sig_points = [20, 15, 15, 20, 15, 10, 15, 10]

    # Radar chart
    with col_a:
        st.markdown("### Signal Radar")
        st.caption("Hover each axis point for signal details")
        fig_radar = go.Figure(go.Scatterpolar(
            r=sig_pcts,
            theta=signal_display,
            fill="toself",
            fillcolor="rgba(168,85,247,0.25)",
            line=dict(color="#a855f7", width=3),
            marker=dict(color="#ec4899", size=10,
                        line=dict(color="white", width=1)),
            name="% of Reviews",
            customdata=list(zip(sig_counts, sig_points, sig_descriptions)),
            hovertemplate=(
                "<b>%{theta}</b><br>"
                "Reviews triggered: %{customdata[0]:,}<br>"
                "% of dataset: %{r:.1f}%<br>"
                "Score points: +%{customdata[1]}<br>"
                "<i>%{customdata[2]}</i>"
                "<extra></extra>"
            ),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(255,255,255,0.03)",
                radialaxis=dict(visible=True, color="#7c3aed",
                                tickfont=dict(color="#c4b5fd", size=9),
                                gridcolor="rgba(168,85,247,0.2)"),
                angularaxis=dict(tickfont=dict(color="#e2e8f0", size=12),
                                 gridcolor="rgba(168,85,247,0.2)",
                                 linecolor="rgba(168,85,247,0.3)")
            ),
            paper_bgcolor="rgba(0,0,0,0)", font_color="white",
            hoverlabel=dict(bgcolor="#1a1a2e", font_color="white", bordercolor="#a855f7"),
            showlegend=False, height=420, margin=dict(t=30, b=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True, key="sig_radar")

    # Bar chart of signals — click to see affected reviews
    with col_b:
        st.markdown("### Signal Counts")
        st.caption("Hover for description · Click a bar to see affected reviews below")
        sig_palette = ["#6366f1","#8b5cf6","#a855f7","#ec4899","#f43f5e","#f97316","#fbbf24","#34d399"]
        fig_sig = go.Figure(go.Bar(
            x=signal_display, y=sig_counts,
            marker=dict(color=sig_palette,
                        line=dict(color="rgba(255,255,255,0.15)", width=1)),
            text=[f"{c:,}" for c in sig_counts],
            textposition="outside",
            textfont=dict(color="white", size=11),
            customdata=list(zip(sig_pcts, sig_points, sig_descriptions)),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Triggered: %{y:,} reviews<br>"
                "% of dataset: %{customdata[0]:.1f}%<br>"
                "Score points: +%{customdata[1]}<br>"
                "<i>%{customdata[2]}</i>"
                "<extra></extra>"
            ),
        ))
        fig_sig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
            font_color="white",
            xaxis=dict(title="Signal", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Reviews Triggered", gridcolor="rgba(255,255,255,0.05)",
                       fixedrange=False),
            hoverlabel=dict(bgcolor="#1a1a2e", font_color="white", bordercolor="#a855f7"),
            height=420, margin=dict(t=30, b=40),
            dragmode="zoom",
        )
        sig_event = st.plotly_chart(fig_sig, use_container_width=True, key="sig_bar",
                                     on_select="rerun", selection_mode="points")

    # Click-to-filter: show reviews for clicked signal
    clicked_sig_idx = None
    try:
        sig_pts = sig_event.selection.get("points", [])
        if sig_pts:
            clicked_sig_idx = sig_pts[0].get("pointIndex", None)
    except Exception:
        pass

    if clicked_sig_idx is not None:
        sig_key = f"S{clicked_sig_idx + 1}"
        sig_name = signal_display[clicked_sig_idx]
        sig_reviews = df[df[f"signal_{sig_key}"] == True].nlargest(10, "suspicion_score")
        st.markdown(f"### Reviews that triggered **{sig_name}** signal ({len(df[df[f'signal_{sig_key}']]):,} total)")
        for _, row in sig_reviews.iterrows():
            sc = int(row["suspicion_score"])
            col_c = score_color(sc)
            st.markdown(f"""
            <div class="review-card" style="border-color:{col_c}">
              <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                <span style="color:#fff;font-weight:700;font-size:0.9rem">{str(row['hotel'])[:50]}</span>
                <span style="background:{col_c};color:#fff;padding:2px 10px;border-radius:14px;font-weight:900;font-size:0.8rem">{sc}/100</span>
              </div>
              <div style="font-size:0.82rem;color:#aaa;font-style:italic">"{str(row['text'])[:220]}…"</div>
            </div>""", unsafe_allow_html=True)
    else:
        # Signal breakdown table
        st.markdown("### Signal Breakdown Table")
        sig_df = pd.DataFrame({
            "Signal": [f"S{i}" for i in range(1,9)],
            "Name": signal_display,
            "Description": sig_descriptions,
            "Reviews Triggered": sig_counts,
            "% of Dataset": [f"{p:.1f}%" for p in sig_pcts],
            "+Points": sig_points,
        })
        st.dataframe(sig_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# TAB 5 — HEATMAP
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 🌡️ Fraud Signal Heatmap")
    st.caption("Hover each cell for exact count · Hotels sorted by total flags · Zoom freely")

    top_n_hotels = st.slider("Number of hotels to show", 10, 40, 20, key="hm_slider")
    signal_display_hm = ["Length","Vague","Extreme","Duplicate","Behavior","Rating","Timing","Exclamation"]

    top_hotels_hm = (
        df.groupby("hotel")[[f"signal_S{i}" for i in range(1,9)]]
        .sum().sum(axis=1).nlargest(top_n_hotels).index
    )
    hm_rows, hm_hotels = [], []
    for hotel in top_hotels_hm:
        sub = df[df["hotel"] == hotel]
        hm_rows.append([int(sub[f"signal_S{i}"].sum()) for i in range(1,9)])
        hm_hotels.append(hotel[:38])

    hm_z    = np.array(hm_rows)
    hm_text = [[str(v) for v in row] for row in hm_rows]

    fig_hm = go.Figure(go.Heatmap(
        z=hm_z,
        x=signal_display_hm,
        y=hm_hotels,
        text=hm_text,
        texttemplate="%{text}",
        textfont=dict(color="white", size=10),
        colorscale=[
            [0.0,  "rgba(30,20,60,0.8)"],
            [0.15, "#312e81"],
            [0.35, "#6d28d9"],
            [0.60, "#db2777"],
            [0.80, "#f97316"],
            [1.0,  "#fbbf24"],
        ],
        hoverongaps=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Signal: <b>%{x}</b><br>"
            "Reviews triggered: <b>%{z}</b>"
            "<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="Reviews", font=dict(color="white")),
            tickfont=dict(color="white"),
            len=0.9,
        ),
    ))
    fig_hm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis=dict(side="top", tickfont=dict(color="white", size=12),
                   gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(tickfont=dict(color="white", size=10), autorange="reversed",
                   gridcolor="rgba(255,255,255,0.05)"),
        hoverlabel=dict(bgcolor="#1a1a2e", font_color="white", bordercolor="#a855f7"),
        height=max(400, top_n_hotels * 28),
        margin=dict(t=60, b=20, l=20, r=80),
        dragmode="zoom",
    )
    st.plotly_chart(fig_hm, use_container_width=True, key="heatmap")

    # Mini bar below heatmap: total triggers per signal
    st.markdown("### Total Triggers per Signal (all hotels)")
    col_totals = hm_z.sum(axis=0)
    sig_palette_hm = ["#6366f1","#8b5cf6","#a855f7","#ec4899","#f43f5e","#f97316","#fbbf24","#34d399"]
    fig_tot = go.Figure(go.Bar(
        x=signal_display_hm, y=col_totals.tolist(),
        marker_color=sig_palette_hm,
        text=[f"{v:,}" for v in col_totals],
        textposition="outside", textfont=dict(color="white"),
        hovertemplate="<b>%{x}</b><br>Total: %{y:,}<extra></extra>",
    ))
    fig_tot.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
        font_color="white",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        hoverlabel=dict(bgcolor="#1a1a2e", font_color="white", bordercolor="#a855f7"),
        height=280, margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig_tot, use_container_width=True, key="hm_totals")

# ════════════════════════════════════════════════════════════
# TAB 6 — DUPLICATES
# ════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## 📋 Duplicate Review Clusters")
    st.markdown(f"Found **{len(dup_clusters):,}** clusters of near-identical reviews (≥80% text similarity)")

    if dup_clusters:
        show_n = st.slider("Show top N clusters", 3, 20, 8)
        sorted_clusters = sorted(dup_clusters, key=len, reverse=True)[:show_n]

        for i, members in enumerate(sorted_clusters, 1):
            rep_text  = df.iloc[members[0]]["text"][:350]
            hotels    = list(df.iloc[members]["hotel"].unique())[:4]
            usernames = list(df.iloc[members]["username"].unique())[:4]
            col = "#ff8800" if len(members) > 5 else "#ff4444"
            st.markdown(f"""
            <div class="review-card" style="border-color:{col}">
              <div style="color:{col};font-weight:700;margin-bottom:8px">
                🔁 Cluster #{i} — Repeated ×{len(members)}
              </div>
              <div style="color:#888;font-size:0.78rem;margin-bottom:8px">
                Hotels: {', '.join(hotels)} | Users: {', '.join(usernames)}
              </div>
              <div style="font-size:0.82rem;color:#aaa;font-style:italic">
                "{rep_text}{'…' if len(str(df.iloc[members[0]]['text']))>350 else ''}"
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No duplicate clusters detected.")

# ════════════════════════════════════════════════════════════
# TAB 7 — SEARCH
# ════════════════════════════════════════════════════════════
with tab7:
    st.markdown("## 🔎 Search & Analyse a Single Review")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        search_query = st.text_input("Search by hotel name or review text", placeholder="e.g. Grand Hotel")
    with col_b:
        search_score = st.slider("Min score", 0, 100, 0)

    if search_query:
        mask = (
            df["hotel"].str.contains(search_query, case=False, na=False) |
            df["text"].str.contains(search_query, case=False, na=False)
        ) & (df["suspicion_score"] >= search_score)
        results = df[mask].sort_values("suspicion_score", ascending=False).head(20)
        st.markdown(f"**{len(results)} results found**")
        for _, row in results.iterrows():
            sc  = int(row["suspicion_score"])
            col_badge = score_color(sc)
            sigs = row["triggered_signals"] if isinstance(row["triggered_signals"], list) else []
            st.markdown(f"""
            <div class="review-card" style="border-color:{col_badge}">
              <b style="color:#fff">{str(row['hotel'])[:50]}</b>
              <span style="float:right;background:{col_badge};color:#fff;padding:2px 10px;
                           border-radius:20px;font-size:0.8rem">{sc}/100</span><br>
              <span style="color:#aaa;font-size:0.8rem;font-style:italic">"{str(row['text'])[:200]}…"</span><br>
              <span style="color:#666;font-size:0.75rem">Signals: {', '.join(SIGNAL_NAMES.get(s,s) for s in sigs)}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧪 Analyse a Single Review")
    custom_text   = st.text_area("Paste a review here", height=120, placeholder="Type or paste a hotel review…")
    custom_rating = st.select_slider("Rating", options=[1,2,3,4,5], value=5)

    if st.button("🔍 Analyse Review") and custom_text.strip():
        score = 0
        fired = []
        words = custom_text.split()

        if len(words) < 15 and custom_rating in [1, 5]:
            score += 20; fired.append("S1: Short + Extreme Rating")
        wds = re.findall(r'\b\w+\b', custom_text.lower())
        v = sum(1 for w in wds if w in VAGUE_WORDS)
        s = sum(1 for w in wds if w in SPECIFIC_WORDS)
        if v / max(s, 1) > 3:
            score += 15; fired.append(f"S2: Vague Language (ratio {v/max(s,1):.1f})")
        try:
            pol = TextBlob(custom_text).sentiment.polarity
            if abs(pol) > 0.8:
                score += 15; fired.append(f"S3: Extreme Sentiment (polarity {pol:.2f})")
        except: pass
        if custom_text.count("!") >= 3:
            score += 10; fired.append(f"S8: Exclamation Overuse ({custom_text.count('!')} marks)")

        score = min(score, 100)
        col   = score_color(score)
        lbl   = ("Likely Fake" if score>80 else "High Suspicion" if score>60
                 else "Moderate" if score>40 else "Low Suspicion" if score>20 else "Genuine")

        st.markdown(f"""
        <div class="review-card" style="border-color:{col};margin-top:16px">
          <div style="font-size:1.4rem;font-weight:900;color:{col}">{score}/100 — {lbl}</div>
          <div style="background:#222;border-radius:6px;height:10px;margin:10px 0">
            <div style="width:{score}%;height:10px;border-radius:6px;background:{col}"></div>
          </div>
          {'<br>'.join(f'<span style="color:{col};font-size:0.85rem">⚠ {f}</span>' for f in fired)
           if fired else '<span style="color:#22cc66">✅ No fraud signals detected</span>'}
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="
  margin-top:40px;
  background:linear-gradient(135deg,rgba(124,58,237,0.15),rgba(37,99,235,0.1),rgba(236,72,153,0.08));
  border:1px solid rgba(168,85,247,0.3);
  border-radius:14px;padding:20px;text-align:center;
">
  <span style="
    background:linear-gradient(90deg,#a855f7,#38bdf8,#34d399);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    font-weight:700;font-size:0.95rem;
  ">🔍 Fake Review Detector</span>
  <span style="color:#555;font-size:0.85rem;"> · Built with Python & Streamlit · {total:,} reviews analyzed</span>
</div>
""", unsafe_allow_html=True)
