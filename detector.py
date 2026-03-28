"""
Fake Review Detector — Full Pipeline
Analyzes hotel_reviews.csv for 8 fraud signals and generates all output files.
"""

import os, re, warnings, base64, json
from io import BytesIO
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data", "hotel_reviews.csv")
OUT    = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

print("=" * 60)
print("FAKE REVIEW DETECTOR — Loading data…")
df = pd.read_csv(DATA)
print(f"Loaded {len(df):,} reviews")

# ─────────────────────────────────────────────
# CLEAN / PREP
# ─────────────────────────────────────────────
df["text"]     = df["reviews.text"].fillna("").astype(str)
df["rating"]   = pd.to_numeric(df["reviews.rating"], errors="coerce").fillna(0)
df["username"] = df["reviews.username"].fillna("anonymous").astype(str)
df["hotel"]    = df["name"].fillna("Unknown Hotel").astype(str)
df["date_raw"] = df["reviews.date"].fillna("")

# Normalize ratings to 1-5 scale (raw is 0-10)
df["rating5"] = (df["rating"] / 2).clip(1, 5).round()

# Parse dates
def parse_date(s):
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            pass
    try:
        return pd.to_datetime(s, infer_datetime_format=True)
    except Exception:
        return pd.NaT

df["date"] = df["date_raw"].apply(parse_date)
df["date_only"] = df["date"].dt.date

# ─────────────────────────────────────────────
# WORD LISTS
# ─────────────────────────────────────────────
VAGUE_WORDS    = {"great","good","bad","nice","terrible","amazing","worst","best",
                  "awesome","horrible","loved","hated","perfect","recommend"}
SPECIFIC_WORDS = {"room","floor","lobby","parking","breakfast","pool","shower","bed",
                  "wifi","staff","neighborhood","elevator","checkout","restaurant",
                  "view","bathroom","towel","pillow","concierge","reception","suite"}

# ─────────────────────────────────────────────
# SIGNAL SCORING
# ─────────────────────────────────────────────
scores  = np.zeros(len(df), dtype=float)
signals = {f"S{i}": np.zeros(len(df), dtype=bool) for i in range(1, 9)}

# --- Signal 1: Review Length -----------------------------------------------
print("Signal 1: Review Length…")
word_counts = df["text"].apply(lambda t: len(t.split()))
df["word_count"] = word_counts
short = word_counts < 15
s1 = short & ((df["rating5"] == 5) | (df["rating5"] == 1))
signals["S1"] = s1
scores[s1] += 20

# --- Signal 2: Vague vs Specific -------------------------------------------
print("Signal 2: Vague vs Specific Language…")
def vague_specific(text):
    words = re.findall(r'\b\w+\b', text.lower())
    v = sum(1 for w in words if w in VAGUE_WORDS)
    s = sum(1 for w in words if w in SPECIFIC_WORDS)
    ratio = v / max(s, 1)
    return ratio

ratios = df["text"].apply(vague_specific)
df["vague_ratio"] = ratios
s2 = ratios > 3
signals["S2"] = s2
scores[s2] += 15

# --- Signal 3: Extreme Sentiment -------------------------------------------
print("Signal 3: Extreme Sentiment (TextBlob)…")
def get_polarity(text):
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0

polarities = df["text"].apply(get_polarity)
df["polarity"] = polarities
s3 = (polarities > 0.8) | (polarities < -0.8)
signals["S3"] = s3
scores[s3] += 15

# --- Signal 4: Duplicate / Similar Text ------------------------------------
print("Signal 4: Duplicate / Similar Text…")
texts = df["text"].tolist()

# Vectorize and find cosine similarity in chunks to handle 35k rows
CHUNK = 500
dup_flags = np.zeros(len(df), dtype=bool)
dup_cluster = np.full(len(df), -1, dtype=int)

vectorizer = TfidfVectorizer(min_df=2, max_features=5000, ngram_range=(1, 2))
try:
    tfidf = vectorizer.fit_transform(texts)
except Exception:
    tfidf = None

duplicate_clusters = []   # list of lists of indices
cluster_id = 0
visited = set()

if tfidf is not None:
    # Block-based similarity scan
    n = len(df)
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        chunk = tfidf[start:end]
        sims = cosine_similarity(chunk, tfidf)
        for i, row in enumerate(sims):
            global_i = start + i
            if global_i in visited:
                continue
            matches = np.where(row >= 0.80)[0].tolist()
            matches = [m for m in matches if m != global_i]
            if matches:
                cluster_members = [global_i] + matches
                new_members = [m for m in cluster_members if m not in visited]
                if len(new_members) > 0:
                    duplicate_clusters.append(cluster_members)
                    for m in cluster_members:
                        dup_flags[m] = True
                        dup_cluster[m] = cluster_id
                        visited.add(m)
                    cluster_id += 1

signals["S4"] = dup_flags
scores[dup_flags] += 20
df["dup_cluster"] = dup_cluster
print(f"  Found {cluster_id} duplicate clusters")

# --- Signal 5: Reviewer Behavior -------------------------------------------
print("Signal 5: Reviewer Behavior…")
# 5a: Same reviewer, multiple hotels, same day
reviewer_day_hotel = df.groupby(["username", "date_only"])["hotel"].nunique()
reviewer_day_hotel = reviewer_day_hotel.reset_index()
reviewer_day_hotel.columns = ["username", "date_only", "hotels_per_day"]
multi_hotel_reviewers = set(
    reviewer_day_hotel[reviewer_day_hotel["hotels_per_day"] > 1]["username"]
)
s5a = df["username"].isin(multi_hotel_reviewers)

# 5b: Only posts 5-star or only 1-star
reviewer_ratings = df.groupby("username")["rating5"].agg(["min", "max", "count"])
extreme_only = reviewer_ratings[(
    ((reviewer_ratings["min"] == 5) & (reviewer_ratings["max"] == 5)) |
    ((reviewer_ratings["min"] == 1) & (reviewer_ratings["max"] == 1))
) & (reviewer_ratings["count"] >= 2)].index
s5b = df["username"].isin(extreme_only)

signals["S5"] = s5a | s5b
scores[s5a] += 15
scores[s5b] += 10

# --- Signal 6: Rating Distribution Anomaly ---------------------------------
print("Signal 6: Rating Distribution Anomaly…")
hotel_ratings = df.groupby("hotel")["rating5"]
hotel_five_pct = hotel_ratings.apply(lambda x: (x == 5).sum() / max(len(x), 1))
manipulated_hotels = set(hotel_five_pct[hotel_five_pct >= 0.90].index)
s6 = df["hotel"].isin(manipulated_hotels)
signals["S6"] = s6
scores[s6] += 10

# --- Signal 7: Timing Bursts -----------------------------------------------
print("Signal 7: Timing Bursts…")
hotel_day_counts = df.groupby(["hotel", "date_only"]).size().reset_index(name="count")
burst_pairs = set(
    hotel_day_counts[hotel_day_counts["count"] >= 5]
    .apply(lambda r: (r["hotel"], r["date_only"]), axis=1)
)
s7 = df.apply(lambda r: (r["hotel"], r["date_only"]) in burst_pairs, axis=1)
signals["S7"] = s7
scores[s7] += 15

# --- Signal 8: Exclamation Overuse -----------------------------------------
print("Signal 8: Exclamation Overuse…")
excl_counts = df["text"].apply(lambda t: t.count("!"))
df["excl_count"] = excl_counts
s8 = excl_counts >= 3
signals["S8"] = s8
scores[s8] += 10

# ─────────────────────────────────────────────
# FINAL SCORE (cap at 100)
# ─────────────────────────────────────────────
df["suspicion_score"] = np.clip(scores, 0, 100).astype(int)

for k, v in signals.items():
    df[f"signal_{k}"] = v

signal_cols = [f"signal_S{i}" for i in range(1, 9)]
df["triggered_signals"] = df[signal_cols].apply(
    lambda r: [f"S{i}" for i in range(1, 9) if r[f"signal_S{i}"]], axis=1
)

def score_label(s):
    if s <= 20:  return "Genuine"
    if s <= 40:  return "Low Suspicion"
    if s <= 60:  return "Moderate"
    if s <= 80:  return "High Suspicion"
    return "Likely Fake"

df["label"] = df["suspicion_score"].apply(score_label)

print("\n=== SCORE DISTRIBUTION ===")
print(df["label"].value_counts())

# ─────────────────────────────────────────────
# HELPER: img → base64
# ─────────────────────────────────────────────
def fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def save_fig(fig, filename):
    path = os.path.join(OUT, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {filename}")
    return path

# ═══════════════════════════════════════════════════════════════════
# OUTPUT 4 — signal_radar.png
# ═══════════════════════════════════════════════════════════════════
print("\nGenerating signal_radar.png…")
signal_names = ["Length", "Vague", "Extreme", "Duplicate", "Behavior", "Rating", "Timing", "Exclamation"]
signal_counts = [signals[f"S{i}"].sum() for i in range(1, 9)]

N = 8
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
values = signal_counts + [signal_counts[0]]

fig_radar, ax_radar = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True), facecolor="#0a0a0a")
ax_radar.set_facecolor("#111111")
ax_radar.plot(angles, values, color="#ff4444", linewidth=2)
ax_radar.fill(angles, values, color="#ff4444", alpha=0.3)
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(signal_names, color="white", size=11)
ax_radar.tick_params(colors="white")
ax_radar.yaxis.set_tick_params(labelcolor="#888")
ax_radar.set_title("Fraud Signal Radar", color="white", size=16, pad=20)
ax_radar.spines["polar"].set_color("#333")
ax_radar.grid(color="#333")
save_fig(fig_radar, "signal_radar.png")

# ═══════════════════════════════════════════════════════════════════
# OUTPUT 2 — suspicion_heatmap.png
# ═══════════════════════════════════════════════════════════════════
print("Generating suspicion_heatmap.png…")
top20_hotels = df.groupby("hotel").size().nlargest(20).index
hotel_signals = []
for hotel in top20_hotels:
    sub = df[df["hotel"] == hotel]
    row = {"Hotel": hotel[:35]}
    for i in range(1, 9):
        row[f"S{i}:{signal_names[i-1]}"] = signals[f"S{i}"][sub.index].sum()
    hotel_signals.append(row)

hm_df = pd.DataFrame(hotel_signals).set_index("Hotel")

fig_hm, ax_hm = plt.subplots(figsize=(14, 10), facecolor="#0a0a0a")
ax_hm.set_facecolor("#0a0a0a")
sns.heatmap(
    hm_df, ax=ax_hm, cmap="Reds", annot=True, fmt="d",
    linewidths=0.5, linecolor="#222",
    cbar_kws={"label": "# Reviews Triggered"}
)
ax_hm.set_title("Fraud Signal Heatmap — Top 20 Hotels", color="white", size=14, pad=12)
ax_hm.tick_params(colors="white")
ax_hm.set_xlabel("Fraud Signal", color="#aaa")
ax_hm.set_ylabel("Hotel", color="#aaa")
plt.xticks(rotation=30, ha="right", color="white")
plt.yticks(color="white")
save_fig(fig_hm, "suspicion_heatmap.png")

# ═══════════════════════════════════════════════════════════════════
# OUTPUT 3 — genuine_vs_fake_wordcloud.png
# ═══════════════════════════════════════════════════════════════════
print("Generating genuine_vs_fake_wordcloud.png…")
try:
    from wordcloud import WordCloud

    genuine_text = " ".join(df[df["suspicion_score"] < 20]["text"].tolist())
    fake_text    = " ".join(df[df["suspicion_score"] > 50]["text"].tolist())

    STOP = {"the","a","an","is","was","were","are","and","or","of","to","in","it",
            "for","on","with","this","that","at","be","we","our","i","my","they",
            "had","have","not","but","so","as","by","from","he","she","very",
            "hotel","room","stay","stayed","would","could","great"}

    wc_genuine = WordCloud(width=700, height=500, background_color="#0a1a0a",
                           colormap="Greens", stopwords=STOP,
                           max_words=150, collocations=False).generate(genuine_text or "genuine review")
    wc_fake    = WordCloud(width=700, height=500, background_color="#1a0a0a",
                           colormap="Reds", stopwords=STOP,
                           max_words=150, collocations=False).generate(fake_text or "fake review")

    fig_wc, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0a0a0a")
    ax_l.imshow(wc_genuine, interpolation="bilinear")
    ax_l.axis("off")
    ax_l.set_title("Genuine Reviews (score < 20)", color="#44ff88", size=14, pad=10)
    ax_r.imshow(wc_fake, interpolation="bilinear")
    ax_r.axis("off")
    ax_r.set_title("Suspicious Reviews (score > 50)", color="#ff4444", size=14, pad=10)
    fig_wc.patch.set_facecolor("#0a0a0a")
    # divider line
    fig_wc.add_artist(plt.Line2D([0.5, 0.5], [0, 1], transform=fig_wc.transFigure,
                                  color="#444", linewidth=2))
    save_fig(fig_wc, "genuine_vs_fake_wordcloud.png")
except Exception as e:
    print(f"  WordCloud skipped: {e}")

# ═══════════════════════════════════════════════════════════════════
# OUTPUT 5 — rating_comparison.png
# ═══════════════════════════════════════════════════════════════════
print("Generating rating_comparison.png…")
genuine_ratings  = df[df["suspicion_score"] < 20]["rating5"]
suspect_ratings  = df[df["suspicion_score"] > 50]["rating5"]

fig_rc, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0a0a0a")
for ax in (ax_l, ax_r):
    ax.set_facecolor("#111")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333")

ax_l.hist(genuine_ratings, bins=[0.5,1.5,2.5,3.5,4.5,5.5], color="#44cc77", edgecolor="#222", rwidth=0.8)
ax_l.set_title("Genuine Reviews\n(score < 20)", color="white", size=13)
ax_l.set_xlabel("Rating (1-5)", color="#aaa")
ax_l.set_ylabel("Count", color="#aaa")

ax_r.hist(suspect_ratings, bins=[0.5,1.5,2.5,3.5,4.5,5.5], color="#ff4444", edgecolor="#222", rwidth=0.8)
ax_r.set_title("Suspicious Reviews\n(score > 50)", color="white", size=13)
ax_r.set_xlabel("Rating (1-5)", color="#aaa")
ax_r.set_ylabel("Count", color="#aaa")

fig_rc.suptitle("Rating Distribution: Genuine vs Suspicious", color="white", size=15, y=1.02)
save_fig(fig_rc, "rating_comparison.png")

# ═══════════════════════════════════════════════════════════════════
# OUTPUT 6 — timeline_bursts.png
# ═══════════════════════════════════════════════════════════════════
print("Generating timeline_bursts.png…")
daily = df.groupby(["date", "hotel"]).size().reset_index(name="count")
daily = daily.dropna(subset=["date"])
daily["is_burst"] = daily["count"] >= 5

fig_tl, ax_tl = plt.subplots(figsize=(16, 7), facecolor="#0a0a0a")
ax_tl.set_facecolor("#111")

normal = daily[~daily["is_burst"]]
bursts = daily[daily["is_burst"]]

ax_tl.scatter(normal["date"], normal["count"], color="#555", alpha=0.4, s=15, label="Normal")
ax_tl.scatter(bursts["date"], bursts["count"], color="#ff3333", s=80, zorder=5, label="Burst (5+)")

# Label top burst hotels
top_bursts = bursts.nlargest(15, "count")
for _, row in top_bursts.iterrows():
    ax_tl.annotate(row["hotel"][:20], (row["date"], row["count"]),
                   textcoords="offset points", xytext=(5, 5),
                   color="#ffaa44", fontsize=7, alpha=0.85)

ax_tl.set_title("Review Burst Timeline — Hotels", color="white", size=14)
ax_tl.set_xlabel("Date", color="#aaa")
ax_tl.set_ylabel("Reviews per Day per Hotel", color="#aaa")
ax_tl.tick_params(colors="white")
for spine in ax_tl.spines.values():
    spine.set_color("#333")
ax_tl.legend(facecolor="#222", labelcolor="white")
plt.xticks(rotation=30)
save_fig(fig_tl, "timeline_bursts.png")

# ═══════════════════════════════════════════════════════════════════
# OUTPUT 7 — top_suspects.png
# ═══════════════════════════════════════════════════════════════════
print("Generating top_suspects.png…")
top5 = df.nlargest(5, "suspicion_score")[
    ["hotel", "text", "rating5", "suspicion_score", "triggered_signals", "username"]
].reset_index(drop=True)

fig_ts, axes = plt.subplots(5, 1, figsize=(14, 18), facecolor="#080808")
fig_ts.suptitle("⚠  MOST WANTED — TOP 5 SUSPECTED FAKE REVIEWS", color="#ff3333",
                 size=18, fontweight="bold", y=0.98)

score_colors = {
    "Likely Fake": "#cc0000",
    "High Suspicion": "#ff6600",
    "Moderate": "#ffaa00",
}

for idx, (_, row) in enumerate(top5.iterrows()):
    ax = axes[idx]
    ax.set_facecolor("#111")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    sc = int(row["suspicion_score"])
    lbl = score_label(sc)
    col = score_colors.get(lbl, "#ff4444")

    # Card background
    rect = FancyBboxPatch((0.01, 0.05), 0.98, 0.9, boxstyle="round,pad=0.01",
                           facecolor="#1a1a1a", edgecolor=col, linewidth=2)
    ax.add_patch(rect)

    # Score bar
    bar_w = min(sc / 100, 1.0) * 0.6
    ax.add_patch(FancyBboxPatch((0.32, 0.55), bar_w, 0.12,
                                 boxstyle="round,pad=0.005",
                                 facecolor=col, alpha=0.85))
    ax.text(0.32 + bar_w + 0.01, 0.61, f"{sc}/100", color=col, fontsize=10, fontweight="bold", va="center")

    # Hotel + rating
    ax.text(0.04, 0.78, row["hotel"][:50], color="white", fontsize=10, fontweight="bold")
    ax.text(0.04, 0.63, f"★ {row['rating5']:.0f}  |  @{str(row['username'])[:20]}",
            color="#aaa", fontsize=9)
    ax.text(0.04, 0.40, f"\"{str(row['text'])[:180]}…\"",
            color="#ccc", fontsize=8, wrap=True,
            verticalalignment="top", style="italic")

    # Signal badges
    sigs = row["triggered_signals"] if isinstance(row["triggered_signals"], list) else []
    sig_labels = {
        "S1": "Length", "S2": "Vague", "S3": "Extreme", "S4": "Duplicate",
        "S5": "Behavior", "S6": "Rating", "S7": "Timing", "S8": "Exclamation"
    }
    for si, sig in enumerate(sigs[:8]):
        ax.text(0.04 + si * 0.12, 0.14, sig_labels.get(sig, sig),
                color=col, fontsize=7, fontweight="bold",
                bbox=dict(facecolor="#2a2a2a", edgecolor=col, pad=2, boxstyle="round"))

    ax.text(0.88, 0.78, lbl, color=col, fontsize=9, fontweight="bold", ha="right")

plt.tight_layout(rect=[0, 0, 1, 0.97])
save_fig(fig_ts, "top_suspects.png")

# ═══════════════════════════════════════════════════════════════════
# COMPUTE DASHBOARD DATA
# ═══════════════════════════════════════════════════════════════════
total        = len(df)
n_genuine    = (df["suspicion_score"] <= 20).sum()
n_low        = ((df["suspicion_score"] > 20) & (df["suspicion_score"] <= 40)).sum()
n_moderate   = ((df["suspicion_score"] > 40) & (df["suspicion_score"] <= 60)).sum()
n_high       = ((df["suspicion_score"] > 60) & (df["suspicion_score"] <= 80)).sum()
n_fake       = (df["suspicion_score"] > 80).sum()
n_suspicious = total - n_genuine

pct_genuine    = 100 * n_genuine / total
pct_suspicious = 100 * n_suspicious / total
pct_fake       = 100 * n_fake / total

top15  = df.nlargest(15, "suspicion_score").reset_index(drop=True)

hotel_stats = df.groupby("hotel").agg(
    total=("suspicion_score", "count"),
    flagged=("suspicion_score", lambda x: (x > 40).sum()),
    avg_score=("suspicion_score", "mean")
).reset_index()
hotel_stats["pct_flagged"] = 100 * hotel_stats["flagged"] / hotel_stats["total"]
top10_hotels = hotel_stats[hotel_stats["total"] >= 5].nlargest(10, "pct_flagged")

# Genuine vs fake language examples
genuine_examples = (
    df[df["suspicion_score"] < 20]["text"]
    .dropna().str.strip()
    .loc[lambda s: s.str.len() > 80]
    .head(5).tolist()
)
fake_examples = (
    df[df["suspicion_score"] > 60]["text"]
    .dropna().str.strip()
    .loc[lambda s: s.str.len() > 30]
    .head(5).tolist()
)

# Duplicate clusters summary
cluster_summaries = []
if duplicate_clusters:
    for cl_indices in duplicate_clusters[:10]:
        rep_text = df.iloc[cl_indices[0]]["text"][:300]
        cluster_summaries.append({
            "count": len(cl_indices),
            "text": rep_text,
            "hotels": list(df.iloc[cl_indices]["hotel"].unique())[:3]
        })

# Burst timeline data
burst_data = hotel_day_counts.nlargest(20, "count").to_dict("records")

# Signal counts for radar
sig_counts = {signal_names[i]: int(signals[f"S{i+1}"].sum()) for i in range(8)}

# Top 1 signal
top_signal = max(sig_counts, key=sig_counts.get)

# ═══════════════════════════════════════════════════════════════════
# LOAD base64 images for embedding
# ═══════════════════════════════════════════════════════════════════
def load_b64(fname):
    path = os.path.join(OUT, fname)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

b64_radar   = load_b64("signal_radar.png")
b64_heatmap = load_b64("suspicion_heatmap.png")
b64_wordcloud = load_b64("genuine_vs_fake_wordcloud.png")
b64_rating  = load_b64("rating_comparison.png")
b64_timeline = load_b64("timeline_bursts.png")
b64_suspects = load_b64("top_suspects.png")

# ═══════════════════════════════════════════════════════════════════
# OUTPUT 1 — dashboard.html
# ═══════════════════════════════════════════════════════════════════
print("Generating dashboard.html…")

def score_color(s):
    if s <= 20:  return "#22cc66"
    if s <= 40:  return "#cccc22"
    if s <= 60:  return "#ff8800"
    if s <= 80:  return "#ff4400"
    return "#cc0000"

def card_html(row):
    sc = int(row["suspicion_score"])
    col = score_color(sc)
    sigs = row["triggered_signals"] if isinstance(row["triggered_signals"], list) else []
    sig_label_map = {
        "S1": "Length", "S2": "Vague", "S3": "Extreme", "S4": "Duplicate",
        "S5": "Behavior", "S6": "Rating", "S7": "Timing", "S8": "Exclamation"
    }
    sig_badges = "".join(
        f'<span class="badge" style="border-color:{col};color:{col}">{sig_label_map.get(s,s)}</span>'
        for s in sigs
    )
    hotel_safe = str(row["hotel"]).replace("<","&lt;").replace(">","&gt;")
    text_safe  = str(row["text"])[:280].replace("<","&lt;").replace(">","&gt;")
    rating = int(row.get("rating5", 0))
    stars  = "★" * rating + "☆" * (5 - rating)
    return f"""
    <div class="case-card">
      <div class="card-header">
        <span class="hotel-name">{hotel_safe}</span>
        <span class="rating-badge">{stars} {rating}/5</span>
        <span class="score-badge" style="background:{col}">{sc}</span>
      </div>
      <div class="review-text">"{text_safe}…"</div>
      <div class="meter-wrap">
        <div class="meter-bar" style="width:{sc}%;background:{col}"></div>
      </div>
      <div class="badge-row">{sig_badges}</div>
    </div>"""

case_cards_html = "\n".join(card_html(row) for _, row in top15.iterrows())

def hotel_card(row):
    sc = float(row["pct_flagged"])
    col = score_color(min(sc * 1.3, 100))
    if sc >= 80:   lbl, lblcol = "CRITICAL", "#cc0000"
    elif sc >= 60: lbl, lblcol = "HIGH", "#ff4400"
    elif sc >= 40: lbl, lblcol = "MODERATE", "#ff8800"
    else:          lbl, lblcol = "LOW", "#cccc22"
    name_safe = str(row["hotel"])[:45].replace("<","&lt;").replace(">","&gt;")
    return f"""
    <div class="hotel-card">
      <div class="hotel-name">{name_safe}</div>
      <div class="hotel-stats">
        <span>Total: <b>{int(row['total'])}</b></span>
        <span>Flagged: <b style="color:{col}">{int(row['flagged'])}</b></span>
        <span>% Flagged: <b style="color:{col}">{sc:.1f}%</b></span>
      </div>
      <div class="meter-wrap"><div class="meter-bar" style="width:{min(sc,100):.1f}%;background:{col}"></div></div>
      <span class="suspicion-badge" style="background:{lblcol}">{lbl}</span>
    </div>"""

hotel_cards_html = "\n".join(hotel_card(row) for _, row in top10_hotels.iterrows())

def quote_li(text):
    safe = str(text)[:200].replace("<","&lt;").replace(">","&gt;")
    return f'<li>"{safe}…"</li>'

genuine_list = "\n".join(quote_li(t) for t in genuine_examples)
fake_list    = "\n".join(quote_li(t) for t in fake_examples)

def dup_card(cl):
    text_safe = str(cl["text"]).replace("<","&lt;").replace(">","&gt;")
    hotels = ", ".join(cl["hotels"])
    return f"""
    <div class="dup-card">
      <div class="dup-count">Repeated ×{cl['count']} — Hotels: {hotels}</div>
      <div class="dup-text">"{text_safe}…"</div>
    </div>"""

dup_cards_html = "\n".join(dup_card(c) for c in cluster_summaries) if cluster_summaries else \
    "<p style='color:#888'>No high-confidence duplicate clusters detected.</p>"

burst_rows_html = "".join(
    f'<div class="burst-dot" style="left:{min(i*9,90)}%;background:{"#ff3333" if r["count"]>=5 else "#555"}" '
    f'title="{r["hotel"][:30]}: {r["count"]} reviews on {r["date_only"]}">'
    f'<span class="burst-label">{r["hotel"][:15]}</span></div>'
    for i, r in enumerate(burst_data[:20])
)

most_suspicious_hotel = hotel_stats.nlargest(1, "avg_score").iloc[0]["hotel"] if len(hotel_stats) else "N/A"

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Fake Review Detection Dashboard</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0a0a0a;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif;line-height:1.5}}
  a{{color:#44aaff}}

  /* HERO */
  .hero{{
    background:linear-gradient(135deg,#0d0d0d 0%,#1a0a0a 40%,#0a0a1a 100%);
    padding:60px 40px 50px;
    border-bottom:2px solid #222;
    position:relative;
    overflow:hidden;
  }}
  .hero::before{{
    content:'';position:absolute;top:-80px;right:-80px;
    width:400px;height:400px;
    background:radial-gradient(circle,rgba(220,30,30,.15) 0%,transparent 70%);
    border-radius:50%;
  }}
  .hero-title{{font-size:2.6rem;font-weight:900;letter-spacing:2px;
    background:linear-gradient(90deg,#ff4444,#ffaa00,#ff4444);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;}}
  .hero-sub{{color:#888;font-size:1rem;margin-top:6px;margin-bottom:40px}}
  .stats-row{{display:flex;gap:30px;flex-wrap:wrap;margin-top:10px}}
  .stat-box{{
    background:rgba(255,255,255,.04);border:1px solid #2a2a2a;
    border-radius:12px;padding:22px 30px;min-width:160px;flex:1;
    position:relative;overflow:hidden;
  }}
  .stat-number{{font-size:2.6rem;font-weight:900;line-height:1}}
  .stat-label{{font-size:.8rem;color:#888;margin-top:4px;text-transform:uppercase;letter-spacing:1px}}
  .genuine{{color:#22cc66}} .suspicious{{color:#ff8800}} .fake{{color:#cc2222}}

  /* SECTIONS */
  section{{padding:50px 40px;border-bottom:1px solid #1a1a1a}}
  h2{{font-size:1.6rem;font-weight:700;margin-bottom:24px;
      border-left:4px solid #ff4444;padding-left:14px;color:#fff}}
  h3{{font-size:1.1rem;color:#ccc;margin-bottom:12px}}

  /* CASE CARDS */
  .cards-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:20px}}
  .case-card{{
    background:#141414;border:1px solid #2a2a2a;border-radius:10px;
    padding:18px;transition:border-color .2s;
  }}
  .case-card:hover{{border-color:#ff4444}}
  .card-header{{display:flex;align-items:center;gap:10px;margin-bottom:10px}}
  .hotel-name{{font-weight:700;color:#fff;flex:1;font-size:.95rem}}
  .rating-badge{{color:#ffcc44;font-size:.85rem}}
  .score-badge{{
    font-size:.8rem;font-weight:900;padding:3px 10px;
    border-radius:20px;color:#fff;white-space:nowrap;
  }}
  .review-text{{font-size:.82rem;color:#aaa;margin:10px 0;font-style:italic;line-height:1.5}}
  .meter-wrap{{background:#222;border-radius:4px;height:6px;margin:8px 0}}
  .meter-bar{{height:6px;border-radius:4px;transition:width .6s}}
  .badge-row{{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}}
  .badge{{font-size:.7rem;padding:2px 8px;border:1px solid;border-radius:12px;background:rgba(0,0,0,.3)}}

  /* HOTEL CARDS */
  .hotel-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}}
  .hotel-card{{background:#141414;border:1px solid #2a2a2a;border-radius:10px;padding:16px}}
  .hotel-card .hotel-name{{font-weight:700;font-size:.95rem;margin-bottom:8px;color:#fff}}
  .hotel-stats{{display:flex;gap:16px;font-size:.82rem;color:#aaa;margin-bottom:8px;flex-wrap:wrap}}
  .suspicion-badge{{font-size:.7rem;padding:3px 10px;border-radius:12px;color:#fff;font-weight:700;margin-top:6px;display:inline-block}}

  /* LANGUAGE COLUMNS */
  .lang-row{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
  .lang-card{{background:#141414;border-radius:10px;padding:20px;border:1px solid}}
  .lang-card.genuine-card{{border-color:#225533}}
  .lang-card.fake-card{{border-color:#552222}}
  .lang-card h3{{margin-bottom:12px}}
  .lang-card ul{{list-style:none;padding:0}}
  .lang-card ul li{{font-size:.82rem;color:#aaa;padding:8px 0;border-bottom:1px solid #1e1e1e;font-style:italic;line-height:1.5}}

  /* DUPLICATES */
  .dup-card{{background:#141414;border:1px solid #2a2a2a;border-radius:8px;padding:16px;margin-bottom:12px}}
  .dup-count{{color:#ff8800;font-size:.8rem;font-weight:700;margin-bottom:8px}}
  .dup-text{{font-size:.82rem;color:#aaa;font-style:italic;line-height:1.5}}

  /* TIMELINE */
  .burst-timeline{{position:relative;height:80px;background:#111;border-radius:8px;overflow:hidden}}
  .burst-dot{{position:absolute;width:24px;height:24px;border-radius:50%;top:50%;transform:translateY(-50%);
              cursor:pointer;transition:transform .2s}}
  .burst-dot:hover{{transform:translateY(-50%) scale(1.4)}}
  .burst-label{{position:absolute;top:28px;left:-20px;white-space:nowrap;font-size:.6rem;color:#ffaa44;width:80px}}

  /* CHART IMAGES */
  .chart-img{{max-width:100%;border-radius:10px;border:1px solid #222}}

  /* VERDICT */
  .verdict-box{{
    background:linear-gradient(135deg,#1a0000,#0d0d0d);
    border:2px solid #cc2222;border-radius:16px;padding:36px;
  }}
  .verdict-big{{font-size:1.5rem;font-weight:900;color:#ff4444;margin-bottom:20px;line-height:1.4}}
  .verdict-bullets{{list-style:none;padding:0}}
  .verdict-bullets li{{padding:10px 0 10px 24px;border-bottom:1px solid #1e1e1e;color:#ccc;font-size:.9rem;position:relative}}
  .verdict-bullets li::before{{content:"▶";position:absolute;left:0;color:#ff4444}}

  /* RADAR chart */
  .center-img{{text-align:center}}

  /* Responsive */
  @media(max-width:700px){{
    .stats-row{{flex-direction:column}}
    .lang-row{{grid-template-columns:1fr}}
    .cards-grid{{grid-template-columns:1fr}}
  }}
</style>
</head>
<body>

<!-- ══════ HERO ══════ -->
<div class="hero">
  <div class="hero-title">🔍 FAKE REVIEW DETECTOR</div>
  <div class="hero-sub">Hotel Review Fraud Intelligence Dashboard · {total:,} Reviews Analyzed</div>
  <div class="stats-row">
    <div class="stat-box">
      <div class="stat-number">{total:,}</div>
      <div class="stat-label">Total Reviews</div>
    </div>
    <div class="stat-box">
      <div class="stat-number genuine">{pct_genuine:.1f}%</div>
      <div class="stat-label">Genuine (score ≤ 20)</div>
    </div>
    <div class="stat-box">
      <div class="stat-number suspicious">{pct_suspicious:.1f}%</div>
      <div class="stat-label">Suspicious (score &gt; 20)</div>
    </div>
    <div class="stat-box">
      <div class="stat-number fake">{pct_fake:.1f}%</div>
      <div class="stat-label">Likely Fake (score &gt; 80)</div>
    </div>
    <div class="stat-box">
      <div class="stat-number" style="color:#ffaa00">{n_high + n_fake:,}</div>
      <div class="stat-label">High Risk Reviews</div>
    </div>
  </div>
</div>

<!-- ══════ INVESTIGATION BOARD ══════ -->
<section>
  <h2>🚨 Investigation Board — Top 15 Most Suspicious Reviews</h2>
  <div class="cards-grid">
    {case_cards_html}
  </div>
</section>

<!-- ══════ SIGNAL RADAR ══════ -->
<section>
  <h2>📡 Signal Radar — Which Fraud Patterns Fire Most</h2>
  <div class="center-img">
    <img class="chart-img" src="data:image/png;base64,{b64_radar}" alt="Signal Radar" style="max-width:600px">
  </div>
</section>

<!-- ══════ SUSPICIOUS HOTELS ══════ -->
<section>
  <h2>🏨 Suspicious Hotels — Top 10 by % Flagged Reviews</h2>
  <div class="hotel-grid">
    {hotel_cards_html}
  </div>
</section>

<!-- ══════ GENUINE vs FAKE LANGUAGE ══════ -->
<section>
  <h2>💬 Genuine vs Suspicious Language</h2>
  <div class="lang-row">
    <div class="lang-card genuine-card">
      <h3 style="color:#22cc66">✅ What Genuine Reviews Sound Like</h3>
      <ul>{genuine_list}</ul>
    </div>
    <div class="lang-card fake-card">
      <h3 style="color:#ff4444">⚠ What Suspicious Reviews Sound Like</h3>
      <ul>{fake_list}</ul>
    </div>
  </div>
</section>

<!-- ══════ WORD CLOUDS ══════ -->
<section>
  <h2>☁ Word Clouds — Genuine vs Suspicious Language</h2>
  <img class="chart-img" src="data:image/png;base64,{b64_wordcloud}" alt="Word Clouds" style="width:100%">
</section>

<!-- ══════ DUPLICATE EVIDENCE ══════ -->
<section>
  <h2>📋 Duplicate Evidence — Near-Identical Review Clusters</h2>
  {dup_cards_html}
</section>

<!-- ══════ TIMING BURST TIMELINE ══════ -->
<section>
  <h2>⏱ Timing Burst Timeline</h2>
  <img class="chart-img" src="data:image/png;base64,{b64_timeline}" alt="Timeline Bursts" style="width:100%">
  <p style="color:#666;margin-top:12px;font-size:.82rem">Red dots = burst days (5+ reviews per hotel per day)</p>
</section>

<!-- ══════ HEATMAP ══════ -->
<section>
  <h2>🌡 Fraud Signal Heatmap — Top 20 Hotels</h2>
  <img class="chart-img" src="data:image/png;base64,{b64_heatmap}" alt="Heatmap" style="width:100%">
</section>

<!-- ══════ RATING COMPARISON ══════ -->
<section>
  <h2>⭐ Rating Distribution: Genuine vs Suspicious</h2>
  <img class="chart-img" src="data:image/png;base64,{b64_rating}" alt="Rating Comparison" style="width:100%">
</section>

<!-- ══════ TOP SUSPECTS IMAGE ══════ -->
<section>
  <h2>🎯 Top 5 Most Suspected Fake Reviews</h2>
  <img class="chart-img" src="data:image/png;base64,{b64_suspects}" alt="Top Suspects" style="width:100%">
</section>

<!-- ══════ VERDICT ══════ -->
<section>
  <h2>⚖ The Verdict</h2>
  <div class="verdict-box">
    <div class="verdict-big">
      {pct_suspicious:.0f}% of reviews show at least one fraud signal.<br>
      The #1 pattern is <span style="color:#ffaa00">{top_signal}</span> — detected in {sig_counts[top_signal]:,} reviews.<br>
      Most suspicious hotel: <span style="color:#ff4444">{most_suspicious_hotel}</span>
    </div>
    <ul class="verdict-bullets">
      <li>Implement automated review velocity limits — no more than 3 reviews per hotel per day from new accounts.</li>
      <li>Flag and manually review any review under 15 words with an extreme (1 or 5) rating before publication.</li>
      <li>Cross-reference reviewer history across properties — reviewers posting to 3+ hotels on the same day should require identity verification.</li>
    </ul>
  </div>
</section>

<footer style="padding:30px 40px;color:#444;font-size:.8rem;text-align:center">
  Fake Review Detector · Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · {total:,} reviews analyzed
</footer>

</body>
</html>"""

dashboard_path = os.path.join(OUT, "dashboard.html")
with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"  Saved: dashboard.html ({os.path.getsize(dashboard_path) // 1024} KB)")

# ═══════════════════════════════════════════════════════════════════
# OUTPUT 8 — fraud_report.md
# ═══════════════════════════════════════════════════════════════════
print("Generating fraud_report.md…")

def md_escape(t):
    return str(t).replace("|", "\\|")

top10_reviews = df.nlargest(10, "suspicion_score")[
    ["hotel", "text", "rating5", "suspicion_score", "triggered_signals", "username"]
].reset_index(drop=True)

top10_hotels_md = hotel_stats[hotel_stats["total"] >= 5].nlargest(10, "avg_score")

report_lines = [
    "# Fake Review Detection — Investigation Report",
    f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
    f"**Dataset:** hotel_reviews.csv  \n",

    "## Executive Summary\n",
    f"A full statistical and linguistic analysis was conducted on **{total:,} hotel reviews** "
    f"using 8 fraud detection signals. The analysis reveals that **{pct_suspicious:.1f}%** of reviews "
    f"exhibit at least one suspicious pattern, with **{pct_fake:.1f}%** classified as Likely Fake "
    f"(suspicion score > 80).\n",

    "## Score Distribution\n",
    f"| Label            | Count   | % of Total |",
    f"|------------------|---------|------------|",
    f"| Genuine (0–20)   | {n_genuine:>7,} | {100*n_genuine/total:>9.1f}% |",
    f"| Low (21–40)      | {n_low:>7,} | {100*n_low/total:>9.1f}% |",
    f"| Moderate (41–60) | {n_moderate:>7,} | {100*n_moderate/total:>9.1f}% |",
    f"| High (61–80)     | {n_high:>7,} | {100*n_high/total:>9.1f}% |",
    f"| Likely Fake (81+)| {n_fake:>7,} | {100*n_fake/total:>9.1f}% |",
    "",

    "## Top 10 Most Suspicious Reviews\n",
]

for i, (_, row) in enumerate(top10_reviews.iterrows(), 1):
    sigs = row["triggered_signals"] if isinstance(row["triggered_signals"], list) else []
    report_lines += [
        f"### #{i} — Score: {int(row['suspicion_score'])}/100 — {row['hotel']}",
        f"**Username:** @{row['username']}  **Rating:** {int(row['rating5'])}/5  "
        f"**Signals:** {', '.join(sigs)}",
        f"> {str(row['text'])[:400]}",
        "",
    ]

report_lines += [
    "## Most Suspicious Hotels\n",
    "| Hotel | Avg Score | Total Reviews | Flagged | % Flagged |",
    "|-------|-----------|--------------|---------|-----------|",
]
for _, row in top10_hotels_md.iterrows():
    report_lines.append(
        f"| {md_escape(str(row['hotel'])[:40])} | {row['avg_score']:.1f} | "
        f"{int(row['total'])} | {int(row['flagged'])} | {row['pct_flagged']:.1f}% |"
    )

report_lines += [
    "",
    "## Duplicate Clusters Found\n",
]
if cluster_summaries:
    for i, cl in enumerate(cluster_summaries[:5], 1):
        report_lines += [
            f"### Cluster {i} — {cl['count']} near-identical reviews",
            f"**Hotels involved:** {', '.join(cl['hotels'])}",
            f"> {cl['text'][:300]}",
            "",
        ]
else:
    report_lines.append("No high-confidence duplicate clusters detected.\n")

report_lines += [
    "## Timing Bursts Detected\n",
    "| Hotel | Date | Reviews |",
    "|-------|------|---------|",
]
for r in burst_data[:10]:
    report_lines.append(f"| {md_escape(str(r.get('hotel','?'))[:35])} | {r.get('date_only','?')} | {r.get('count','?')} |")

report_lines += [
    "",
    "## Fraud Signal Counts\n",
    "| Signal | Name | Reviews Triggered |",
    "|--------|------|-------------------|",
]
for i, (name, count) in enumerate(sig_counts.items(), 1):
    report_lines.append(f"| S{i} | {name} | {count:,} |")

report_lines += [
    "",
    "## Genuine vs Suspicious Language\n",
    "**Genuine reviews** tend to be longer, mention specific hotel features (room numbers, "
    "staff names, breakfast quality, specific amenities), and use moderate sentiment.\n",
    "**Suspicious reviews** tend to be very short, use generic superlatives (amazing, perfect, "
    "worst ever), contain excessive exclamation marks, and appear in clusters around the same date.\n",

    "## 5 Recommendations\n",
    "1. **Rate-limit reviews:** Impose a maximum of 3 reviews per user per day across all properties.",
    "2. **Short-extreme filter:** Automatically flag reviews under 15 words with 1- or 5-star ratings for manual review before publishing.",
    "3. **Velocity monitoring:** Alert operations teams when a hotel receives 5+ reviews in a single day.",
    "4. **Reviewer profile scoring:** Track each reviewer's rating entropy — accounts that exclusively post 5-star or 1-star reviews should receive additional verification.",
    "5. **Duplicate detection at submission:** Run cosine similarity at review submission time; block or flag reviews with >80% overlap with existing content.",
    "",
    "## Verdict\n",
    f"Manipulation is **present and measurable** in this dataset. "
    f"**{pct_suspicious:.0f}%** of reviews carry at least one fraud indicator. "
    f"The most pervasive pattern is **{top_signal}**, firing on {sig_counts[top_signal]:,} reviews. "
    f"The hotel with the highest average suspicion score is **{most_suspicious_hotel}**. "
    f"While not every flagged review is definitively fake, the volume and clustering of signals "
    f"suggest coordinated manipulation affecting at minimum the high-suspicion tier "
    f"({n_high + n_fake:,} reviews, {100*(n_high+n_fake)/total:.1f}% of the dataset).",
    "",
    "---",
    f"*Report generated by Fake Review Detector on {datetime.now().strftime('%Y-%m-%d')}*",
]

with open(os.path.join(OUT, "fraud_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print("  Saved: fraud_report.md")

# ═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print(f"Total flagged (score > 20): {n_suspicious:,} / {total:,} ({pct_suspicious:.1f}%)")
print(f"Most suspicious hotel: {most_suspicious_hotel}")
print(f"#1 fraud pattern: {top_signal} ({sig_counts[top_signal]:,} reviews)")
print(f"\nAll outputs saved to: {OUT}")
print("="*60)
