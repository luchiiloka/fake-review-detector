# 🔍 Fake Review Detector

> An AI-powered fraud detection system that analyses hotel reviews to identify suspicious, fake, and manipulated content using statistical patterns and NLP text analysis.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

## 📌 Project Overview

This project analyses **35,000+ real hotel reviews** and scores each one across **8 fraud detection signals** to identify potentially fake or manipulated content. Results are displayed in a fully interactive web dashboard built with Streamlit and Plotly.

### 🎯 Why This Matters
Fake reviews cost businesses and consumers billions annually. Platforms like TripAdvisor, Booking.com and Google Maps all face this challenge. This tool demonstrates how data science can detect coordinated review fraud at scale.

---

## 🚀 Live Demo

> **Run it locally** — see setup instructions below.

---

## ✨ Features

| Feature | Description |
|---|---|
| **8 Fraud Signals** | Length, Vague Language, Extreme Sentiment, Duplicates, Reviewer Behaviour, Rating Anomaly, Timing Bursts, Exclamation Overuse |
| **Interactive Dashboard** | 7-tab Streamlit app with Plotly charts — hover, zoom, click-to-filter |
| **Suspicion Scoring** | Each review scored 0–100 with colour-coded risk labels |
| **Upload Any CSV** | Works with any hotel review dataset, not just the sample data |
| **Live Review Analyser** | Paste any review text and get an instant fraud score |
| **Click-to-Filter** | Click a hotel bar → see its reviews; click a signal bar → see affected reviews |
| **Zoomable Timeline** | Range slider + 3m/6m/1y/All buttons with burst day annotations |
| **Duplicate Clustering** | TF-IDF cosine similarity finds near-identical review clusters |

---

## 📊 Dashboard Tabs

```
📊 Overview          — Hero stats, donut chart, rating distribution, zoomable timeline
🚨 Suspicious Reviews — Top N flagged reviews as styled investigation cards
🏨 Hotels            — Bar chart + scatter plot with click-to-drill-down
📡 Signals           — Radar chart + bar chart with click-to-filter reviews
🌡️ Heatmap           — Interactive hotel × signal grid
📋 Duplicates        — Near-identical review clusters
🔎 Search            — Search by hotel/text + live single review analyser
```

---

## 🔬 Fraud Signals Explained

| Signal | Trigger | Score |
|---|---|---|
| **S1 — Length** | Under 15 words + rating 1 or 5 | +20 pts |
| **S2 — Vague Language** | Vague-to-specific word ratio > 3 | +15 pts |
| **S3 — Extreme Sentiment** | TextBlob polarity > 0.8 or < -0.8 | +15 pts |
| **S4 — Duplicate Text** | 80%+ cosine similarity to another review | +20 pts |
| **S5 — Reviewer Behaviour** | Multi-hotel same-day posting or always extreme ratings | +10–15 pts |
| **S6 — Rating Anomaly** | Hotel with 90%+ five-star reviews | +10 pts |
| **S7 — Timing Burst** | 5+ reviews for one hotel on one day | +15 pts |
| **S8 — Exclamation Overuse** | 3+ exclamation marks in a review | +10 pts |

### Suspicion Score Labels
```
 0–20  ✅ Genuine          (green)
21–40  🟡 Low Suspicion    (yellow)
41–60  🟠 Moderate         (orange)
61–80  🔴 High Suspicion   (red)
81–100 🚨 Likely Fake      (dark red)
```

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit** — interactive web app framework
- **Plotly** — interactive charts (donut, bar, scatter, radar, heatmap, timeline)
- **Pandas / NumPy** — data manipulation
- **TextBlob** — NLP sentiment analysis
- **scikit-learn** — TF-IDF vectorisation + cosine similarity for duplicate detection
- **Seaborn / Matplotlib** — static chart generation
- **WordCloud / Pillow** — word cloud images

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/fake-review-detector.git
cd fake-review-detector
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your data
Place your `hotel_reviews.csv` in the `data/` folder.

**Required columns:**
| Column | Description |
|---|---|
| `reviews.text` | Review body text |
| `reviews.rating` | Numeric rating |
| `name` | Hotel name |
| `reviews.username` | Reviewer username |
| `reviews.date` | Review date |

### 5. Run the app
```bash
streamlit run app.py
```

### 6. (Optional) Generate static output files
```bash
python detector.py
```
This creates all 8 output files in the `output/` folder.

---

## 📁 Project Structure

```
fake-review-detector/
│
├── app.py               # Streamlit web application
├── detector.py          # Core analysis engine + static output generator
├── requirements.txt     # Python dependencies
├── README.md            # This file
│
├── data/
│   └── hotel_reviews.csv    # Input dataset (35,912 reviews)
│
└── output/
    ├── dashboard.html           # Self-contained HTML report
    ├── fraud_report.md          # Markdown investigation report
    ├── signal_radar.png         # Radar chart (8 fraud signals)
    ├── suspicion_heatmap.png    # Hotel × signal heatmap
    ├── genuine_vs_fake_wordcloud.png
    ├── rating_comparison.png
    ├── timeline_bursts.png
    └── top_suspects.png
```

---

## 📈 Key Findings (Sample Dataset)

- **35,912** hotel reviews analysed
- **11.2%** of reviews show at least one fraud signal
- **2,044** duplicate text clusters detected
- **#1 fraud pattern:** Reviewer Behaviour — triggered on 12,948 reviews
- **Most suspicious hotel:** A Swallow's Nest Motel

---

## 🚀 Deployment

Deploy for free on **Streamlit Cloud**:

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → set main file to `app.py`
5. Click **Deploy** — your app gets a public URL!

---

## 💡 Future Improvements

- [ ] Train a supervised ML classifier using suspicion scores as labels
- [ ] Add REST API endpoint (FastAPI) for real-time review scoring
- [ ] Integrate SQLite for persistent storage and trend tracking
- [ ] Email alert system for burst activity detection
- [ ] Multilingual support with language detection

---

## 👤 Author

**Blessing Iloka**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- Email: bi21aaj@herts.ac.uk

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

*Built with Python & Streamlit · Fake Review Detector v1.0*
