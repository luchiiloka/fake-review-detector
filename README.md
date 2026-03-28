#  Fake Review Detector

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

I built this project because I kept wondering how many of the reviews I read online are actually real? Platforms like TripAdvisor and Booking.com have millions of reviews, and it's genuinely hard to know which ones to trust. So I decided to build something that tries to answer that question using data.

This tool takes a dataset of hotel reviews and scores every single one across 8 different fraud signals — things like suspiciously short text, duplicate reviews posted across multiple hotels, or a reviewer who somehow visited 5 different hotels on the same day. Each review ends up with a suspicion score from 0 to 100, and the results are displayed in an interactive dashboard you can explore.

---

## What it does

You load in a CSV of hotel reviews, and the app:

- Runs every review through 8 fraud detection checks
- Gives each review a suspicion score (0 = looks genuine, 100 = very suspicious)
- Shows you which hotels have the most flagged reviews
- Groups together reviews that are nearly identical word-for-word
- Flags days where a hotel got an unusual spike in reviews
- Lets you paste in any review text and score it live

The whole thing runs in a Streamlit web app with interactive Plotly charts — you can hover over data points, click on a hotel bar to see its reviews, zoom into specific time periods, and filter everything from the sidebar.

---

## The 8 fraud signals

These are the checks I built into the scoring system. Each one adds points to the suspicion score:

| Signal | What I look for | Points |
|---|---|---|
| **Length** | Very short review (under 15 words) with a 1 or 5 star rating | +20 |
| **Vague Language** | Way more "amazing/perfect/terrible" than actual hotel details | +15 |
| **Extreme Sentiment** | TextBlob polarity above 0.8 or below -0.8 — unnaturally extreme | +15 |
| **Duplicate Text** | 80%+ text overlap with another review in the dataset | +20 |
| **Reviewer Behaviour** | Same person reviewing multiple hotels on the same day, or only ever giving 1s and 5s | +10–15 |
| **Rating Anomaly** | Hotel where over 90% of reviews are 5 stars | +10 |
| **Timing Burst** | 5 or more reviews for one hotel posted on a single day | +15 |
| **Exclamation Overuse** | 3+ exclamation marks — real reviews rarely look this enthusiastic | +10 |

Once all signals are tallied up, the score gets capped at 100 and labelled:

```
 0–20   Genuine
21–40   Low Suspicion
41–60   Moderate
61–80   High Suspicion
81–100  Likely Fake
```

---

## What I found in the sample dataset

Running this on 35,912 real hotel reviews:

- **11.2%** of reviews flagged at least one fraud signal
- **2,044** clusters of near-identical duplicate reviews found
- The biggest pattern was **Reviewer Behaviour** — flagged on 12,948 reviews
- The most suspicious hotel in the dataset was **A Swallow's Nest Motel**

---

## Dashboard tabs

```
 Overview            Stats summary, donut chart, rating breakdown, zoomable timeline
 Suspicious Reviews  Top flagged reviews shown as investigation-style cards
 Hotels              Click any hotel bar to drill into its individual reviews
 Signals             radar chart + bar chart, click a signal to see which reviews triggered it
 Heatmap             hotel × signal grid showing where fraud clusters
 Duplicates          groups of near-identical reviews side by side
 Search              search by hotel name or keyword + live single review scorer
```

---

## Tech stack

- **Python** — main language
- **Streamlit** — the web app
- **Plotly** — all the interactive charts
- **Pandas / NumPy** — data wrangling
- **TextBlob** — sentiment analysis for Signal 3
- **scikit-learn** — TF-IDF + cosine similarity for duplicate detection
- **Matplotlib / Seaborn** — static chart exports
- **WordCloud / Pillow** — word cloud image generation

---

## Getting it running

**1. Clone the repo**
```bash
git clone https://github.com/luchiiloka/fake-review-detector.git
cd fake-review-detector
```

**2. Set up a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Drop in your data**

Put your `hotel_reviews.csv` in the `data/` folder. It needs these columns:

| Column | What it is |
|---|---|
| `reviews.text` | The actual review text |
| `reviews.rating` | Star rating (numeric) |
| `name` | Hotel name |
| `reviews.username` | Who wrote it |
| `reviews.date` | When it was written |

**5. Run the app**
```bash
streamlit run app.py
```

**6. Or just generate the static reports**
```bash
python detector.py
```
This drops 8 output files into the `output/` folder — charts, a full HTML dashboard, and a markdown investigation report.

---

## Project structure

```
fake-review-detector/
│
├── app.py               # Streamlit web app
├── detector.py          # Analysis engine + static output generator
├── requirements.txt     # Dependencies
├── README.md
│
├── data/
│   └── hotel_reviews.csv
│
└── output/
    ├── dashboard.html
    ├── fraud_report.md
    ├── signal_radar.png
    ├── suspicion_heatmap.png
    ├── genuine_vs_fake_wordcloud.png
    ├── rating_comparison.png
    ├── timeline_bursts.png
    └── top_suspects.png
```

---

---

## About me

**Blessing Iloka**
- GitHub: [@luchiiloka](https://github.com/luchiiloka)
- Email: boluchi23@gmail.com

---

MIT License — feel free to use or build on this.
