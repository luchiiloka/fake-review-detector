# Day 10 — Fake Review Detector
### Can You Trust What You Read Online? | 10x.in

---

## The Problem

Yesterday we analyzed what 35,000 hotel guests said.
Today we ask — can we trust them?

30-40% of online reviews are fake.
Paid praise. Fake attacks. Copy-paste bots.

Today we build a fraud detection system
that scores every review for suspicion.

---

## Same Data. Different Question.

| Yesterday (Day 9) | Today (Day 10) |
|---|---|
| What are guests saying? | Can we trust what they're saying? |
| Sentiment analysis | Fraud detection |
| Topics and emotions | Suspicion scoring |
| Which hotel is best? | Which hotel is cheating? |

---

## The 8 Fraud Signals

| Signal | What It Catches |
|---|---|
| Short + Extreme | 5-star review with only "Great hotel!" |
| Vague Language | Generic words, no specifics about the stay |
| Extreme Sentiment | Perfectly positive or negative — too good to be real |
| Duplicate Text | Same review copy-pasted across hotels |
| Reviewer Behavior | One person, 10 hotels, same day, all 5 stars |
| Rating Anomaly | Hotel with 95% five-star reviews |
| Timing Burst | 5 positive reviews on one hotel in one day |
| Exclamation Overuse | "AMAZING!!! BEST EVER!!!" — bots love exclamation marks |

---

## The Suspicion Score

```
0-20    Genuine          (green)
21-40   Low Suspicion    (yellow)
41-60   Moderate         (orange)
61-80   High Suspicion   (red)
81-100  Likely Fake      (dark red)
```

More signals = higher score = more suspicious.

---

## The 8 Outputs

### 1. Investigation Dashboard
Dark themed fraud dashboard. Case cards for suspicious reviews.
Signal radar. Hotel suspects. Genuine vs fake language. Timeline.

### 2. Suspicion Heatmap
Hotels vs fraud signals. Red hotspots show where fraud clusters.

### 3. Genuine vs Fake Word Cloud
Side by side — what real guests say vs what fake reviews say.

### 4. Signal Radar Chart
Spider chart — which fraud signals fire most?

### 5. Rating Comparison
How genuine reviews rate vs how suspicious reviews rate.
Fakes cluster at 1 and 5 stars. Real reviews spread across 2-4.

### 6. Timeline Bursts
Scatter plot showing normal review days vs suspicious spikes.

### 7. Top Suspects Poster
"Most Wanted" — top 5 most suspicious reviews as an infographic.

### 8. Fraud Investigation Report
Full written report with evidence, findings, recommendations.

---

## Project Structure

```
fake-review-detector/
├── CLAUDE.md                  ← Claude Code reads this
├── data/
│   └── hotel_reviews.csv      ← 35,000 hotel reviews
└── output/                    ← 8 files land here
```

---

## The Prompts

---

### Prompt 01 — Run It

```
Run it
```

---

### Prompt 02 — Dashboard

```
Open the dashboard
```

---

### Prompt 03 — Suspects

```
Show me the most suspicious reviews
```

---

### Prompt 04 — Cheating Hotels

```
Which hotels have the most fake reviews? Show evidence
```

---

### Prompt 05 — Language

```
Show me genuine vs fake word clouds
```

---

### Prompt 06 — Duplicates

```
Find copy-paste reviews
```

---

### Prompt 07 — Timing

```
Show timing bursts
```

---

### Prompt 08 — Investigation Report

```
Write the fraud investigation report. Make it visual
```

---

## What You Built Today

| Before | After |
|---|---|
| Trust all reviews blindly | Every review scored 0-100 |
| No way to spot fakes | 8 fraud signals checked |
| Trust hotel ratings | Manipulated ratings exposed |
| Can't find duplicates | Copy-paste clusters found |
| No timing analysis | Burst campaigns detected |

---

## The Real Insight

> The internet runs on trust.
> Reviews, ratings, recommendations.
> But 30-40% aren't real.
>
> Today you built a system that tells the difference.
> The signal from the noise. The real from the fake.

---

*Day 10 of 28 | Built with Claude Code | 10x.in*
