# Fake Review Detector

> You are a fraud detection analyst. Analyze hotel reviews in `data/hotel_reviews.csv`
> and flag suspicious, fake, or manipulated reviews using statistical patterns and text analysis.

## Setup
- Reviews CSV is in `data/hotel_reviews.csv`
- ~35,000 hotel reviews
- Key columns: name, reviews.rating, reviews.text, reviews.title, reviews.date, reviews.username, city, country, reviews.doRecommend
- Save all outputs in `output/` folder
- Install missing packages: textblob pandas matplotlib numpy scikit-learn pillow seaborn wordcloud
- Do NOT sample — use ALL reviews

## Fraud Signals (score each review)

### Signal 1 — Review Length
- Under 15 words + rating 5 = suspicious praise (+20 points)
- Under 15 words + rating 1 = suspicious attack (+20 points)

### Signal 2 — Vague vs Specific Language
- Count vague words: great, good, bad, nice, terrible, amazing, worst, best, awesome, horrible, loved, hated, perfect, recommend
- Count specific words: room, floor, lobby, parking, breakfast, pool, shower, bed, wifi, staff name, neighborhood, elevator, checkout
- Vague-to-specific ratio > 3 = suspicious (+15 points)

### Signal 3 — Extreme Sentiment
- TextBlob polarity > 0.8 or < -0.8 = suspiciously extreme (+15 points)

### Signal 4 — Duplicate / Similar Text
- Find reviews with 80%+ same words = near-duplicate (+20 points)
- Group duplicates into clusters

### Signal 5 — Reviewer Behavior
- Same reviewer, multiple hotels, same day = suspicious (+15 points)
- Reviewer who only posts 5-star or only 1-star = suspicious (+10 points)

### Signal 6 — Rating Distribution Anomaly
- Hotel with 90%+ five-star reviews = likely manipulated (+10 points per review)

### Signal 7 — Timing Bursts
- 5+ reviews for one hotel in one day = suspicious burst (+15 points each)

### Signal 8 — Exclamation Overuse
- 3+ exclamation marks in a review = suspicious enthusiasm (+10 points)
- Real reviews rarely use many exclamation marks

## Suspicion Score (0-100)
- 0-20: Genuine (green)
- 21-40: Low Suspicion (yellow)
- 41-60: Moderate Suspicion (orange)
- 61-80: High Suspicion (red)
- 81-100: Likely Fake (dark red)

## Output Files

### 1. `output/dashboard.html`
Full investigation dashboard — dark theme (#0a0a0a), modern design:

**Hero section:**
- Big numbers: total reviews, % genuine, % suspicious, % likely fake
- Animated-looking gradient header
- Color accents: green genuine, red fake

**Investigation Board section:**
- Show top 15 most suspicious reviews as "case cards"
- Each card shows: review text, hotel name, rating badge, suspicion score as a colored meter bar, which signals triggered as small badges
- Cards should look like a police investigation board — dark cards with red/orange accents

**Signal Radar section:**
- One big radar/spider chart showing which fraud signals fire most across the entire dataset
- Labels on each axis: Length, Vague, Extreme, Duplicate, Behavior, Rating, Timing, Exclamation

**Suspicious Hotels section:**
- Top 10 hotels with highest % of flagged reviews
- Each hotel as a card with: name, total reviews, flagged count, % flagged, suspicion level badge

**Genuine vs Fake Language section:**
- Two columns side by side
- Left: green card with "What genuine reviews sound like" — 5 example quotes
- Right: red card with "What suspicious reviews sound like" — 5 example quotes

**Duplicate Evidence section:**
- Show duplicate clusters — groups of reviews with near-identical text
- Each group in a card showing the repeated text and how many times it appears

**Timing Burst Timeline section:**
- Visual timeline showing review bursts
- Normal days in gray, burst days in red

**The Verdict section:**
- Overall assessment: how trustworthy is this dataset?
- Key finding in big text
- 3 bullet recommendations

All in ONE self-contained HTML file. Embed all charts as base64 PNG.

### 2. `output/suspicion_heatmap.png`
Heatmap grid: hotels (rows) x fraud signals (columns)
Color intensity shows how many reviews triggered each signal per hotel.
Red = many triggers. Gray = few.
Top 20 hotels only.

### 3. `output/genuine_vs_fake_wordcloud.png`
Split image — left half: green word cloud (genuine reviews under score 20)
Right half: red word cloud (suspicious reviews over score 50)
Both in ONE image side by side with a divider line.

### 4. `output/signal_radar.png`
Radar/spider chart showing 8 fraud signals.
Each axis = one signal. Value = how many reviews triggered it.
Filled area in semi-transparent red.

### 5. `output/rating_comparison.png`
Two rating distribution histograms side by side:
- Left: rating distribution of genuine reviews (score < 20)
- Right: rating distribution of suspicious reviews (score > 50)
Show the difference — suspicious reviews cluster at 5 and 1 stars.

### 6. `output/timeline_bursts.png`
Scatter plot: date on x-axis, reviews per day per hotel on y-axis.
Normal days in gray dots. Burst days (5+ reviews) in large red dots.
Label the burst dots with hotel name.

### 7. `output/top_suspects.png`
Infographic-style image showing top 5 most suspicious reviews:
Each review as a styled card with text, score bar, signal badges.
Like a "Most Wanted" poster for fake reviews.

### 8. `output/fraud_report.md`
Investigation report:
- Executive summary
- Total analyzed, total flagged, breakdown by level
- Top 10 most suspicious reviews with full text + evidence
- Most suspicious hotels with evidence
- Duplicate clusters found
- Timing bursts detected
- Genuine vs suspicious language analysis
- 5 recommendations for platform trust & safety
- Verdict: how widespread is manipulation in this dataset?

## Rules
1. Read this file first
2. Install packages silently
3. Write and run detector.py — fix errors yourself
4. Use ALL reviews, no sampling
5. Create ALL 8 output files
6. End with: total flagged, most suspicious hotel, #1 fraud pattern

## IMPORTANT: Visual Output Rule
When I ask follow-up questions:
- NEVER just print text in terminal
- ALWAYS create a NEW visual file (PNG or HTML) in output/
- Every answer must produce a file I can open and show on screen
- Keep terminal short — just say what file was created
