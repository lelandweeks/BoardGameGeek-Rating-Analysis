# BoardGameGeek Rating Analysis
DSCI 631: Applied Machine Learning — Final Project
Group 2 | Leland Weeks | Spring 2026

## Overview

This project analyzes board game ratings using data from [BoardGameGeek](https://boardgamegeek.com), the largest online database of board games. Using logistic, linear, and tree regression models, I explore three research questions about what makes a board game successful and how ratings relate to popularity and complexity.

**Dataset:** [Board Game Database from BoardGameGeek](https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek) (Kaggle)

---

## Research Questions

**Q1: What games are overrated and underrated?**
Classification task. Games are labeled overrated, middle-rated, or underrated based on the ratio of average rating to number of owners. A game with high ownership but low ratings is overrated, while a game with few owners but high ratings is underrated.

**Q2: Can we predict a game's average rating based on its features?**
Regression task. Uses numeric features (complexity, player counts, playtime, year published) and 373 binary features (mechanics, themes, subcategories) to predict average rating.

**Q3: Does complexity predict rating within a genre?**
Genre-segmented regression. Tests whether `GameWeight` (a 1-5 complexity scale) predicts average rating differently across the 8 BoardGameGeek genres.

---

## Results Summary

| Question | Model | Result |
|----------|-------|--------|
| Q1 | Logistic Regression | Accuracy: 0.578 |
| Q2 | Random Forest | R^2 = 0.543 |
| Q2 | Lasso (alpha=0.001) | R^2 = 0.425 |
| Q2 | Linear Regression | R^2 = 0.424 |
| Q3 | Linear Regression (Thematic) | R^2 = 0.306 |
| Q3 | Linear Regression (Party) | R^2 = -0.011 |

---

## Project Structure

```
├── data/                   # BGG dataset CSV files
├── notebooks/
│   ├── project_report.ipynb # includes EDA
│   └── project_proposal.ipynb
├── src/
│   ├── data.py             # Data loading and merging
│   ├── features.py         # Feature engineering
│   ├── models.py           # Model training (Linear, Lasso, Ridge, DT, RF, Logistic)
│   ├── evaluate.py         # Evaluation metrics
│   ├── logger.py           # Run logging
│   └── main.py             # Entry point
├── output/
│   └── runs_log.json       # Experiment log
└── README.md
```

---

## How to Run

**Requirements:** Python 3.12, pandas, numpy, scikit-learn

Install dependencies:
```bash
pip install pandas numpy scikit-learn
```

Run all three research questions:
```bash
python src/main.py
```

Run a specific question:
```bash
python src/main.py --model q1
python src/main.py --model q2
python src/main.py --model q3
```

Each run appends results to `output/runs_log.json` for experiment tracking.

---

## Key Findings

**Q1 — What games are overrated and underrated?**
- `NumOwned` is extremely right-skewed — need to apply log1p
- Positive correlation between ownership and rating
- Equal-bin `pd.qcut` produces balanced classes

**Q2 — What features make a boardgame enjoyable?**
- `AvgRating` is approximately normal — great regression target
- `GameWeight` is the strongest predictor
- `MfgPlaytime`, `NumExpansions`, `MaxPlayers` are severely right-skewed — log1p applied
- `YearPublished` has extreme negative skew — log does not help, used as-is
- StandardScaler applied before linear models
- 400+ binary features from mechanics/themes/subcategories

**Q3 — Does complexity predict rating within a genre?**
- `GameWeight` varies across genres
- Thematic and Strategy genres are the strongest `GameWeight` signal
- Party games show zero complexity is not predictive
- Genre-level differences justify regression modeling