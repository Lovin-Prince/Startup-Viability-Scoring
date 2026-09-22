# Startup Viability & Investment Risk Scoring

A classification pipeline that scores early-stage startups on likelihood
of success, using funding, operational, and industry attributes.
Built to benchmark multiple classifiers and surface the features that
actually drive investment risk.

## Overview

| | |
|---|---|
| **Problem type** | Binary classification (success / fail) |
| **Models compared** | Logistic Regression, Decision Tree, Gradient Boosting |
| **Key challenge** | Class imbalance (~77% success / 23% fail) |
| **Stack** | Python, Pandas, NumPy, Scikit-learn |
| **Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |

## Pipeline

```
src/
├── 01_generate_data.py         # Build the startup dataset
├── 02_feature_engineering.py   # Clean, encode, scale, split
├── 03_train_models.py          # Train Logistic Regression / Decision Tree / Gradient Boosting
├── 04_evaluate.py              # Score all models on held-out test data
└── 05_feature_importance.py    # Rank the strongest predictive signals
```

Run in order:

```bash
pip install -r requirements.txt
cd src
python 01_generate_data.py
python 02_feature_engineering.py
python 03_train_models.py
python 04_evaluate.py
python 05_feature_importance.py
```

Outputs (`model_comparison.csv`, `feature_importance.csv`) land in `src/`
and are also checked into `results/` for reference.

> **Note on data source:** `01_generate_data.py` synthesizes a dataset
> that mirrors the structure and statistical relationships of a real
> startup-funding dataset (industry, funding stage, funding totals,
> investor count, milestones, burn rate, etc.), with a success label
> that depends on the features plus realistic noise. Swap this step for
> `pd.read_csv("your_real_dataset.csv")` to run the identical pipeline
> on real data — nothing downstream changes.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.652 | 0.860 | 0.652 | 0.742 | **0.709** |
| Gradient Boosting | 0.768 | 0.778 | 0.976 | 0.866 | 0.673 |
| Decision Tree | 0.573 | 0.845 | 0.543 | 0.661 | 0.639 |

**Finding:** Logistic Regression achieved the best ROC-AUC despite being
the simplest model. `GradientBoostingClassifier` in scikit-learn doesn't
support `class_weight`, so it leaned toward the majority class — high
recall on "success" (0.98) but very low recall on "fail" (0.09). This is
a reminder that model complexity isn't automatically better once class
imbalance enters the picture; it depends on whether the model can
actually account for it.

### Top predictive features (Gradient Boosting)

1. `log_funding_total` — total capital raised (log-scaled)
2. `runway_months` — funding ÷ monthly burn rate
3. `log_burn_rate` — monthly spend (log-scaled)
4. `milestones_hit` — product/business milestones achieved
5. `has_top_tier_investor` — backing from a recognized investor

## Next steps

- Swap in a real startup-funding dataset (e.g. a public Crunchbase-style CSV)
- Add SMOTE or a custom sample-weighting scheme to fix the Gradient
  Boosting class-imbalance gap
- Build a Power BI dashboard on `model_comparison.csv` and
  `feature_importance.csv` to surface risk scores by industry and
  funding stage

## License

MIT
