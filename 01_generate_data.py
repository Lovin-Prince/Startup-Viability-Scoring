"""
STEP 1 — Generate the dataset
------------------------------
In the real version of this project you'd pull data from a public source
(e.g. a Crunchbase-style startup funding dataset on Kaggle). Since this
environment has no internet access, this script generates a synthetic
dataset with the SAME shape and statistical relationships a real one
would have — realistic feature names, realistic noise, and a target
that depends on the features in a non-trivial way.

Swap this file out for a real CSV loader (pd.read_csv(...)) once you
plug in an actual dataset — everything downstream (02, 03, 04) doesn't
need to change.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 3000

industries = ["SaaS", "Fintech", "E-commerce", "Healthtech", "EdTech", "Biotech", "AI/ML"]
funding_stages = ["Seed", "Series A", "Series B", "Series C+"]
states = ["CA", "NY", "MA", "TX", "WA", "IL", "Other"]

df = pd.DataFrame({
    "startup_id": range(1, N + 1),
    "industry": np.random.choice(industries, N, p=[.22, .16, .18, .12, .10, .08, .14]),
    "funding_stage": np.random.choice(funding_stages, N, p=[.40, .30, .18, .12]),
    "state": np.random.choice(states, N, p=[.28, .16, .10, .12, .10, .09, .15]),
    "funding_total_usd": np.random.lognormal(mean=14, sigma=1.3, size=N).round(0),
    "funding_rounds": np.random.poisson(2, N) + 1,
    "num_investors": np.random.poisson(4, N) + 1,
    "founding_year": np.random.randint(2008, 2023, N),
    "last_funding_year": np.random.randint(2015, 2026, N),
    "num_founders": np.random.choice([1, 2, 3, 4], N, p=[.25, .45, .22, .08]),
    "num_employees": np.random.lognormal(mean=3.2, sigma=1.0, size=N).round(0).astype(int),
    "has_top_tier_investor": np.random.binomial(1, 0.28, N),
    "milestones_hit": np.random.poisson(2.2, N),
    "monthly_burn_rate_usd": np.random.lognormal(mean=10.5, sigma=0.9, size=N).round(0),
})

# months between founding and last funding round — a real signal of momentum
df["months_active"] = (df["last_funding_year"] - df["founding_year"]).clip(lower=0) * 12 + np.random.randint(0, 12, N)
df["runway_months"] = (df["funding_total_usd"] / df["monthly_burn_rate_usd"]).clip(0, 60).round(1)

# ---- construct a target that genuinely depends on the features (with noise) ----
# This mimics how real-world "success" labels correlate with funding signals,
# without just hard-coding the answer.
logit = (
    0.9 * df["has_top_tier_investor"]
    + 0.35 * np.log1p(df["funding_total_usd"]) - 5.2
    + 0.25 * df["milestones_hit"]
    + 0.15 * df["num_investors"]
    + 0.10 * df["runway_months"] / 12
    - 0.20 * (df["funding_stage"] == "Seed").astype(int)
    + 0.15 * (df["industry"].isin(["SaaS", "Fintech", "AI/ML"])).astype(int)
    + np.random.normal(0, 1.1, N)  # noise — real-world outcomes are never fully predictable
)
prob_success = 1 / (1 + np.exp(-logit))
df["success"] = np.random.binomial(1, prob_success)

df.to_csv("startup_data.csv", index=False)
print(f"Generated {len(df)} rows. Success rate: {df['success'].mean():.1%}")
print(df.head())
