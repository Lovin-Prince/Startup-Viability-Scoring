"""
STEP 2 — Feature engineering & preprocessing
----------------------------------------------
Goal: turn raw columns into a clean matrix the models can learn from.

Key decisions (be ready to explain WHY, not just WHAT):
1. Drop identifiers (startup_id) — they carry no predictive signal and
   would let a tree-based model "memorize" rows instead of generalizing.
2. One-hot encode categoricals (industry, funding_stage, state) — these
   have no natural order, so label-encoding them as 0/1/2/3 would wrongly
   imply an ordinal relationship.
3. Log-transform heavily skewed money columns (funding_total_usd,
   monthly_burn_rate_usd) — raw dollar amounts span several orders of
   magnitude; log-scaling keeps a handful of huge outliers from
   dominating distance-based / gradient-based learning.
4. Scale numeric features — Logistic Regression is sensitive to feature
   scale (it optimizes via gradient descent on a weighted sum); tree
   models don't need this but it doesn't hurt them.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("startup_data.csv")

# --- 1. Drop non-predictive identifier ---
df = df.drop(columns=["startup_id"])

# --- 2. Log-transform skewed monetary columns ---
df["log_funding_total"] = np.log1p(df["funding_total_usd"])
df["log_burn_rate"] = np.log1p(df["monthly_burn_rate_usd"])
df = df.drop(columns=["funding_total_usd", "monthly_burn_rate_usd"])

# --- 3. One-hot encode categoricals ---
df = pd.get_dummies(df, columns=["industry", "funding_stage", "state"], drop_first=True)

# --- 4. Separate features / target ---
X = df.drop(columns=["success"])
y = df["success"]

# --- 5. Train/test split (stratified — preserves the success/fail ratio in both sets) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 6. Scale numeric features (fit on train only — never let test data leak into fitting!) ---
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print(f"Train shape: {X_train.shape}  |  Test shape: {X_test.shape}")
print(f"Train success rate: {y_train.mean():.1%}  |  Test success rate: {y_test.mean():.1%}")
