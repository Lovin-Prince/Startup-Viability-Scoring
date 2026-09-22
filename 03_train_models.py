"""
STEP 3 — Train & benchmark three classifiers
-----------------------------------------------
Why these three specifically (a good interview answer):
  - Logistic Regression: a fast, interpretable BASELINE. If a complex
    model can't beat this, the complexity isn't earning its keep.
  - Decision Tree: interpretable, captures non-linear splits, but prone
    to overfitting on its own.
  - Gradient Boosting: builds trees sequentially, each one correcting
    the previous one's errors — usually the strongest of the three on
    tabular data like this, at the cost of interpretability.

We also handle class imbalance (76% success / 24% fail) with
class_weight="balanced" so the model doesn't just learn to always
predict "success".
"""

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
}

trained = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained[name] = model
    joblib.dump(model, f"model_{name.replace(' ', '_').lower()}.pkl")
    print(f"Trained: {name}")

print("\nAll models trained and saved.")
