"""
STEP 5 — Feature importance
------------------------------
Which signals actually drive the prediction? This is usually the part
a stakeholder (or interviewer) cares about most — the "so what."
"""

import pandas as pd
import joblib

X_train = pd.read_csv("X_train.csv")
gb_model = joblib.load("model_gradient_boosting.pkl")

importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": gb_model.feature_importances_
}).sort_values("importance", ascending=False).head(10)

importance_df.to_csv("feature_importance.csv", index=False)
print("Top 10 predictive features (Gradient Boosting):")
print(importance_df.to_string(index=False))
