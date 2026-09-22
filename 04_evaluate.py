"""
STEP 4 — Evaluation
----------------------
Why these metrics specifically:
  - Accuracy alone is misleading on imbalanced data (76% success rate
    means "always predict success" gets 76% accuracy for free — useless).
  - Precision: of the startups we PREDICTED would succeed, how many
    actually did? (matters if false positives are costly — e.g. an
    investor acting on a bad recommendation)
  - Recall: of the startups that ACTUALLY succeeded, how many did we
    catch? (matters if false negatives are costly — missing a good bet)
  - F1: harmonic mean of precision & recall — a single balanced number.
  - ROC-AUC: how well the model ranks positives above negatives across
    ALL possible thresholds, not just the default 0.5 cutoff — the
    right metric when you care about ranking/scoring, not just a
    yes/no call.
"""

import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

model_files = {
    "Logistic Regression": "model_logistic_regression.pkl",
    "Decision Tree": "model_decision_tree.pkl",
    "Gradient Boosting": "model_gradient_boosting.pkl",
}

results = []
for name, path in model_files.items():
    model = joblib.load(path)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    })

    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["Fail", "Success"]))

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
results_df.to_csv("model_comparison.csv", index=False)
print("\nFinal comparison table:")
print(results_df.to_string(index=False))
