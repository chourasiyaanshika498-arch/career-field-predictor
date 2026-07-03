"""
train.py
--------
Trains a multi-class XGBoost classifier to predict the best-fit career field
for a student, evaluates it, and saves:
  - trained model (models/model.json)
  - label encoder classes (models/classes.npy)
  - feature column order (models/features.json)
  - a SHAP summary plot (models/shap_summary.png)

Run:
    python train.py
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save the plot
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_PATH = "data/students.csv"
MODEL_DIR = "models"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["best_field"])
    y_raw = df["best_field"]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        random_state=42,
    )

    # Small grid search to tune key hyperparameters (kept small so it runs
    # fast on a low-RAM machine; expand if you have more compute available)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    }

    print("Running GridSearchCV (this may take a minute)...")
    grid = GridSearchCV(base_model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_
    print("Best params:", grid.best_params_)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print(f"Macro F1-score: {f1:.3f}\n")
    print(classification_report(y_test, preds, target_names=le.classes_))

    # --- Save model artifacts ---
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(f"{MODEL_DIR}/model.json")
    np.save(f"{MODEL_DIR}/classes.npy", le.classes_)
    with open(f"{MODEL_DIR}/features.json", "w") as f:
        json.dump(list(X.columns), f)

    # --- SHAP explainability ---
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # shap_values is a list (one array per class) for multiclass XGBoost
    plt.figure()
    if isinstance(shap_values, list):
        # Older SHAP API: list of (n_samples, n_features) arrays, one per class
        mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        # Newer SHAP API: single array shaped (n_samples, n_features, n_classes)
        mean_abs = np.mean(np.abs(shap_values), axis=2)
    else:
        mean_abs = shap_values

    shap.summary_plot(mean_abs, X_test, show=False, plot_type="bar")

    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/shap_summary.png", dpi=150)
    print(f"Saved SHAP summary plot to {MODEL_DIR}/shap_summary.png")

    print("\nTraining complete. Model artifacts saved in 'models/'.")


if __name__ == "__main__":
    main()
