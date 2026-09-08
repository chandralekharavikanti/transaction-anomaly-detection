"""
detect_anomalies.py
--------------------
Loads the transaction dataset, engineers a small set of behavioral features,
trains an Isolation Forest to flag anomalous transactions, and writes out
a scored dataset ready to load into Power BI.

Run: python src/detect_anomalies.py
Output: outputs/scored_transactions.csv
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

CONTAMINATION = 0.03  # our assumed % of anomalies in the data (matches generator)


def load_data(path="data/transactions.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def engineer_features(df):
    """
    Turn raw transaction fields into signals a model can actually use.
    Each feature here maps to a real fraud/anomaly heuristic analysts use:
      - hour_of_day: fraud often clusters at odd hours (2-4 AM)
      - amount: unusually large transactions are a classic red flag
      - account_age_days: brand-new accounts making big payments is risky
      - transactions_last_24h: a burst of activity ("velocity") is suspicious
      - amount_to_avg_ratio: how unusual this amount is vs. the account's own history
    """
    df = df.copy()
    df["hour_of_day"] = df["timestamp"].dt.hour

    # Per-account historical average amount, used to spot "this is unusual FOR THIS USER"
    account_avg = df.groupby("account_id")["amount"].transform("mean")
    df["amount_to_avg_ratio"] = df["amount"] / account_avg.replace(0, 1)

    feature_cols = [
        "amount",
        "hour_of_day",
        "account_age_days",
        "transactions_last_24h",
        "amount_to_avg_ratio",
    ]
    return df, feature_cols


def run_isolation_forest(df, feature_cols):
    X = df[feature_cols]

    # Scale features so no single column (e.g. amount, which can be in the
    # thousands) dominates the distance-based splits the model uses internally.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42,
    )
    model.fit(X_scaled)

    # decision_function: higher = more "normal", lower/negative = more anomalous
    df["anomaly_score"] = model.decision_function(X_scaled)
    # predict: -1 = anomaly, 1 = normal -> convert to a readable flag
    raw_pred = model.predict(X_scaled)
    df["is_flagged_anomaly"] = np.where(raw_pred == -1, 1, 0)

    return df, model


def evaluate(df):
    """
    Compare model flags against the ground-truth labels we injected when
    generating the data. In a real deployment you wouldn't have these labels -
    this step exists purely to sanity-check that the model is working.
    """
    true_pos = ((df["is_flagged_anomaly"] == 1) & (df["is_actual_anomaly"] == 1)).sum()
    false_pos = ((df["is_flagged_anomaly"] == 1) & (df["is_actual_anomaly"] == 0)).sum()
    false_neg = ((df["is_flagged_anomaly"] == 0) & (df["is_actual_anomaly"] == 1)).sum()

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0

    print("--- Evaluation against injected ground-truth anomalies ---")
    print(f"True Positives:  {true_pos}")
    print(f"False Positives: {false_pos}")
    print(f"False Negatives: {false_neg}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")


def main():
    df = load_data()
    df, feature_cols = engineer_features(df)
    df, model = run_isolation_forest(df, feature_cols)

    evaluate(df)

    output_cols = [
        "transaction_id", "timestamp", "account_id", "amount",
        "merchant_category", "payment_method", "location",
        "account_age_days", "transactions_last_24h",
        "hour_of_day", "amount_to_avg_ratio",
        "anomaly_score", "is_flagged_anomaly", "is_actual_anomaly",
    ]
    df[output_cols].to_csv("outputs/scored_transactions.csv", index=False)
    print("\nSaved scored dataset to outputs/scored_transactions.csv")


if __name__ == "__main__":
    main()
