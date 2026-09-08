"""
generate_data.py
-----------------
Generates a synthetic digital payments transaction dataset with a small
percentage of injected anomalies (fraud-like patterns), simulating what
a real fintech transaction log might look like.

Run: python src/generate_data.py
Output: data/transactions.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_TRANSACTIONS = 10000
ANOMALY_RATE = 0.03  # ~3% of transactions are anomalous

MERCHANT_CATEGORIES = [
    "Grocery", "Electronics", "Travel", "Dining", "Utilities",
    "Entertainment", "Healthcare", "Retail", "Fuel", "Subscription"
]

PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Wallet"]

LOCATIONS = [
    "Cincinnati", "New York", "Chicago", "Austin", "Seattle",
    "Miami", "Denver", "Boston", "Atlanta", "Phoenix"
]


def generate_normal_transactions(n):
    start_date = datetime(2026, 1, 1)
    data = {
        "transaction_id": [f"TXN{100000 + i}" for i in range(n)],
        "timestamp": [
            start_date + timedelta(
                days=np.random.randint(0, 240),
                hours=np.random.randint(6, 23),  # normal active hours
                minutes=np.random.randint(0, 60),
            )
            for _ in range(n)
        ],
        "account_id": [f"ACC{np.random.randint(1000, 3000)}" for _ in range(n)],
        "amount": np.round(np.random.gamma(shape=2.0, scale=45, size=n), 2),
        "merchant_category": np.random.choice(MERCHANT_CATEGORIES, size=n),
        "payment_method": np.random.choice(PAYMENT_METHODS, size=n),
        "location": np.random.choice(LOCATIONS, size=n),
        "account_age_days": np.random.randint(30, 2000, size=n),
        "transactions_last_24h": np.random.poisson(lam=2, size=n),
    }
    return pd.DataFrame(data)


def inject_anomalies(df, rate):
    n_anomalies = int(len(df) * rate)
    anomaly_idx = np.random.choice(df.index, size=n_anomalies, replace=False)

    df["is_actual_anomaly"] = 0  # ground truth label, kept aside for evaluation only

    for idx in anomaly_idx:
        anomaly_type = np.random.choice(
            ["high_amount", "odd_hour", "velocity_spike", "new_account_high_value"]
        )

        if anomaly_type == "high_amount":
            df.loc[idx, "amount"] = np.round(np.random.uniform(2000, 8000), 2)

        elif anomaly_type == "odd_hour":
            ts = df.loc[idx, "timestamp"]
            df.loc[idx, "timestamp"] = ts.replace(hour=np.random.choice([1, 2, 3, 4]))

        elif anomaly_type == "velocity_spike":
            df.loc[idx, "transactions_last_24h"] = np.random.randint(15, 40)

        elif anomaly_type == "new_account_high_value":
            df.loc[idx, "account_age_days"] = np.random.randint(0, 5)
            df.loc[idx, "amount"] = np.round(np.random.uniform(1000, 4000), 2)

        df.loc[idx, "is_actual_anomaly"] = 1

    return df


def main():
    df = generate_normal_transactions(N_TRANSACTIONS)
    df = inject_anomalies(df, ANOMALY_RATE)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv("data/transactions.csv", index=False)
    print(f"Generated {len(df)} transactions, {df['is_actual_anomaly'].sum()} labeled anomalies.")
    print("Saved to data/transactions.csv")


if __name__ == "__main__":
    main()
