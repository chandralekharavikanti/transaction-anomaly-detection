# Transaction Anomaly Detection

A lightweight, explainable anomaly detection system for digital payment
transactions, built to demonstrate how unsupervised machine learning can
support fraud monitoring in a fintech analytics context — designed to be
paired with a Power BI dashboard for stakeholder-facing reporting.

## Project Overview

Financial platforms process thousands of transactions per day, and only a
tiny fraction are ever fraudulent or erroneous. Manually reviewing all of
them isn't feasible, so this project uses an **Isolation Forest** — an
unsupervised anomaly detection algorithm — to automatically score every
transaction on "how unusual" it looks, so analysts can focus review effort
on the highest-risk transactions instead of the whole dataset.

## Why Isolation Forest?

Unlike classification models (e.g. logistic regression), Isolation Forest
doesn't need labeled fraud data to train — which reflects real-world
conditions, since most companies don't have a clean "this transaction was
fraud" label for every historical record. It works by randomly partitioning
the data and measuring how quickly a point gets "isolated" from the rest;
anomalies isolate faster because they sit far from normal clusters. This
makes it a practical, low-maintenance first line of defense.

## Data

Since no real transaction data was available, this project uses a
**synthetic dataset** of 10,000 transactions generated in `src/generate_data.py`,
modeled after realistic payment attributes: amount, merchant category,
payment method, location, account age, and recent transaction velocity.
About 3% of transactions have injected anomaly patterns (unusually high
amounts, odd-hour activity, sudden spikes in transaction frequency, and
high-value payments from brand-new accounts) — the kinds of patterns real
fraud analysts actually look for.

## Approach

1. **Data Generation** (`generate_data.py`) — Creates realistic transaction
   records and injects a small, known set of anomalies, so the model's
   performance can be validated against ground truth.
2. **Feature Engineering** (`detect_anomalies.py`) — Raw transaction fields
   are turned into behavioral signals:
   - `hour_of_day` — fraud often clusters at unusual hours
   - `amount_to_avg_ratio` — how unusual a transaction is *for that specific
     account*, not just in absolute terms
   - `transactions_last_24h` — sudden activity bursts ("velocity") are a
     common fraud signal
   - `account_age_days` — new accounts making large payments carry more risk
3. **Modeling** — Features are scaled and passed into an Isolation Forest,
   which assigns every transaction an anomaly score and a flag.
4. **Evaluation** — Model flags are compared against the injected
   ground-truth labels (precision/recall) purely to sanity-check performance;
   in production, this step wouldn't exist since real fraud labels usually
   aren't available up front.
5. **Output** — A scored CSV (`outputs/scored_transactions.csv`) is produced,
   ready to be loaded directly into Power BI for a monitoring dashboard.

## Results

On the synthetic test set, the model achieved:
- **Precision: ~0.79**
- **Recall: ~0.79**

against the known injected anomalies — solid performance for an unsupervised
model with no access to labels during training.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn (Isolation Forest), Power BI

## How to Run

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/detect_anomalies.py
```

Output lands in `outputs/scored_transactions.csv`.

## Next Steps

- Build a Power BI dashboard on top of `scored_transactions.csv`: flagged
  transaction volume over time, breakdown by merchant category/location,
  and a drill-through view for individual flagged transactions.
- Experiment with additional features (e.g. geographic distance between
  consecutive transactions).
