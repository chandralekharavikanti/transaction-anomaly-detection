"""
query_tools.py
--------------
Safe, purpose-built functions for querying the scored transaction dataset.

Rather than letting the LLM generate and execute arbitrary code (a security
risk), it is given a small set of well-defined tools it can call. The LLM's
job is to decide *which* tool fits the user's question and *what arguments*
to pass — the actual data access is fully controlled by this code.
"""

import pandas as pd
from pathlib import Path

# Resolve the path relative to this file's location (not the current working
# directory), so this works whether you run the app from the project root
# or from inside the chatbot/ folder.
DATA_PATH = Path(__file__).resolve().parent.parent / "outputs" / "scored_transactions.csv"


def _load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])


def filter_transactions(location=None, merchant_category=None,
                         payment_method=None, only_flagged=None,
                         min_amount=None, max_amount=None):
    """
    Filter transactions by any combination of fields and return a summary
    (count + a small sample), not the full raw table, to keep responses
    concise and avoid dumping huge amounts of data back through the LLM.
    """
    df = _load_data()

    if location:
        df = df[df["location"].str.lower() == location.lower()]
    if merchant_category:
        df = df[df["merchant_category"].str.lower() == merchant_category.lower()]
    if payment_method:
        df = df[df["payment_method"].str.lower() == payment_method.lower()]
    if only_flagged is not None:
        df = df[df["is_flagged_anomaly"] == (1 if only_flagged else 0)]
    if min_amount is not None:
        df = df[df["amount"] >= min_amount]
    if max_amount is not None:
        df = df[df["amount"] <= max_amount]

    sample = df.sort_values("anomaly_score").head(5)[
        ["transaction_id", "amount", "merchant_category", "location",
         "payment_method", "is_flagged_anomaly", "anomaly_score"]
    ]

    return {
        "matching_count": int(len(df)),
        "sample_rows": sample.to_dict(orient="records"),
    }


def aggregate_by_column(column, only_flagged=True):
    """
    Group transactions by a categorical column (e.g. 'merchant_category',
    'location', 'payment_method') and return counts, sorted descending.
    Useful for "which X has the most anomalies" style questions.
    """
    df = _load_data()

    valid_columns = ["merchant_category", "location", "payment_method"]
    if column not in valid_columns:
        return {"error": f"column must be one of {valid_columns}"}

    if only_flagged:
        df = df[df["is_flagged_anomaly"] == 1]

    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "count"]

    return {"breakdown": counts.to_dict(orient="records")}


def get_top_anomalies(n=5):
    """
    Return the n most anomalous transactions (lowest anomaly_score = most
    unusual), for "show me the riskiest transactions" style questions.
    """
    df = _load_data()
    top = df.sort_values("anomaly_score").head(n)[
        ["transaction_id", "amount", "merchant_category", "location",
         "payment_method", "account_age_days", "transactions_last_24h",
         "anomaly_score"]
    ]
    return {"top_anomalies": top.to_dict(orient="records")}


# Tool schemas passed to the Claude API so it knows what each tool does
# and what arguments it accepts.
TOOL_DEFINITIONS = [
    {
        "name": "filter_transactions",
        "description": (
            "Filter transactions by location, merchant category, payment "
            "method, flagged status, or amount range. Returns a count and "
            "a small sample of matching transactions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "merchant_category": {"type": "string"},
                "payment_method": {"type": "string"},
                "only_flagged": {
                    "type": "boolean",
                    "description": "True = only flagged anomalies, False = only normal transactions",
                },
                "min_amount": {"type": "number"},
                "max_amount": {"type": "number"},
            },
        },
    },
    {
        "name": "aggregate_by_column",
        "description": (
            "Count flagged anomalies grouped by a category (merchant_category, "
            "location, or payment_method). Use for 'which X has the most "
            "anomalies' style questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "enum": ["merchant_category", "location", "payment_method"],
                },
                "only_flagged": {"type": "boolean"},
            },
            "required": ["column"],
        },
    },
    {
        "name": "get_top_anomalies",
        "description": "Return the n most anomalous (riskiest) transactions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number of transactions to return"},
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "filter_transactions": filter_transactions,
    "aggregate_by_column": aggregate_by_column,
    "get_top_anomalies": get_top_anomalies,
}
