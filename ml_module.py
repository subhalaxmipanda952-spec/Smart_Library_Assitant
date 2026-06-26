"""
ML module (Section B Q1: capability iii, and part (c) of the question).

Demand prediction is framed as SUPERVISED LEARNING (regression):
given historical features about a book, predict a numeric "future
borrow count" (demand). This is the best-fit ML paradigm because we
have labeled historical outcomes (past borrow counts) to learn from.

Features engineered from data/borrowing_history.csv:
    - total_borrows_last_6_months
    - total_borrows_last_30_days
    - avg_loan_duration_days
    - num_copies_total
    - section (one-hot)
Target:
    - borrows_next_period (label, simulated by holding out the most
      recent period during training)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

HERE = Path(__file__).parent
HISTORY_PATH = HERE / "data" / "borrowing_history.csv"
CATALOGUE_PATH = HERE / "data" / "catalogue.json"
MODEL_PATH = HERE / "data" / "demand_model.joblib"


def build_features() -> pd.DataFrame:
    history = pd.read_csv(HISTORY_PATH, parse_dates=["borrow_date", "return_date"],
                           dtype={"isbn": str})
    with open(CATALOGUE_PATH) as f:
        catalogue = {b["isbn"]: b for b in json.load(f)}

    cutoff = history["borrow_date"].max() - pd.Timedelta(days=90)

    rows = []
    for isbn, grp in history.groupby("isbn"):
        book = catalogue[isbn]
        past = grp[grp["borrow_date"] <= cutoff]
        future = grp[grp["borrow_date"] > cutoff]

        loan_days = (past["return_date"] - past["borrow_date"]).dt.days
        rows.append({
            "isbn": isbn,
            "title": book["title"],
            "section": book["section"],
            "copies_total": book["copies_total"],
            "borrows_last_180d": len(past[past["borrow_date"] > cutoff - pd.Timedelta(days=180)]),
            "borrows_last_30d": len(past[past["borrow_date"] > cutoff - pd.Timedelta(days=30)]),
            "avg_loan_duration": loan_days.mean() if len(loan_days) else 14.0,
            "borrows_next_period": len(future),  # label
        })

    return pd.DataFrame(rows)


def train_model() -> dict:
    df = build_features()
    df_encoded = pd.get_dummies(df, columns=["section"])

    feature_cols = [c for c in df_encoded.columns
                     if c not in ("isbn", "title", "borrows_next_period")]
    X = df_encoded[feature_cols]
    y = df_encoded["borrows_next_period"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_PATH)

    df_encoded["predicted_demand"] = model.predict(X)
    result = df_encoded[["title", "borrows_next_period", "predicted_demand"]].copy()
    result = result.merge(df[["title", "section"]], on="title")
    result = result.sort_values("predicted_demand", ascending=False)

    return {"mae": mae, "table": result}


def predict_demand_for_all() -> pd.DataFrame:
    if not MODEL_PATH.exists():
        train_model()
    bundle = joblib.load(MODEL_PATH)
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    df = build_features()
    df_encoded = pd.get_dummies(df, columns=["section"])
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded["predicted_demand"] = model.predict(df_encoded[feature_cols])

    out = df_encoded[["title", "section" if "section" in df_encoded.columns else feature_cols[0]]].copy() \
        if "section" in df_encoded.columns else df_encoded[["title"]].copy()
    out = df[["title", "section"]].copy()
    out["predicted_demand"] = df_encoded["predicted_demand"].round(1)
    return out.sort_values("predicted_demand", ascending=False).reset_index(drop=True)
