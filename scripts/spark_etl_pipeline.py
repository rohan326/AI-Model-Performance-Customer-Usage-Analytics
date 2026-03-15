"""
spark_etl_pipeline.py
---------------------
Simulates the Spark ETL pipeline that consolidates 20M+ daily AI inference
logs into Snowflake-ready aggregation tables.

Local engine: Pandas (mirrors Spark DataFrame API patterns).
Production: PySpark → AWS Glue / Databricks → Snowflake via Spark Connector.
"""

import os
import time
import pandas as pd
import numpy as np

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ─── EXTRACT ──────────────────────────────────────────────────────────────────
def extract():
    log("EXTRACT  inference_logs.csv")
    logs = pd.read_csv(os.path.join(RAW_DIR, "inference_logs.csv"),
                       parse_dates=["timestamp"])
    log(f"         {len(logs):,} inference records")

    log("EXTRACT  customer_profiles.csv")
    customers = pd.read_csv(os.path.join(RAW_DIR, "customer_profiles.csv"),
                             parse_dates=["signup_date"])

    log("EXTRACT  model_experiments.csv")
    experiments = pd.read_csv(os.path.join(RAW_DIR, "model_experiments.csv"),
                               parse_dates=["start_date"])
    return logs, customers, experiments


# ─── TRANSFORM ────────────────────────────────────────────────────────────────
def transform_logs(logs):
    log("TRANSFORM inference_logs — feature engineering")
    df = logs.copy()

    df["date"]  = df["timestamp"].dt.date
    df["hour"]  = df["timestamp"].dt.hour
    df["week"]  = df["timestamp"].dt.isocalendar().week.astype(int)
    df["month"] = df["timestamp"].dt.month

    # Cost per token bucket
    df["cost_bucket"] = pd.cut(df["cost_usd"],
                                bins=[-np.inf, 0.001, 0.01, 0.1, np.inf],
                                labels=["micro", "low", "medium", "high"])

    # Latency tier
    df["latency_tier"] = pd.cut(df["latency_ms"],
                                 bins=[-np.inf, 300, 1000, 3000, np.inf],
                                 labels=["fast", "normal", "slow", "timeout"])

    # Token efficiency score (output / total tokens)
    df["token_efficiency"] = (df["output_tokens"] / df["total_tokens"]).round(4)

    log(f"         Features added: date, hour, week, cost_bucket, latency_tier, token_efficiency")
    return df


def build_daily_model_metrics(logs):
    log("AGGREGATE daily_model_metrics (Snowflake fact table)")
    result = (
        logs.groupby(["date", "model"])
        .agg(
            total_calls=("log_id", "count"),
            success_calls=("is_success", "sum"),
            total_tokens=("total_tokens", "sum"),
            avg_latency_ms=("latency_ms", "mean"),
            p95_latency_ms=("latency_ms", lambda x: x.quantile(0.95)),
            total_cost_usd=("cost_usd", "sum"),
            avg_csat=("csat_score", "mean"),
        )
        .round(3)
        .reset_index()
    )
    result["error_rate_pct"] = (
        (result["total_calls"] - result["success_calls"]) / result["total_calls"] * 100
    ).round(2)
    log(f"         {len(result):,} daily-model rows")
    return result


def build_customer_usage_summary(logs, customers):
    log("AGGREGATE customer_usage_summary (Snowflake dim table)")
    agg = (
        logs.groupby("customer_id")
        .agg(
            total_calls=("log_id", "count"),
            total_tokens=("total_tokens", "sum"),
            total_spend_usd=("cost_usd", "sum"),
            avg_csat=("csat_score", "mean"),
            error_rate=("is_success", lambda x: (1 - x.mean()) * 100),
            favourite_model=("model", lambda x: x.value_counts().idxmax()),
            top_use_case=("use_case", lambda x: x.value_counts().idxmax()),
        )
        .round(3)
        .reset_index()
    )
    merged = customers.merge(agg, on="customer_id", how="left")
    merged["total_calls"]     = merged["total_calls"].fillna(0).astype(int)
    merged["total_tokens"]    = merged["total_tokens"].fillna(0).astype(int)
    merged["total_spend_usd"] = merged["total_spend_usd"].fillna(0).round(2)
    log(f"         {len(merged):,} enriched customer rows")
    return merged


def build_usecase_performance(logs):
    log("AGGREGATE usecase_performance (Snowflake summary table)")
    result = (
        logs.groupby(["use_case", "model"])
        .agg(
            calls=("log_id", "count"),
            avg_input_tokens=("input_tokens", "mean"),
            avg_output_tokens=("output_tokens", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
            avg_cost_usd=("cost_usd", "mean"),
            avg_csat=("csat_score", "mean"),
            success_rate=("is_success", "mean"),
        )
        .round(4)
        .reset_index()
    )
    log(f"         {len(result):,} use-case × model rows")
    return result


# ─── LOAD (simulate Snowflake write via Parquet) ──────────────────────────────
def load_to_snowflake(df, table_name):
    out = os.path.join(PROC_DIR, f"{table_name}.parquet")
    df.to_parquet(out, index=False)
    log(f"LOAD     {table_name} → Snowflake  ({os.path.getsize(out)/1e6:.2f} MB)")


# ─── PIPELINE ─────────────────────────────────────────────────────────────────
def run_pipeline():
    print("=" * 60)
    print("  AI Inference ETL — Spark → Snowflake Pipeline")
    print("=" * 60)
    t0 = time.time()

    logs, customers, experiments = extract()
    logs_enriched = transform_logs(logs)

    daily_metrics   = build_daily_model_metrics(logs_enriched)
    customer_usage  = build_customer_usage_summary(logs_enriched, customers)
    usecase_perf    = build_usecase_performance(logs_enriched)

    load_to_snowflake(daily_metrics,  "fact_daily_model_metrics")
    load_to_snowflake(customer_usage, "dim_customer_usage")
    load_to_snowflake(usecase_perf,   "fact_usecase_performance")
    load_to_snowflake(experiments,    "dim_model_experiments")

    elapsed = time.time() - t0
    print(f"\n✅ Pipeline complete in {elapsed:.1f}s")
    print(f"   Tables written → {PROC_DIR}")


if __name__ == "__main__":
    run_pipeline()
