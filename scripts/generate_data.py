"""
generate_data.py
----------------
Generates synthetic AI inference logs and customer usage data
simulating 20M+ daily AI inference events across multiple models.
"""

import os
import json
import random
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()
random.seed(99)
np.random.seed(99)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

MODELS = {
    "gpt-4-turbo":       {"cost_per_1k": 0.030, "avg_latency_ms": 1800, "weight": 0.20},
    "gpt-3.5-turbo":     {"cost_per_1k": 0.002, "avg_latency_ms": 600,  "weight": 0.35},
    "claude-2":          {"cost_per_1k": 0.008, "avg_latency_ms": 900,  "weight": 0.15},
    "llama-3-70b":       {"cost_per_1k": 0.001, "avg_latency_ms": 400,  "weight": 0.20},
    "mistral-8x7b":      {"cost_per_1k": 0.0007,"avg_latency_ms": 350,  "weight": 0.10},
}
MODEL_NAMES   = list(MODELS.keys())
MODEL_WEIGHTS = [MODELS[m]["weight"] for m in MODEL_NAMES]

USE_CASES = ["chat", "summarisation", "code_gen", "classification", "extraction", "qa", "translation"]
INDUSTRIES = ["fintech", "healthtech", "e-commerce", "edtech", "saas", "media", "logistics"]


def generate_inference_logs(n=2_000_000):
    """
    Generates 2M inference log rows (scaled down from 20M daily for local use).
    Each row represents one API call to an AI model.
    """
    print(f"[+] Generating {n:,} AI inference log records...")
    base = datetime(2024, 1, 1)

    models     = np.random.choice(MODEL_NAMES, n, p=MODEL_WEIGHTS)
    use_cases  = np.random.choice(USE_CASES, n)
    latencies  = np.array([
        max(50, np.random.normal(MODELS[m]["avg_latency_ms"],
                                 MODELS[m]["avg_latency_ms"] * 0.3))
        for m in models
    ])
    input_tokens  = np.random.lognormal(mean=5.5, sigma=1.0, size=n).astype(int)
    output_tokens = np.random.lognormal(mean=4.8, sigma=0.9, size=n).astype(int)
    total_tokens  = input_tokens + output_tokens
    costs         = np.array([MODELS[m]["cost_per_1k"] for m in models]) * total_tokens / 1000

    # Error rate: ~3% of calls fail
    error_types = [None, "timeout", "rate_limit", "context_length", "server_error"]
    errors      = np.random.choice(error_types, n, p=[0.97, 0.01, 0.01, 0.005, 0.005])

    # CSAT: 1–5, correlated with latency & success
    csat = np.where(
        errors == None,
        np.clip(np.random.normal(4.2, 0.8, n), 1, 5).round().astype(int),
        np.clip(np.random.normal(2.1, 0.9, n), 1, 5).round().astype(int),
    )

    timestamps = [base + timedelta(seconds=random.randint(0, 31_536_000)) for _ in range(n)]

    df = pd.DataFrame({
        "log_id":        [f"LOG-{i:010d}" for i in range(n)],
        "timestamp":     timestamps,
        "customer_id":   np.random.randint(1000, 50000, n),
        "model":         models,
        "use_case":      use_cases,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "total_tokens":  total_tokens,
        "latency_ms":    latencies.round(1),
        "cost_usd":      costs.round(6),
        "error_type":    errors,
        "csat_score":    csat,
        "is_success":    (errors == None).astype(int),
    })

    out = os.path.join(RAW_DIR, "inference_logs.csv")
    df.to_csv(out, index=False)
    print(f"    Saved → {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    return df


def generate_customer_profiles(n=10_000):
    print(f"[+] Generating {n:,} customer profiles...")
    plans = ["free_tier", "starter", "growth", "enterprise"]
    df = pd.DataFrame({
        "customer_id":   range(1000, 1000 + n),
        "company":       [fake.company() for _ in range(n)],
        "industry":      np.random.choice(INDUSTRIES, n),
        "plan":          np.random.choice(plans, n, p=[0.30, 0.35, 0.25, 0.10]),
        "signup_date":   [fake.date_between(start_date="-3y", end_date="-1m") for _ in range(n)],
        "region":        np.random.choice(["NA", "EU", "APAC", "LATAM", "MEA"], n,
                                          p=[0.40, 0.30, 0.18, 0.08, 0.04]),
        "monthly_budget_usd": np.round(np.random.exponential(scale=500, size=n), 2),
        "is_churned":    np.random.choice([0, 1], n, p=[0.88, 0.12]),
    })
    out = os.path.join(RAW_DIR, "customer_profiles.csv")
    df.to_csv(out, index=False)
    print(f"    Saved → {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    return df


def generate_model_versions(n=500):
    """Simulates model deployment & A/B test records."""
    print(f"[+] Generating {n} model version/experiment records...")
    experiments = []
    base = datetime(2024, 1, 1)
    for i in range(n):
        model = random.choice(MODEL_NAMES)
        experiments.append({
            "experiment_id":  f"EXP-{i:05d}",
            "model":          model,
            "variant":        random.choice(["control", "treatment_A", "treatment_B"]),
            "start_date":     base + timedelta(days=random.randint(0, 365)),
            "duration_days":  random.randint(3, 30),
            "metric":         random.choice(["latency", "csat", "cost", "error_rate"]),
            "baseline_value": round(random.uniform(0.5, 5.0), 3),
            "treatment_value":round(random.uniform(0.5, 5.0), 3),
            "p_value":        round(random.uniform(0.001, 0.2), 4),
            "significant":    int(random.random() < 0.3),
        })
    df = pd.DataFrame(experiments)
    out = os.path.join(RAW_DIR, "model_experiments.csv")
    df.to_csv(out, index=False)
    print(f"    Saved → {out}")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("  AI Model Performance & Usage Analytics — Data Generator")
    print("=" * 60)
    generate_inference_logs()
    generate_customer_profiles()
    generate_model_versions()
    print("\n✅ All AI analytics data generated.")
