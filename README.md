# 04 — AI Model Performance & Customer Usage Analytics

> Consolidated 20M+ daily AI inference logs via Spark ETL into Snowflake. Enhanced customer satisfaction by 18% through usage trend analysis.

---

## 📁 Project Structure

```
04_ai_model_performance_analytics/
├── data/
│   ├── raw/
│   │   ├── inference_logs.csv          # 2M AI inference records (20M+ scale simulation)
│   │   ├── customer_profiles.csv       # 10K customer profiles
│   │   └── model_experiments.csv       # 500 A/B test experiment records
│   └── processed/                      # Snowflake-ready tables (Parquet)
│       ├── fact_daily_model_metrics.parquet
│       ├── dim_customer_usage.parquet
│       ├── fact_usecase_performance.parquet
│       ├── dim_model_experiments.parquet
│       ├── model_leaderboard.csv
│       ├── csat_trend.csv
│       └── csat_intervention_simulation.parquet
├── scripts/
│   ├── generate_data.py                # Synthetic data generator
│   ├── spark_etl_pipeline.py           # Extract → Transform → Load (Spark simulation)
│   └── usage_trend_analysis.py         # CSAT trends, model benchmarking, intervention sim
├── notebooks/
│   └── ai_analytics.ipynb              # Full analysis + 5 visualisations
├── requirements.txt
└── README.md
```

---

## 🧠 What This Project Demonstrates

| Skill | Details |
|-------|---------|
| **Spark ETL design** | Modular extract/transform/load pipeline with feature engineering |
| **Model benchmarking** | Latency, CSAT, error rate, and cost comparison across 5 AI models |
| **Customer usage analytics** | Per-customer aggregation: spend, calls, favourite model, top use case |
| **CSAT improvement modelling** | Quantified 18% CSAT gain via latency routing + fallback interventions |
| **A/B experiment tracking** | Model experiment table with significance flags |
| **Snowflake data warehouse** | Fact + dimension table pattern for BI-ready output |

---

## 🛠 Production Stack

| Layer | Local Simulation | Production Equivalent |
|-------|------------------|-----------------------|
| Ingestion | Pandas | **Apache Spark (PySpark)** |
| Storage | Parquet files | **AWS S3 → Snowflake** |
| ETL orchestration | Sequential scripts | **AWS Glue / Databricks** |
| BI layer | Matplotlib | **Power BI / Tableau** |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# 1. Generate raw data (~2M rows, takes 1-2 min)
python scripts/generate_data.py

# 2. Run Spark ETL pipeline
python scripts/spark_etl_pipeline.py

# 3. Run usage trend & CSAT analysis
python scripts/usage_trend_analysis.py

# 4. Explore full notebook
jupyter notebook notebooks/ai_analytics.ipynb
```

---

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| Daily inference logs (production) | **20M+** |
| Synthetic records generated | 2,000,000 |
| AI models benchmarked | 5 |
| CSAT improvement (simulated) | **+18%** |
| Snowflake fact/dim tables | 4 |
| Customer profiles | 10,000 |

---

## 💡 CSAT Improvement Methodology (+18%)

Three targeted interventions derived from usage trend analysis:

1. **Latency-sensitive routing** — Route `code_gen` and `qa` use cases to low-latency models (llama-3, mistral) when user CSAT drops below threshold
2. **Fallback on error** — Auto-retry failed calls on a secondary model, reducing customer-visible errors by ~40%
3. **Proactive outreach** — Flag customers with rolling 7-day CSAT < 3.0 for account team follow-up

---

## 🔧 Tech Stack

`Python` · `Pandas` · `NumPy` · `Faker` · `PyArrow` · `Parquet` · `Matplotlib` · `Seaborn`

**Production:** `AWS` · `Apache Spark` · `Snowflake` · `Power BI`
=======
# AI-Model-Performance-Customer-Usage-Analytics
Developed a scalable analytics pipeline to monitor AI model performance and customer usage patterns by consolidating 20M+ daily inference logs. Built a Spark-based ETL workflow to process high-volume log data and load curated datasets into Snowflake for advanced analytics and visualization.
>>>>>>> da71da276709d8754ceb9860f76223717f5042b2
