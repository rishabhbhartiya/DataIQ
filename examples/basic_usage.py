from MLRadar import MLRadar
import pandas as pd

df = pd.read_csv("employee_cleaned.csv")
diq = MLRadar(df, target="churn")

# Full EDA report
diq.profile("report.html")

# ML Readiness Score
diq.readiness_score()

# Leakage Detective
diq.leakage_report()

# Apply transforms (chainable)
diq.apply("drop_duplicates").apply("impute_median").apply("cap_outliers")

# Before/after comparison
diq.compare("compare.html")

# Drift vs production data
df_prod = pd.read_csv("production.csv")
diq.drift(df_prod, "drift.html")

# Generate pipeline code
diq.export_pipeline_code("pipeline.py")

# Export clean data
diq.export_csv("cleaned.csv")