import pandas as pd
from itertools import product

def analyze_combinations(df, feature_columns):
    # 1. Generate the complete truth table 
    truth_table = pd.DataFrame(
        product([0.0, 1.0], repeat=len(feature_columns)),
        columns=feature_columns
    )

    # 2. Count combinations that actually occurred in  Cortex data
    observed = (
        df.groupby(feature_columns, dropna=False)
          .size()
          .reset_index(name="occurrence_count")
    )

    # 3. Match the observed data to the truth table
    results = truth_table.merge(
        observed,
        on=feature_columns,
        how="left"
    )

    # Clean up the results
    results["occurrence_count"] = results["occurrence_count"].fillna(0).astype(int)
    results["ever_occurred"] = results["occurrence_count"] > 0

    return results

file_path = "btc_cortex_features.csv"
df = pd.read_csv(file_path)

# # --- NEW: Rename the column to match the Prometheus label ---
# new_metric_name = "_weighted24h48h7d"
# df = df.rename(columns={"poly_transition": new_metric_name})

# --- Update the features list to use the new column name ---
features = ["7d_derivative", "7d_derivative_prime","poly_transition(_weighted24h48h7d)"]
combination_report = analyze_combinations(df, features)
print(combination_report)