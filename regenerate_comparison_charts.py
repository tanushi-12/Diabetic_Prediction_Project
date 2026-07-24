"""
Regenerates the Model Comparison bar charts
"""
 
import os
 
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import seaborn as sns
 
from src.config import OUTPUT_DIR
 
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics", "model_metrics.csv")
COMPARISON_DIR = os.path.join(OUTPUT_DIR, "model_comparison")
os.makedirs(COMPARISON_DIR, exist_ok=True)
 
 
def main():
 
    if not os.path.exists(METRICS_PATH):
        print(f"No metrics file found at {METRICS_PATH}. Run main.py first.")
        return
 
    results_df = pd.read_csv(METRICS_PATH)
    results_df = results_df.sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
 
    print(f"Regenerating charts for {len(results_df)} models: {list(results_df['Model'])}")
 
    metrics = ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 Score", "ROC AUC"]
 
    for metric in metrics:
 
        if metric not in results_df.columns:
            print(f"Skipping '{metric}' -- column not found in CSV.")
            continue
 
        plt.figure(figsize=(9, 5))
 
        sns.barplot(
            data=results_df,
            x="Model",
            y=metric,
            hue="Model",
            palette="viridis",
            legend=False
        )
 
        plt.title(f"{metric} Comparison")
        plt.xticks(rotation=15)
        plt.tight_layout()
 
        safe_metric = metric.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out_path = os.path.join(COMPARISON_DIR, f"{safe_metric}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
 
        print(f"Saved: {out_path}")
 
    print("\nDone. Restart Streamlit to see the updated charts (all 7 models included).")
 
 
if __name__ == "__main__":
    main()
