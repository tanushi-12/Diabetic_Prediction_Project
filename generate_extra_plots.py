"""
Generates a risk score distribution histogram for every locally-available
model (i.e. every model with a .pkl file -- the two transformer checkpoints
are excluded, since they can't reliably run local batch inference the
same way).
 
Usage:
    python generate_extra_plots.py
"""
 
import os
 
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
 
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessing
from src.config import OUTPUT_DIR, MODEL_DIR, NEEDS_SCALING
from src.save_models import ModelSaver
 
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
 
CLASS_LABELS = {0: "Healthy", 1: "Prediabetic", 2: "Diabetic"}
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics", "model_metrics.csv")
 
 
def compute_risk_score(proba_row):
    """Same weighted formula used in the Streamlit dashboard."""
    _, pre_p, diab_p = proba_row[0], proba_row[1], proba_row[2]
    return min((pre_p * 50) + (diab_p * 100), 100)
 
 
def get_all_local_models():
    """Every model in model_metrics.csv that has a .pkl file (excludes the
    two transformer checkpoints, which aren't run through this script)."""
    results_df = pd.read_csv(METRICS_PATH).sort_values("F1 Score", ascending=False)
 
    available = []
    for _, row in results_df.iterrows():
        model_name = row["Model"]
        pkl_path = os.path.join(MODEL_DIR, f"{model_name.replace(' ', '_')}.pkl")
        if os.path.exists(pkl_path):
            available.append(model_name)
 
    if not available:
        raise RuntimeError("No locally-available (.pkl) models found in model_metrics.csv")
 
    return available
 
 
def main():
 
    print("Loading and splitting data (same split used for training)...")
    loader = DataLoader()
    df = loader.load_dataset()
    df = loader.remove_duplicates()
 
    preprocess = DataPreprocessing()
    X_train, X_test, y_train, y_test = preprocess.split(df)
 
    model_names = get_all_local_models()
    print(f"Generating risk score distributions for: {model_names}")
 
    scaler = None
    if any(name in NEEDS_SCALING for name in model_names):
        scaler = ModelSaver.load_scaler()
 
    # ---------------- Plot 1: Risk Score Distribution per model ----------------
 
    for model_name in model_names:
 
        print(f"\nProcessing {model_name}...")
        model = ModelSaver.load(model_name)
 
        if model_name in NEEDS_SCALING:
            X_input = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        else:
            X_input = X_test
 
        probas = model.predict_proba(X_input)
        risk_scores = np.array([compute_risk_score(row) for row in probas])
 
        plot_df = pd.DataFrame({
            "Risk Score": risk_scores,
            "True Class": [CLASS_LABELS[c] for c in y_test.values]
        })
 
        plt.figure(figsize=(9, 5))
        sns.histplot(data=plot_df, x="Risk Score", hue="True Class", bins=30, multiple="stack", palette="viridis")
        plt.title(f"Risk Score Distribution on Test Set ({model_name})")
        plt.xlabel("Risk Score (0-100)")
        plt.ylabel("Number of Patients")
        plt.tight_layout()
 
        safe_name = model_name.replace(" ", "_")
        out_path = os.path.join(PLOTS_DIR, f"risk_score_distribution_{safe_name}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")
 
    print(f"\nDone. Generated {len(model_names)} risk score distributions.")
 
 
if __name__ == "__main__":
    main()