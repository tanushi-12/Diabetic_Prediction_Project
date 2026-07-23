import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
 
from sklearn.preprocessing import label_binarize
 
from src.config import OUTPUT_DIR
 
# ---------------------------------------------------
# Output Directories
# ---------------------------------------------------
 
CONFUSION_DIR = os.path.join(OUTPUT_DIR, "confusion_matrix")
METRIC_DIR = os.path.join(OUTPUT_DIR, "metrics")
COMPARISON_DIR = os.path.join(OUTPUT_DIR, "model_comparison")
ROC_DIR = os.path.join(OUTPUT_DIR, "roc_curve")
 
for d in (CONFUSION_DIR, METRIC_DIR, COMPARISON_DIR, ROC_DIR):
    os.makedirs(d, exist_ok=True)
 
CLASS_LABELS = ["Healthy", "Prediabetic", "Diabetic"]
CLASS_VALUES = [0, 1, 2]
 
# The metric used to rank models / choose the "best" one.
# Accuracy is intentionally NOT used here because the target classes
# are imbalanced (far more "Healthy" than "Prediabetic"/"Diabetic"),
# so accuracy alone rewards a model for defaulting to the majority class.
RANKING_METRIC = "F1 Score"
 
 
class Evaluator:
 
    def __init__(self):
        self.results = []
 
    @staticmethod
    def _safe_name(name):
        return name.replace(" ", "_")
 
    def evaluate(self, model_name, model, X_test, y_test):
 
        print("\n" + "=" * 70)
        print(f" Evaluating {model_name}")
        print("=" * 70)
 
        safe_name = self._safe_name(model_name)
 
        # ----------------------------
        # Predictions
        # ----------------------------
 
        y_pred = model.predict(X_test)
 
        y_prob = None
 
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
 
        # ----------------------------
        # Metrics (weighted = accounts for class imbalance,
        # macro = treats every class as equally important)
        # ----------------------------
 
        accuracy = accuracy_score(y_test, y_pred)
 
        precision_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall_w = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
 
        precision_m = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall_m = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)
 
        # ----------------------------
        # ROC AUC + ROC Curve
        # ----------------------------
 
        roc_auc = None
        y_test_bin = label_binarize(y_test, classes=CLASS_VALUES)
 
        if y_prob is not None:
 
            roc_auc = roc_auc_score(
                y_test_bin,
                y_prob,
                average="weighted",
                multi_class="ovr"
            )
 
            # One-vs-rest ROC curve, one line per class
            plt.figure(figsize=(7, 6))
 
            for i, label in enumerate(CLASS_LABELS):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc_score(y_test_bin[:, i], y_prob[:, i]):.3f})")
 
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{model_name} — ROC Curve (One-vs-Rest)")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(ROC_DIR, f"{safe_name}.png"), dpi=300)
            plt.close()
 
        # ----------------------------
        # Print Report
        # ----------------------------
 
        print(classification_report(y_test, y_pred, target_names=CLASS_LABELS, zero_division=0))
 
        # ----------------------------
        # Confusion Matrix
        # ----------------------------
 
        cm = confusion_matrix(y_test, y_pred)
 
        fig, ax = plt.subplots(figsize=(6, 5))
 
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=CLASS_LABELS
        )
 
        disp.plot(cmap="Blues", colorbar=False, ax=ax)
 
        plt.title(f"{model_name} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(CONFUSION_DIR, f"{safe_name}.png"), dpi=300)
        plt.close()
 
        # ----------------------------
        # Save Metrics
        # ----------------------------
 
        self.results.append({
            "Model": model_name,
            "Accuracy": round(accuracy, 4),
            "Precision (weighted)": round(precision_w, 4),
            "Recall (weighted)": round(recall_w, 4),
            "F1 Score": round(f1_w, 4),
            "Precision (macro)": round(precision_m, 4),
            "Recall (macro)": round(recall_m, 4),
            "F1 Score (macro)": round(f1_m, 4),
            "ROC AUC": round(roc_auc, 4) if roc_auc is not None else None
        })
 
    # -------------------------------------------------
 
    def save_results(self):
 
        results_df = pd.DataFrame(self.results)
 
        # Ranked by F1 (weighted), NOT accuracy — see RANKING_METRIC note above
        results_df = results_df.sort_values(by=RANKING_METRIC, ascending=False).reset_index(drop=True)
 
        csv_path = os.path.join(METRIC_DIR, "model_metrics.csv")
        results_df.to_csv(csv_path, index=False)
 
        print(f"\nMetrics saved to:\n{csv_path}")
        print(f"🏆 Best model by {RANKING_METRIC}: {results_df.iloc[0]['Model']}")
 
        # ------------------------------------------
        # Comparison Graphs
        # ------------------------------------------
 
        metrics = [
            "Accuracy",
            "Precision (weighted)",
            "Recall (weighted)",
            "F1 Score",
            "ROC AUC"
        ]
 
        for metric in metrics:
 
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
            plt.savefig(os.path.join(COMPARISON_DIR, f"{safe_metric}.png"), dpi=300)
            plt.close()
 
        print(" Model comparison graphs generated successfully.")
 
        return results_df