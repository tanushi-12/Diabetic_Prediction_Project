"""
Generates confusion matrix + ROC curve plots for FT-Transformer and
TabTransformer, matching the style/location used for your other 5 models.
 """
 
import os
 
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
 
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessing
from src.config import OUTPUT_DIR
from src.transformer_utils import (
    TRANSFORMER_MODELS, transformer_available, load_transformer_model, predict_with_transformer
)
 
CLASS_LABELS = ["Healthy", "Prediabetic", "Diabetic"]
CLASS_VALUES = [0, 1, 2]
TARGET_COLUMN = "Diabetes_012"
 
CONFUSION_DIR = os.path.join(OUTPUT_DIR, "confusion_matrix")
ROC_DIR = os.path.join(OUTPUT_DIR, "roc_curve")
os.makedirs(CONFUSION_DIR, exist_ok=True)
os.makedirs(ROC_DIR, exist_ok=True)
 
 
def main():
 
    print("Loading and splitting data (same split used for training)...")
    loader = DataLoader()
    df = loader.load_dataset()
    df = loader.remove_duplicates()
 
    preprocess = DataPreprocessing()
    X_train, X_test, y_train, y_test = preprocess.split(df)
 
    y_test_bin = label_binarize(y_test, classes=CLASS_VALUES)
 
    for model_name in TRANSFORMER_MODELS.keys():
 
        if not transformer_available(model_name):
            print(f"Skipping {model_name} -- no checkpoint found in models/")
            continue
 
        print(f"\n{'='*60}\nGenerating plots for {model_name}\n{'='*60}")
 
        model = load_transformer_model(model_name)
        preds, probas = predict_with_transformer(model, X_test, TARGET_COLUMN)
 
        safe_name = model_name.replace(" ", "_")
 
        # Confusion Matrix 
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)
        disp.plot(cmap="Blues", colorbar=False, ax=ax)
        plt.title(f"{model_name} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(CONFUSION_DIR, f"{safe_name}.png"), dpi=300)
        plt.close()
        print(f"Saved confusion matrix: {safe_name}.png")
 
        #  ROC Curve (one-vs-rest) 
        plt.figure(figsize=(7, 6))
        for i, label in enumerate(CLASS_LABELS):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], probas[:, i])
            auc = roc_auc_score(y_test_bin[:, i], probas[:, i])
            plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} -- ROC Curve (One-vs-Rest)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(ROC_DIR, f"{safe_name}.png"), dpi=300)
        plt.close()
        print(f"Saved ROC curve: {safe_name}.png")
 
    print("\nDone. Restart Streamlit to see the new plots on the Model Comparison page.")
 
 
if __name__ == "__main__":
    main()
