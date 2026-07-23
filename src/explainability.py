import os
 
import matplotlib.pyplot as plt
import shap
 
from src.config import OUTPUT_DIR
 
SHAP_DIR = os.path.join(OUTPUT_DIR, "shap")
FI_DIR = os.path.join(OUTPUT_DIR, "feature_importance")
 
os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(FI_DIR, exist_ok=True)
 
TREE_MODELS = {"Random Forest", "XGBoost", "LightGBM", "CatBoost"}
 
 
def _safe_name(name):
    return name.replace(" ", "_")
 
 
def plot_feature_importance(model_name, model, feature_names, top_n=15):
    """Built-in feature_importances_ plot -- only meaningful for tree models."""
 
    if not hasattr(model, "feature_importances_"):
        print(f"  {model_name} has no feature_importances_, skipping.")
        return
 
    importances = model.feature_importances_
 
    order = importances.argsort()[::-1][:top_n]
 
    plt.figure(figsize=(9, 6))
    plt.barh(
        [feature_names[i] for i in order][::-1],
        importances[order][::-1],
        color="teal"
    )
    plt.title(f"{model_name} — Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(FI_DIR, f"{_safe_name(model_name)}.png"), dpi=300)
    plt.close()
 
    print(f" Feature importance saved for {model_name}")
 
 
def _get_explainer(model_name, model, X_background):
    """Pick the right SHAP explainer for the model type."""
 
    if model_name in TREE_MODELS:
        return shap.TreeExplainer(model)
 
    if model_name == "Logistic Regression":
        return shap.LinearExplainer(model, X_background)
 
    # Fallback for anything else (e.g. SVM) -- slower but model-agnostic
    return shap.KernelExplainer(model.predict_proba, shap.sample(X_background, 100))
 
 
def generate_shap_plots(model_name, model, X_background, X_sample, class_index=2, sample_row=0):
    """
    Generates a SHAP summary plot (global) and a waterfall plot (single
    prediction). class_index=2 -> "Diabetic" class by default, since that's
    usually the most clinically important class to explain.
    """
 
    print(f"\n Generating SHAP explanations for {model_name}...")
 
    explainer = _get_explainer(model_name, model, X_background)
    shap_values = explainer.shap_values(X_sample)
 
    # Multiclass models return a list of arrays (one per class) OR a single
    # 3D array depending on the SHAP version / model type -- normalize both.
    if isinstance(shap_values, list):
        class_shap_values = shap_values[class_index]
    elif shap_values.ndim == 3:
        class_shap_values = shap_values[:, :, class_index]
    else:
        class_shap_values = shap_values
 
    safe_name = _safe_name(model_name)
 
    # ---- Summary plot (global feature impact) ----
    plt.figure()
    shap.summary_plot(class_shap_values, X_sample, show=False)
    plt.title(f"{model_name} — SHAP Summary (Diabetic class)")
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, f"{safe_name}_summary.png"), dpi=300, bbox_inches="tight")
    plt.close()
 
    # ---- Waterfall plot (single prediction explanation) ----
    try:
        expected_value = explainer.expected_value
        if hasattr(expected_value, "__len__") and len(expected_value) > 1:
            expected_value = expected_value[class_index]
 
        explanation = shap.Explanation(
            values=class_shap_values[sample_row],
            base_values=expected_value,
            data=X_sample.iloc[sample_row],
            feature_names=X_sample.columns.tolist()
        )
 
        plt.figure()
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(SHAP_DIR, f"{safe_name}_waterfall.png"), dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  Waterfall plot skipped for {model_name}: {e}")
 
    print(f" SHAP plots saved for {model_name}")
 
 
def get_top_risk_factors(model_name, model, X_background, X_instance, top_n=4, class_index=2):
    """
    Returns the top_n features driving a single prediction toward the
    given class (default: Diabetic), as a list of (feature_name, shap_value).
    Used by the Streamlit dashboard to populate 'Top Risk Factors'.
    """
 
    explainer = _get_explainer(model_name, model, X_background)
    shap_values = explainer.shap_values(X_instance)
 
    if isinstance(shap_values, list):
        row_values = shap_values[class_index][0]
    elif shap_values.ndim == 3:
        row_values = shap_values[0, :, class_index]
    else:
        row_values = shap_values[0]
 
    feature_names = X_instance.columns.tolist()
 
    pairs = list(zip(feature_names, row_values))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
 
    return pairs[:top_n]