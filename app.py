import os
import traceback

import pandas as pd
import numpy as np
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score
)
from sklearn.preprocessing import label_binarize

from src.config import MODEL_DIR, OUTPUT_DIR, MODELS, TARGET_COLUMN, NEEDS_SCALING
from src.save_models import ModelSaver
from src.explainability import get_top_risk_factors
from src.field_config import FIELD_OPTIONS, CONTINUOUS_FIELDS, FIELD_HELP
from src.pdf_extractor import extract_fields_from_report
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessing
from src.transformer_utils import (
    TRANSFORMER_MODELS, is_transformer_model, transformer_available,
    load_transformer_model, predict_with_transformer
)

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="Di",
    layout="wide"
)

CLASS_LABELS = {0: "Healthy", 1: "Prediabetic", 2: "Diabetic"}
CLASS_TONE = {0: "healthy", 1: "prediabetic", 2: "diabetic"}

METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics", "model_metrics.csv")

# ------------------------------------------------------------
# Design tokens (pastel clinical palette)
# ------------------------------------------------------------

TONE_COLORS = {
    "healthy":     {"bg": "#DCEEE1", "fg": "#2F6B4F"},
    "prediabetic": {"bg": "#FBEED9", "fg": "#92651B"},
    "diabetic":    {"bg": "#F8DCE0", "fg": "#9B3B4A"},
}

GAUGE_GRADIENT = (
    "linear-gradient(to right, "
    "#DCEEE1 0%, #DCEEE1 33%, "
    "#FBEED9 33%, #FBEED9 66%, "
    "#F8DCE0 66%, #F8DCE0 100%)"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Base page background & text (forced, not reliant on theme cache) ---- */
.stApp {
    background-color: #F3F7F6;
}

[data-testid="stAppViewContainer"] {
    background-color: #F3F7F6;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

h1, h2, h3, h4, h5, h6 {
    color: #1E293B !important;
    font-weight: 700;
    letter-spacing: -0.01em;
}

p, label, .stMarkdown, [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
[data-testid="stCaptionContainer"], span {
    color: #33415C;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background-color: #E7F0EE;
    border-right: 1px solid #D7E4E1;
}

section[data-testid="stSidebar"] * {
    color: #1E293B;
}

.nav-title {
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-size: 0.72rem;
    font-weight: 700;
    color: #5F8F8A;
    margin: 4px 0 18px 4px;
}

section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    border-radius: 10px;
    border: none;
    background-color: transparent;
    color: #33415C;
    font-weight: 600;
    padding: 0.6rem 1rem;
    margin-bottom: 4px;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #D7E9E6;
    color: #1E293B;
}

section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background-color: #6FA8A3;
    color: #FFFFFF;
}

/* ---- Eyebrow labels ---- */
.eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.75rem;
    font-weight: 700;
    color: #6FA8A3;
    margin-bottom: 4px;
}

.subtle {
    color: #5B6B82;
    font-size: 0.95rem;
}

hr {
    border: none;
    border-top: 1px solid #DDE6E3;
    margin: 1.5rem 0;
}

/* ---- Card containers (st.container(border=True)) ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border-radius: 14px;
    border: 1px solid #E3E8EE;
    padding: 1.1rem 1.4rem;
    box-shadow: 0 1px 4px rgba(30, 41, 59, 0.05);
}

/* ---- Inputs ---- */
.stTextInput input, .stNumberInput input, [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border-radius: 8px !important;
    border: 1px solid #D7DEE5 !important;
    color: #1E293B !important;
}

/* ---- File uploader (PDF/CSV drag-and-drop area) ---- */
[data-testid="stFileUploaderDropzone"],
.stFileUploader section,
.stFileUploader div[data-testid] {
    background-color: #F7FAF9 !important;
    border: 1px dashed #B9CFCB !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploaderDropzone"] *,
.stFileUploader section * {
    color: #33415C !important;
}

[data-testid="stFileUploaderDropzone"] button,
.stFileUploader button {
    background-color: #FFFFFF !important;
    border: 1px solid #D7DEE5 !important;
    color: #33415C !important;
    border-radius: 8px !important;
}

[data-testid="stFileUploaderDropzone"] svg,
.stFileUploader svg {
    fill: #6FA8A3 !important;
}

/* Uploaded file "chip" shown after upload */
[data-testid="stFileUploaderFile"] {
    background-color: #FFFFFF !important;
    color: #33415C !important;
    border: 1px solid #E3E8EE !important;
    border-radius: 8px !important;
}

/* ---- Dropdown menu popover (renders via portal, needs its own rules) ---- */
[data-baseweb="popover"] [data-baseweb="menu"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E3E8EE !important;
    border-radius: 8px !important;
}

ul[role="listbox"] {
    background-color: #FFFFFF !important;
}

li[role="option"] {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
}

li[role="option"]:hover,
li[aria-selected="true"] {
    background-color: #E7F0EE !important;
    color: #1E293B !important;
}

/* ---- Number input +/- stepper buttons ---- */
button[data-testid="stNumberInputStepUp"],
button[data-testid="stNumberInputStepDown"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D7DEE5 !important;
    color: #33415C !important;
}

button[data-testid="stNumberInputStepUp"]:hover,
button[data-testid="stNumberInputStepDown"]:hover {
    background-color: #E7F0EE !important;
}

/* ---- Buttons (main content) ---- */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
}

.main .stButton > button[kind="primary"] {
    background-color: #6FA8A3;
    border-color: #6FA8A3;
    color: #FFFFFF;
}

/* ---- DataFrame ---- */
.stDataFrame {
    border: 1px solid #E3E8EE;
    border-radius: 10px;
    overflow: hidden;
}

/* ---- st.table (plain HTML table, used for the metrics comparison) ---- */
.stTable table {
    background-color: #FFFFFF;
    border-collapse: collapse;
    width: 100%;
}

.stTable thead th {
    background-color: #E7F0EE !important;
    color: #1E293B !important;
    font-weight: 700 !important;
    text-align: center !important;
    padding: 10px 14px !important;
    border-bottom: 2px solid #D7E4E1 !important;
}

.stTable tbody td {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    padding: 9px 14px !important;
    text-align: center !important;
    border-bottom: 1px solid #EEF2F5 !important;
}

.stTable tbody tr:nth-child(even) td {
    background-color: #F7FAF9 !important;
}

.stTable tbody th {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    font-weight: 600 !important;
    padding: 9px 14px !important;
    text-align: left !important;
    border-bottom: 1px solid #EEF2F5 !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------
# Small UI helpers
# ------------------------------------------------------------

def eyebrow(text):
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def render_badge(label, tone):
    colors = TONE_COLORS[tone]
    st.markdown(
        f'<span style="background:{colors["bg"]};color:{colors["fg"]};'
        f'padding:5px 14px;border-radius:999px;font-weight:600;'
        f'font-size:0.9rem;display:inline-block;">{label}</span>',
        unsafe_allow_html=True
    )


def render_risk_gauge(score):
    score = max(0, min(100, score))
    st.markdown(
        f"""
        <div style="position:relative; width:100%; height:30px; margin-top:6px;">
            <div style="position:absolute; top:8px; left:0; right:0; height:14px;
                        border-radius:8px; overflow:hidden; background:{GAUGE_GRADIENT};
                        border:1px solid #E3E8EE;"></div>
            <div style="position:absolute; top:0px; left:calc({score}% - 2px);
                        width:4px; height:30px; background:#2B3648; border-radius:2px;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.8rem;
                    color:#64748B; margin-top:4px;">
            <span>Lower risk</span>
            <span style="font-weight:600; color:#2B3648;">{score} / 100</span>
            <span>Higher risk</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def bullet_list(items):
    html = "<ul style='margin-top:4px; padding-left:1.1rem; color:#2B3648;'>"
    for item in items:
        html += f"<li style='margin-bottom:6px;'>{item}</li>"
    html += "</ul>"
    st.markdown(html, unsafe_allow_html=True)


# ------------------------------------------------------------
# Cached loaders (avoid reloading model/scaler on every interaction)
# ------------------------------------------------------------

@st.cache_resource
def load_model(model_name):
    if is_transformer_model(model_name):
        try:
            return load_transformer_model(model_name)
        except Exception as e:
            st.error(f"Failed to load {model_name} checkpoint: {e}")
            with st.expander("Full technical details"):
                st.code(traceback.format_exc())
            return None
    try:
        return ModelSaver.load(model_name)
    except FileNotFoundError:
        return None


@st.cache_resource
def load_scaler():
    try:
        return ModelSaver.load_scaler()
    except FileNotFoundError:
        return None


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        return pd.read_csv(METRICS_PATH)
    return None


@st.cache_data
def load_test_set():
    """Recreates the exact same train/test split used during training,
    so classification reports are computed on real, held-out data."""
    loader = DataLoader()
    df = loader.load_dataset()
    df = loader.remove_duplicates()

    preprocess = DataPreprocessing()
    X_train, X_test, y_train, y_test = preprocess.split(df)
    return X_test, y_test


@st.cache_data
def compute_classification_report(model_name):
    """Runs the given model against the real test set and returns a
    (report_dataframe, accuracy) tuple, matching the console output
    format used during training."""

    X_test, y_test = load_test_set()
    model = load_model(model_name)

    if model is None:
        return None, None

    if model_name in NEEDS_SCALING:
        scaler = load_scaler()
        X_input = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    else:
        X_input = X_test

    preds = model.predict(X_input)
    preds = np.array(preds).reshape(-1)  # CatBoost returns shape (n,1)

    report_dict = classification_report(
        y_test, preds, target_names=list(CLASS_LABELS.values()), output_dict=True, zero_division=0
    )
    accuracy = report_dict.pop("accuracy")

    report_df = pd.DataFrame(report_dict).transpose().round(3)
    # support column should read as whole numbers, not decimals
    report_df["support"] = report_df["support"].astype(int)

    return report_df, accuracy


def _get_model_predictions(model_name):
    """Shared helper: returns (preds, probas, X_test, y_test) for a local
    sklearn model, run live against the real test set."""
    X_test, y_test = load_test_set()
    model = load_model(model_name)

    if model is None:
        return None, None, None, None

    if model_name in NEEDS_SCALING:
        scaler = load_scaler()
        X_input = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    else:
        X_input = X_test

    preds = model.predict(X_input)
    preds = np.array(preds).reshape(-1)  # CatBoost returns shape (n,1)
    probas = model.predict_proba(X_input) if hasattr(model, "predict_proba") else None

    return preds, probas, X_test, y_test


@st.cache_resource
def compute_confusion_matrix_fig(model_name):
    preds, _, _, y_test = _get_model_predictions(model_name)
    if preds is None:
        return None

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(CLASS_LABELS.values()))
    disp.plot(cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"{model_name} Confusion Matrix")
    fig.tight_layout()
    return fig


@st.cache_resource
def compute_roc_curve_fig(model_name):
    preds, probas, _, y_test = _get_model_predictions(model_name)
    if probas is None:
        return None

    y_test_bin = label_binarize(y_test, classes=list(CLASS_LABELS.keys()))

    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    for i, label in enumerate(CLASS_LABELS.values()):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], probas[:, i])
        auc = roc_auc_score(y_test_bin[:, i], probas[:, i])
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} -- ROC Curve (One-vs-Rest)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


@st.cache_resource
def compute_risk_score_fig(model_name):
    preds, probas, _, y_test = _get_model_predictions(model_name)
    if probas is None:
        return None

    risk_scores = np.array([compute_risk_score(row) for row in probas])
    plot_df = pd.DataFrame({
        "Risk Score": risk_scores,
        "True Class": [CLASS_LABELS[c] for c in y_test.values]
    })

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    sns.histplot(data=plot_df, x="Risk Score", hue="True Class", bins=30, multiple="stack", palette="viridis", ax=ax)
    ax.set_title(f"Risk Score Distribution on Test Set ({model_name})")
    ax.set_xlabel("Risk Score (0-100)")
    ax.set_ylabel("Number of Patients")
    fig.tight_layout()
    return fig


@st.cache_resource
def compute_bmi_vs_age_fig():
    """Dataset-level scatter, not tied to any specific model."""
    X_test, y_test = load_test_set()

    scatter_df = X_test[["BMI", "Age"]].copy()
    scatter_df["Class"] = [CLASS_LABELS[c] for c in y_test.values]
    scatter_sample = scatter_df.sample(n=min(3000, len(scatter_df)), random_state=42)

    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    sns.scatterplot(
        data=scatter_sample, x="BMI", y="Age", hue="Class",
        palette="viridis", alpha=0.5, s=20, ax=ax
    )
    ax.set_title("BMI vs Age, Colored by Class")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Age Group Code (1=18-24 ... 13=80+)")
    fig.tight_layout()
    return fig


def available_models():
    """All models with saved artifacts (.pkl OR checkpoint folder) --
    used for Model Comparison, where we just display metrics/plots."""
    names = []
    for name in MODELS.keys():
        path = os.path.join(MODEL_DIR, f"{name.replace(' ', '_')}.pkl")
        if os.path.exists(path):
            names.append(name)
    for name in TRANSFORMER_MODELS.keys():
        if transformer_available(name):
            names.append(name)
    return names


def predictable_models():
    """Models usable for LIVE prediction on the Predict page. Excludes the
    transformers -- their checkpoints were trained on Colab's GPU and
    loading them locally on CPU-only machines has proven unreliable across
    pytorch_tabular versions. They're still shown/compared on the Model
    Comparison page, just not offered here."""
    return [name for name in available_models() if not is_transformer_model(name)]


def compute_risk_score(proba):
    """
    Weighted composite risk score (0-100), not just raw confidence.
    Prediabetic probability counts half, Diabetic counts fully -- this
    reflects clinical severity rather than just 'most likely class'.
    """
    _, pre_p, diab_p = proba[0], proba[1], proba[2]
    score = (pre_p * 50) + (diab_p * 100)
    return round(min(score, 100), 1)


def get_recommendations(top_factors):
    """Simple rule-based recommendations keyed off top SHAP features."""
    tips = {
        "BMI": "Aim for gradual weight loss through diet and activity.",
        "HighBP": "Monitor blood pressure regularly and reduce sodium intake.",
        "HighChol": "Get cholesterol checked annually; consider dietary fat reduction.",
        "PhysActivity": "Aim for at least 30 minutes of physical activity most days.",
        "GenHlth": "Discuss your general health trend with a primary care provider.",
        "HeartDiseaseorAttack": "Follow up with a cardiologist given prior heart history.",
        "Smoker": "Consider a smoking cessation program.",
        "Age": "Routine screening becomes more important with age.",
        "Income": "Look into community health screening programs if cost is a barrier.",
        "DiffWalk": "Discuss mobility-friendly exercise options with a physical therapist.",
    }
    default_tip = "Discuss this factor with a healthcare provider for personalized guidance."

    recs = [tips.get(feature_name, default_tip) for feature_name, _ in top_factors]
    recs.append("Get an annual HbA1c test to track blood sugar trends over time.")

    return recs[:5]


# ------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------

st.sidebar.markdown('<div class="nav-title">Diabetes Risk Platform</div>', unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "Predict"

for page_name in ["Predict", "Model Comparison"]:
    is_active = st.session_state.page == page_name
    if st.sidebar.button(
        page_name,
        key=f"nav_{page_name}",
        type="primary" if is_active else "secondary",
        use_container_width=True
    ):
        st.session_state.page = page_name
        st.rerun()

page = st.session_state.page

models_ready = available_models()

if not models_ready:
    st.error(
        "No trained models found in `models/`. Run `python main.py` first "
        "to train and save models before using this dashboard."
    )
    st.stop()


# ==============================================================
# PAGE 1 -- PREDICT
# ==============================================================

if page == "Predict":

    eyebrow("Patient Report")
    st.title("Diabetes Risk Report")
    st.markdown(
        '<div class="subtle">Upload patient data or enter values manually to generate a risk report.</div>',
        unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    selected_model_name = None
    model = None
    scaler = None
    input_df = None

    with st.container(border=True):
        selected_model_name = st.selectbox("Model", predictable_models())

        model = load_model(selected_model_name)
        scaler = load_scaler()

        input_mode = st.radio(
            "Input method",
            ["Upload PDF Report", "Manual entry"],
            horizontal=True
        )

        feature_columns = [c for c in pd.read_csv(
            os.path.join("data", "diabetes_012_health_indicators_BRFSS2015.csv"), nrows=1
        ).columns if c != TARGET_COLUMN]

        def render_entry_form(prefill=None, fields_to_render=None):
            """Renders input widgets only for `fields_to_render` (defaults to
            all fields). Any feature present in `prefill` but NOT in
            fields_to_render is used as-is, with no widget shown for it.
            Returns a single-row DataFrame covering every feature column."""
            prefill = prefill or {}
            fields_to_render = feature_columns if fields_to_render is None else fields_to_render

            values = dict(prefill)  # extracted/known values start pre-populated

            widget_fields = [f for f in feature_columns if f in fields_to_render]

            cols = st.columns(3)
            for i, feature in enumerate(widget_fields):
                with cols[i % 3]:
                    if feature in FIELD_OPTIONS:
                        options = FIELD_OPTIONS[feature]
                        option_labels = list(options.values())
                        option_codes = list(options.keys())
                        default_index = option_codes.index(prefill[feature]) if feature in prefill else 0
                        label = st.selectbox(
                            feature, option_labels, index=default_index,
                            help=FIELD_HELP.get(feature), key=f"field_{feature}"
                        )
                        code = [k for k, v in options.items() if v == label][0]
                        values[feature] = code
                    elif feature in CONTINUOUS_FIELDS:
                        lo, hi, default, step, help_text = CONTINUOUS_FIELDS[feature]
                        default_value = prefill.get(feature, default)
                        values[feature] = st.number_input(
                            feature, min_value=lo, max_value=hi, value=default_value,
                            step=step, help=help_text, key=f"field_{feature}"
                        )
                    else:
                        default_value = prefill.get(feature, 0.0)
                        values[feature] = st.number_input(
                            feature, value=default_value, step=1.0, key=f"field_{feature}"
                        )

            return pd.DataFrame([values])[feature_columns]

        if input_mode == "Upload PDF Report":

            st.caption(
                "Extraction works best on reports with clearly labeled fields "
                "(e.g. \"BMI: 27\", \"Blood Pressure: 130/85\"). Only fields it "
                "can't find will show up below for you to fill in -- always "
                "review before generating the report."
            )

            uploaded_pdf = st.file_uploader("Upload a health/lab report PDF", type=["pdf"])

            if uploaded_pdf is not None:
                try:
                    extracted, found_fields, missing_fields = extract_fields_from_report(uploaded_pdf)

                    if found_fields:
                        st.success(f"Extracted from PDF: {', '.join(found_fields)}")

                    if missing_fields:
                        st.info(f"Please fill in (not found in PDF): {', '.join(missing_fields)}")
                        input_df = render_entry_form(prefill=extracted, fields_to_render=missing_fields)
                    else:
                        st.info("All fields were extracted from the PDF -- nothing left to fill in.")
                        input_df = render_entry_form(prefill=extracted, fields_to_render=[])

                except Exception as e:
                    st.error(f"Could not read this PDF: {e}")

        else:
            st.write("Enter patient values:")
            input_df = render_entry_form()

        generate_clicked = st.button("Generate Report", type="primary")

    if input_df is not None and generate_clicked:

        if model is None:
            st.stop()

        X = input_df.copy()

        if is_transformer_model(selected_model_name):
            # pytorch_tabular models were trained on raw, unscaled features
            # and return a DataFrame, not a plain array -- normalize it here.
            X_model_input = X
            preds, probas = predict_with_transformer(model, X_model_input, TARGET_COLUMN)
        else:
            if selected_model_name in NEEDS_SCALING and scaler is not None:
                X_model_input = pd.DataFrame(
                    scaler.transform(X), columns=X.columns, index=X.index
                )
            else:
                X_model_input = X

            preds = model.predict(X_model_input)
            preds = np.array(preds).reshape(-1)  # CatBoost returns shape (n,1) instead of (n,)
            probas = model.predict_proba(X_model_input) if hasattr(model, "predict_proba") else None

        for row_idx in range(len(X)):

            pred_class = int(preds[row_idx])
            proba_row = probas[row_idx] if probas is not None else None

            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

            with st.container(border=True):
                eyebrow(f"Patient {row_idx + 1}")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric("Model", selected_model_name)

                with c2:
                    st.write("**Prediction**")
                    render_badge(CLASS_LABELS[pred_class], CLASS_TONE[pred_class])

                with c3:
                    confidence = round(proba_row[pred_class] * 100, 1) if proba_row is not None else None
                    st.metric("Confidence", f"{confidence}%" if confidence is not None else "N/A")

                if proba_row is not None:
                    risk_score = compute_risk_score(proba_row)
                    st.write("**Risk Score**")
                    render_risk_gauge(risk_score)

                    col_a, col_b = st.columns(2)

                    if is_transformer_model(selected_model_name):
                        st.info(
                            "Detailed SHAP-based risk factors aren't available for "
                            "deep learning models in this app yet. General guidance:"
                        )
                        bullet_list(get_recommendations([]))
                    else:
                        try:
                            background = X_model_input
                            top_factors = get_top_risk_factors(
                                selected_model_name,
                                model,
                                background,
                                X_model_input.iloc[[row_idx]],
                                top_n=4
                            )

                            with col_a:
                                st.write("**Top Risk Factors**")
                                factor_items = []
                                for feat, val in top_factors:
                                    direction = "increases" if val > 0 else "decreases"
                                    factor_items.append(f"<strong>{feat}</strong> &mdash; {direction} predicted risk")
                                bullet_list(factor_items)

                            with col_b:
                                st.write("**Recommendations**")
                                bullet_list(get_recommendations(top_factors))

                        except Exception as e:
                            st.info(f"Explanation unavailable for this row ({e}).")


# ==============================================================
# PAGE 2 -- MODEL COMPARISON
# ==============================================================

else:

    eyebrow("Evaluation")
    st.title("Model Comparison")

    metrics_df = load_metrics()

    if metrics_df is None:
        st.warning("No metrics found. Run `python main.py` to train models and generate `model_metrics.csv`.")
    else:
        st.markdown(
            '<div class="subtle">Models are ranked by F1 Score, not accuracy. This dataset is '
            'imbalanced (far more Healthy cases than Prediabetic or Diabetic), so accuracy alone '
            'would favor a model that just predicts the majority class.</div>',
            unsafe_allow_html=True
        )
        st.markdown("<hr>", unsafe_allow_html=True)

        best_model = metrics_df.sort_values("F1 Score", ascending=False).iloc[0]["Model"]
        st.success(f"Best model by F1 Score: **{best_model}**", icon=None)

        with st.container(border=True):
            st.table(metrics_df.set_index("Model"))

        st.markdown("### Comparison Charts")
        st.markdown(
            '<div class="subtle">Rendered live from the current metrics table above -- always in sync, '
            'no separate regeneration step needed.</div>',
            unsafe_allow_html=True
        )

        chart_metrics = ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 Score", "ROC AUC"]
        chart_cols = st.columns(2)

        for i, metric in enumerate(chart_metrics):
            if metric not in metrics_df.columns:
                continue
            with chart_cols[i % 2]:
                fig, ax = plt.subplots(figsize=(4.5, 2.9))
                plot_data = metrics_df.sort_values("F1 Score", ascending=False)
                sns.barplot(data=plot_data, x="Model", y=metric, hue="Model", palette="viridis", legend=False, ax=ax)
                ax.set_title(f"{metric} Comparison")
                ax.tick_params(axis="x", rotation=25)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)

        st.markdown("### Confusion Matrices")

        selected_cm_model = st.selectbox("View confusion matrix for", models_ready, key="cm_select")

        if is_transformer_model(selected_cm_model):
            cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix", f"{selected_cm_model.replace(' ', '_')}.png")
            if os.path.exists(cm_path):
                cm_col, _ = st.columns([1, 1])
                with cm_col:
                    st.image(cm_path, width=480)
            else:
                st.info(
                    f"No confusion matrix generated for {selected_cm_model} yet "
                    "(run `colab_generate_transformer_plots.py` in Colab)."
                )
        else:
            with st.spinner(f"Computing confusion matrix for {selected_cm_model}..."):
                fig = compute_confusion_matrix_fig(selected_cm_model)
            if fig is not None:
                cm_col, _ = st.columns([1, 1])
                with cm_col:
                    st.pyplot(fig, use_container_width=False)
            else:
                st.info(f"Could not compute confusion matrix for {selected_cm_model}.")

        st.markdown("### ROC Curves")

        selected_roc_model = st.selectbox("View ROC curve for", models_ready, key="roc_select")

        if is_transformer_model(selected_roc_model):
            roc_path = os.path.join(OUTPUT_DIR, "roc_curve", f"{selected_roc_model.replace(' ', '_')}.png")
            if os.path.exists(roc_path):
                roc_col, _ = st.columns([1, 1])
                with roc_col:
                    st.image(roc_path, width=480)
            else:
                st.info(
                    f"No ROC curve generated for {selected_roc_model} yet "
                    "(run `colab_generate_transformer_plots.py` in Colab)."
                )
        else:
            with st.spinner(f"Computing ROC curve for {selected_roc_model}..."):
                fig = compute_roc_curve_fig(selected_roc_model)
            if fig is not None:
                roc_col, _ = st.columns([1, 1])
                with roc_col:
                    st.pyplot(fig, use_container_width=False)
            else:
                st.info(f"Could not compute ROC curve for {selected_roc_model}.")

        st.markdown("### Risk Score Distribution")
        st.markdown(
            '<div class="subtle">How confidently each model spreads patients across the 0-100 risk '
            'scale, broken down by their true class.</div>',
            unsafe_allow_html=True
        )

        selected_risk_model = st.selectbox("View risk score distribution for", models_ready, key="risk_select")

        if is_transformer_model(selected_risk_model):
            risk_path = os.path.join(
                OUTPUT_DIR, "plots", f"risk_score_distribution_{selected_risk_model.replace(' ', '_')}.png"
            )
            if os.path.exists(risk_path):
                risk_col, _ = st.columns([1, 1])
                with risk_col:
                    st.image(risk_path, width=480)
            else:
                st.info(
                    f"No risk score distribution generated for {selected_risk_model} yet "
                    "(run `colab_generate_transformer_extras.py` in Colab)."
                )
        else:
            with st.spinner(f"Computing risk score distribution for {selected_risk_model}..."):
                fig = compute_risk_score_fig(selected_risk_model)
            if fig is not None:
                risk_col, _ = st.columns([1, 1])
                with risk_col:
                    st.pyplot(fig, use_container_width=False)
            else:
                st.info(f"Could not compute risk score distribution for {selected_risk_model}.")

        st.markdown("### BMI vs Age")
        st.markdown(
            '<div class="subtle">How BMI and age relate across the three classes in the test set.</div>',
            unsafe_allow_html=True
        )

        bmi_fig = compute_bmi_vs_age_fig()
        bmi_col, _ = st.columns([1, 1])
        with bmi_col:
            st.pyplot(bmi_fig, use_container_width=False)

        st.markdown("### Classification Report")
        st.markdown(
            '<div class="subtle">Per-class precision, recall, F1, and support. Computed live for '
            'local models; loaded from a precomputed file (generated in Colab) for the transformers.</div>',
            unsafe_allow_html=True
        )

        selected_cr_model = st.selectbox("View classification report for", models_ready, key="cr_select")

        if is_transformer_model(selected_cr_model):
            cr_path = os.path.join(
                OUTPUT_DIR, "classification_reports", f"{selected_cr_model.replace(' ', '_')}.csv"
            )
            if os.path.exists(cr_path):
                report_df = pd.read_csv(cr_path, index_col=0)
                with st.container(border=True):
                    st.table(report_df)
            else:
                st.info(
                    f"No classification report generated for {selected_cr_model} yet. "
                    "Run `colab_generate_transformer_extras.py` in Colab, then place the "
                    f"downloaded CSV at `outputs/classification_reports/{selected_cr_model.replace(' ', '_')}.csv`."
                )
        else:
            with st.spinner(f"Computing classification report for {selected_cr_model}..."):
                report_df, cr_accuracy = compute_classification_report(selected_cr_model)

            if report_df is not None:
                st.write(f"**Overall Accuracy:** {round(cr_accuracy, 4)}")
                with st.container(border=True):
                    st.table(report_df)
            else:
                st.info(f"Could not compute classification report for {selected_cr_model}.")