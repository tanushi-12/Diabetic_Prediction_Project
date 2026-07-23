
import os
 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
 
# --------------------------------------------------
 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 
DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "diabetes_012_health_indicators_BRFSS2015.csv"
)
 
MODEL_DIR = os.path.join(BASE_DIR, "models")
 
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
 
TARGET_COLUMN = "Diabetes_012"
 
TEST_SIZE = 0.20
 
RANDOM_STATE = 42
 
# Which models require scaled input (distance/gradient-based models)
NEEDS_SCALING = {"Logistic Regression", "SVM"}
 
# Which models require sample_weight passed at fit() time instead of
# a class_weight constructor arg (XGBoost has no native class_weight)
NEEDS_SAMPLE_WEIGHT = {"XGBoost"}
 
# --------------------------------------------------
 
MODELS = {
 
    # Tuned via RandomizedSearchCV (see tune_models.py), scored on f1_weighted.
    # Best f1_weighted found during tuning: 0.6981
    "Logistic Regression":
        LogisticRegression(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            C=0.01
        ),
 
    # Best f1_weighted found during tuning: 0.7838
    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        ),
 
    "SVM":
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
 
    # Best f1_weighted found during tuning: 0.7934 -- your best sklearn model
    "XGBoost":
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=1.0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE
        ),
 
    # Best f1_weighted found during tuning: 0.7095
    "LightGBM":
        LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            num_leaves=63,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            verbose=-1
        ),
 
    # Best f1_weighted found during tuning: 0.6982
    "CatBoost":
        CatBoostClassifier(
            iterations=300,
            learning_rate=0.01,
            depth=8,
            random_state=RANDOM_STATE,
            verbose=False,
            auto_class_weights="Balanced"
        )
}