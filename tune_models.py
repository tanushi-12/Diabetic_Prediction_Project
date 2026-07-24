"""
Hyperparameter tuning for the 5 sklearn-compatible models using
RandomizedSearchCV, scored on F1 (weighted)
 
"""
 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
 
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessing
from src.config import RANDOM_STATE
 
SCORING = "f1_weighted"
CV_FOLDS = 3
N_ITER = 10
 
 
def tune(name, model, param_dist, X, y, n_iter=N_ITER):
    print(f"\n{'='*70}\nTuning {name}...\n{'='*70}")
 
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=CV_FOLDS,
        scoring=SCORING,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    search.fit(X, y)
 
    print(f"\nBest {SCORING} for {name}: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
 
    return search.best_params_, search.best_score_
 
 
def main():
 
    print("Loading and splitting data...")
    loader = DataLoader()
    df = loader.load_dataset()
    df = loader.remove_duplicates()
 
    preprocess = DataPreprocessing()
    X_train, X_test, y_train, y_test = preprocess.split(df)
    X_train_scaled, X_test_scaled = preprocess.scale(X_train, X_test)
 
    results = {}
 
    #  Logistic Regression 
   
    results["Logistic Regression"] = tune(
        "Logistic Regression",
        LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE),
        {
            "C": [0.01, 0.1, 1, 10, 100],
        },
        X_train_scaled, y_train, n_iter=5
    )
 
    # Random Forest 
    results["Random Forest"] = tune(
        "Random Forest",
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        X_train, y_train
    )
 
    #  XGBoost 
  
    results["XGBoost"] = tune(
        "XGBoost",
        XGBClassifier(
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", random_state=RANDOM_STATE
        ),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
        },
        X_train, y_train
    )
 
    # LightGBM
    results["LightGBM"] = tune(
        "LightGBM",
        LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE, verbose=-1),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 8, -1],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
        },
        X_train, y_train
    )
 
    #  CatBoost 
    results["CatBoost"] = tune(
        "CatBoost",
        CatBoostClassifier(auto_class_weights="Balanced", random_state=RANDOM_STATE, verbose=False),
        {
            "iterations": [100, 200, 300],
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
        },
        X_train, y_train
    )
 
    print("\n\n" + "=" * 70)
    print("TUNING RESULTS SUMMARY -- copy these into src/config.py")
    print("=" * 70)
    for name, (params, score) in results.items():
        print(f"\n{name}  (best {SCORING}: {score:.4f})")
        for k, v in params.items():
            print(f"    {k}: {v!r}")
 
 
if __name__ == "__main__":
    main()
