import matplotlib
matplotlib.use("Agg")  # Non-interactive backend -- avoids a Windows-specific
                        # Tkinter threading crash that happens when matplotlib's
                        # default GUI backend clashes with sklearn's n_jobs=-1
                        # parallel processing. Safe here since every plot in
                        # this pipeline is saved straight to file anyway.
 
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessing
from src.visualization import Visualizer
from src.train_models import ModelTrainer
from src.explainability import plot_feature_importance, generate_shap_plots
 
 
def main():
 
    loader = DataLoader()
 
    df = loader.load_dataset()
 
    loader.dataset_summary()
 
    df = loader.remove_duplicates()
 
    visual = Visualizer()
 
    visual.diabetes_distribution(df)
    visual.correlation(df)
    visual.bmi_distribution(df)
    visual.age_distribution(df)
    visual.blood_pressure(df)
 
    preprocess = DataPreprocessing()
 
    # Split
    X_train, X_test, y_train, y_test = preprocess.split(df)
 
    # SMOTE -- oversample minority classes (esp. Prediabetic, only ~1.8% of
    # data) in the TRAINING set only. Test set stays real, untouched data.
    X_train_res, y_train_res = preprocess.apply_smote(X_train, y_train, method="borderline")
 
    # Scale (fit on the SMOTE-resampled training set, since that's what
    # Logistic Regression/SVM will actually train on)
    X_train_scaled, X_test_scaled = preprocess.scale(X_train_res, X_test)
 
    # Train Models
    # include_svm=False -- SVM disabled for now (SVC(probability=True) is
    # too slow on this dataset's full training size). Flip to True later
    # once you're ready to include it in the final comparison run.
    trainer = ModelTrainer(include_svm=False)
 
    results_df = trainer.train_all(
        X_train_res,
        X_train_scaled,
        y_train_res,
        X_test,
        X_test_scaled,
        y_test,
        preprocess.scaler
    )
 
    print("\n")
    print(results_df)
 
    # ------------------------------------------
    # Explainability for the top-ranked model
    # ------------------------------------------
 
    best_model_name = results_df.iloc[0]["Model"]
    print(f"\n Generating explainability for best model: {best_model_name}")
 
    best_model = trainer.models[best_model_name]
 
    feature_names = X_train.columns.tolist()
 
    plot_feature_importance(best_model_name, best_model, feature_names)
 
    background = X_train_scaled if best_model_name in ("Logistic Regression", "SVM") else X_train
    sample = X_test_scaled if best_model_name in ("Logistic Regression", "SVM") else X_test
 
    generate_shap_plots(
        best_model_name,
        best_model,
        background,
        sample.iloc[:200]  # subsample for speed
    )
 
    print("\n")
    print("=" * 70)
    print(" FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()