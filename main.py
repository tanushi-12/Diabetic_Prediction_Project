import matplotlib
matplotlib.use("Agg")  
 
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
 
    # SMOTE -in the TRAINING set only
    X_train_res, y_train_res = preprocess.apply_smote(X_train, y_train, method="borderline")
 
  
    X_train_scaled, X_test_scaled = preprocess.scale(X_train_res, X_test)
 
    # Train Models
 
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
 
    # Explainability for the top-ranked model

 
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
        sample.iloc[:200]  
    )
 
    print("\n")
    print("=" * 70)
    print(" FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()
