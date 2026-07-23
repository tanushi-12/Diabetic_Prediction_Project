from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
 
from src.config import MODELS, NEEDS_SCALING, NEEDS_SAMPLE_WEIGHT
from src.evaluate import Evaluator
from src.save_models import ModelSaver
 
# SVC(probability=True) scales ~O(n^2)-O(n^3) with training rows.
# On large datasets (100k+ rows) it can take hours. Cap the training
# sample for SVM specifically so it stays a fair comparison point
# without blocking the rest of the pipeline.
SVM_MAX_TRAIN_ROWS = 15000
 
 
class ModelTrainer:
 
    def __init__(self, include_svm=True):
 
        self.evaluator = Evaluator()
 
        self.models = dict(MODELS)
 
        if not include_svm:
            self.models.pop("SVM", None)
 
    def train_all(
        self,
        X_train,
        X_train_scaled,
        y_train,
        X_test,
        X_test_scaled,
        y_test,
        scaler=None
    ):
 
        print("\n" + "=" * 70)
        print(" TRAINING MODELS")
        print("=" * 70)
 
        for name, model in self.models.items():
 
            print(f"\nTraining {name}...")
 
            use_scaled = name in NEEDS_SCALING
            train_X = X_train_scaled if use_scaled else X_train
            test_X = X_test_scaled if use_scaled else X_test
            train_y = y_train
 
            if name == "SVM" and len(train_X) > SVM_MAX_TRAIN_ROWS:
                print(f"  Subsampling SVM training data: {len(train_X)} -> {SVM_MAX_TRAIN_ROWS} rows (stratified)")
                train_X, _, train_y, _ = train_test_split(
                    train_X, train_y,
                    train_size=SVM_MAX_TRAIN_ROWS,
                    stratify=train_y,
                    random_state=42
                )
 
            if name in NEEDS_SAMPLE_WEIGHT:
                # XGBoost has no class_weight param -- pass balanced
                # sample weights explicitly at fit time instead.
                sample_weights = compute_sample_weight(class_weight="balanced", y=train_y)
                model.fit(train_X, train_y, sample_weight=sample_weights)
            else:
                model.fit(train_X, train_y)
 
            print(f" {name} Training Completed")
 
            self.evaluator.evaluate(name, model, test_X, y_test)
 
            ModelSaver.save(model, name)
 
        if scaler is not None:
            ModelSaver.save_scaler(scaler)
 
        results_df = self.evaluator.save_results()
 
        print("\n" + "=" * 70)
        print(" ALL MODELS TRAINED SUCCESSFULLY")
        print("=" * 70)
 
        return results_df