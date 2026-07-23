import os
import joblib
 
from src.config import MODEL_DIR
 
os.makedirs(MODEL_DIR, exist_ok=True)
 
 
class ModelSaver:
 
    @staticmethod
    def save(model, model_name):
 
        filename = f"{model_name.replace(' ', '_')}.pkl"
 
        filepath = os.path.join(MODEL_DIR, filename)
 
        joblib.dump(model, filepath)
 
        print(f" {model_name} saved successfully.")
 
    @staticmethod
    def save_scaler(scaler, name="scaler"):
 
        filepath = os.path.join(MODEL_DIR, f"{name}.pkl")
 
        joblib.dump(scaler, filepath)
 
        print(" Scaler saved successfully.")
 
    @staticmethod
    def load(model_name):
 
        filename = f"{model_name.replace(' ', '_')}.pkl"
 
        filepath = os.path.join(MODEL_DIR, filename)
 
        return joblib.load(filepath)
 
    @staticmethod
    def load_scaler(name="scaler"):
 
        filepath = os.path.join(MODEL_DIR, f"{name}.pkl")
 
        return joblib.load(filepath)