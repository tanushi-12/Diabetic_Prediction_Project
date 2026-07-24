import pandas as pd
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek
 
from src.config import TEST_SIZE, RANDOM_STATE, TARGET_COLUMN
 
 
class DataPreprocessing:
 
    def __init__(self):
 
        self.scaler = StandardScaler()
 
    def engineer_features(self, df):
       
 
        df = df.copy()
 
        
        df["BMI_Category"] = pd.cut(
            df["BMI"], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]
        ).astype(int)
 
        
        df["UnhealthyDays"] = df["MentHlth"] + df["PhysHlth"]
 
        
        df["BP_Chol_Risk"] = df["HighBP"] * df["HighChol"]
 
        return df
 
    def split(self, df):
 
        df = self.engineer_features(df)
 
        X = df.drop(TARGET_COLUMN, axis=1)
 
        y = df[TARGET_COLUMN]
 
        X_train, X_test, y_train, y_test = train_test_split(
 
            X,
 
            y,
 
            test_size=TEST_SIZE,
 
            random_state=RANDOM_STATE,
 
            stratify=y
 
        )
 
        print("\nDataset Split Completed")
 
        print(f"Training Samples : {len(X_train)}")
 
        print(f"Testing Samples  : {len(X_test)}")
 
        return X_train, X_test, y_train, y_test
 
    def apply_smote(self, X_train, y_train, method="borderline"):
       
        print("\nClass distribution BEFORE resampling")
        print(y_train.value_counts())
 
        if method == "borderline":
            try:
                sampler = BorderlineSMOTE(random_state=RANDOM_STATE, kind="borderline-1")
                X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
            except ValueError:
               
                print("BorderlineSMOTE failed (likely too few neighbors for a class) -- falling back to plain SMOTE.")
                sampler = SMOTE(random_state=RANDOM_STATE)
                X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
        elif method == "smote_tomek":
            sampler = SMOTETomek(random_state=RANDOM_STATE)
            X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
        else:
            sampler = SMOTE(random_state=RANDOM_STATE)
            X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
 
        print(f"\nClass distribution AFTER resampling (method={method})")
        print(y_train_res.value_counts())
 
        return X_train_res, y_train_res
 
    def scale(self, X_train, X_test):
 
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
 
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
 
        print("\nFeature Scaling Completed")
 
        return X_train_scaled, X_test_scaled
