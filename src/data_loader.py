import pandas as pd
from src.config import DATA_PATH, TARGET_COLUMN
 
 
class DataLoader:
 
    def __init__(self):
        self.data = None
 
    def load_dataset(self):
        print("=" * 70)
        print(" Loading Dataset...")
        print("=" * 70)
 
        self.data = pd.read_csv(DATA_PATH)
 
        print("✅ Dataset Loaded Successfully!\n")
 
        return self.data
 
    def dataset_summary(self):
 
        print("=" * 70)
        print(" DATASET OVERVIEW")
        print("=" * 70)
 
        print(f"\nDataset Shape : {self.data.shape}")
 
        print("\nColumns\n")
        print(self.data.columns.tolist())
 
        print("\nData Types\n")
        print(self.data.dtypes)
 
        print("\nMissing Values\n")
        print(self.data.isnull().sum())
 
        duplicates = self.data.duplicated().sum()
 
        print(f"\nDuplicate Records : {duplicates}")
 
        print("\nTarget Class Distribution\n")
        print(self.data[TARGET_COLUMN].value_counts())
        print(self.data[TARGET_COLUMN].value_counts(normalize=True).round(3))
 
        print("\nStatistical Summary\n")
        print(self.data.describe())
 
        print("=" * 70)
 
    def remove_duplicates(self):
 
        before = self.data.shape[0]
 
        self.data = self.data.drop_duplicates().reset_index(drop=True)
 
        after = self.data.shape[0]
 
        print("\n🧹 Removing Duplicate Records...")
 
        print(f"Rows Before : {before}")
 
        print(f"Rows After  : {after}")
 
        print(f"Duplicates Removed : {before - after}")
 
        return self.data