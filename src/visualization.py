import os
 
import matplotlib.pyplot as plt
import seaborn as sns
 
from src.config import OUTPUT_DIR
 
EDA_FOLDER = os.path.join(OUTPUT_DIR, "eda")
 
os.makedirs(EDA_FOLDER, exist_ok=True)
 
CLASS_LABELS = ["Healthy", "Prediabetic", "Diabetic"]
CLASS_VALUES = [0, 1, 2]
 
AGE_CODE_LABEL = "Age Group Code (1=18-24 ... 13=80+)"
 
 
class Visualizer:
 
    def diabetes_distribution(self, df):
 
        plt.figure(figsize=(8, 6))
 
        ax = sns.countplot(
            x="Diabetes_012",
            hue="Diabetes_012",
            data=df,
            palette="viridis",
            legend=False,
            order=CLASS_VALUES
        )
 
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(CLASS_LABELS)
 
        plt.title("Distribution of Diabetes Categories", fontsize=16, weight="bold")
        plt.xlabel("Patient Category", fontsize=12)
        plt.ylabel("Number of Patients", fontsize=12)
 
        for p in ax.patches:
            ax.annotate(
                f"{int(p.get_height()):,}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=10
            )
 
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_FOLDER, "diabetes_distribution.png"), dpi=300)
        plt.close()
 
    def correlation(self, df):
 
        plt.figure(figsize=(18, 12))
 
        sns.heatmap(df.corr(), cmap="coolwarm", annot=True, fmt=".2f", annot_kws={"size": 6})
 
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_FOLDER, "correlation_heatmap.png"), dpi=300)
        plt.close()
 
    def bmi_distribution(self, df):
 
        plt.figure(figsize=(8, 5))
 
        sns.histplot(df["BMI"], kde=True, bins=35, color="royalblue")
 
        plt.title("BMI Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_FOLDER, "bmi_distribution.png"), dpi=300)
        plt.close()
 
    def age_distribution(self, df):
 
        plt.figure(figsize=(8, 5))
 
        sns.histplot(df["Age"], bins=13, color="green")
 
        plt.title("Age Distribution")
        plt.xlabel(AGE_CODE_LABEL)
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_FOLDER, "age_distribution.png"), dpi=300)
        plt.close()
 
    def blood_pressure(self, df):
 
        plt.figure(figsize=(8, 5))
 
        ax = sns.countplot(
            x="HighBP",
            hue="Diabetes_012",
            data=df,
            palette="Set2",
            hue_order=CLASS_VALUES
        )
 
        ax.legend(title="Category", labels=CLASS_LABELS)
 
        plt.title("High Blood Pressure vs Diabetes")
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_FOLDER, "blood_pressure.png"), dpi=300)
        plt.close()