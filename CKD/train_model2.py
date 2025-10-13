import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, roc_curve, auc
)
from imblearn.over_sampling import SMOTE
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')
    print("Created 'models' directory")

# -------------------- DATA PREPROCESSING --------------------
def preprocess_data(df):
    """
    Cleans and preprocesses the CKD dataset.
    """
    print("\n--- Class distribution before SMOTE ---")
    print(df["class"].value_counts())
    column_mapping = {
        'bp': 'blood_pressure', 'sg': 'specific_gravity', 'al': 'albumin',
        'su': 'sugar', 'rbc': 'red_blood_cells', 'pc': 'pus_cell',
        'pcc': 'pus_cell_clumps', 'ba': 'bacteria', 'bgr': 'blood_glucose_random',
        'bu': 'blood_urea', 'sc': 'serum_creatinine', 'sod': 'sodium',
        'pot': 'potassium', 'hemo': 'hemoglobin', 'pcv': 'packed_cell_volume',
        'wbcc': 'white_blood_cell_count', 'rbcc': 'red_blood_cell_count',
        'htn': 'hypertension', 'dm': 'diabetes_mellitus', 'cad': 'coronary_artery_disease',
        'appet': 'appetite', 'pe': 'pedal_edema', 'ane': 'anemia',
        'class': 'classification'
    }
    df.rename(columns=column_mapping, inplace=True)

    numerical_cols = [
        'age', 'blood_pressure', 'blood_glucose_random', 'blood_urea',
        'serum_creatinine', 'sodium', 'potassium', 'hemoglobin',
        'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count'
    ]
    categorical_cols = [col for col in df.columns if col not in numerical_cols]

    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in categorical_cols:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    df['diabetes_mellitus'].replace({'\tyes': 'yes', ' yes': 'yes'}, inplace=True)
    df['coronary_artery_disease'].replace({'\tno': 'no', ' no': 'no'}, inplace=True)
    df['classification'].replace({'ckd\t': 'ckd'}, inplace=True)

    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    binary_map = {
        'normal': 1, 'abnormal': 0, 'present': 1, 'notpresent': 0,
        'yes': 1, 'no': 0, 'good': 1, 'poor': 0, 'ckd': 1, 'notckd': 0
    }

    for col in [
        'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
        'hypertension', 'diabetes_mellitus', 'coronary_artery_disease',
        'appetite', 'pedal_edema', 'anemia', 'classification'
    ]:
        df[col] = df[col].replace(binary_map)

    for col in ['specific_gravity', 'albumin', 'sugar']:
        df[col] = df[col].astype('category')

    df = pd.get_dummies(df, columns=['specific_gravity', 'albumin', 'sugar'], drop_first=True)

    df.to_csv("predataset.csv", index=False)
    print("--- Preprocessed data saved to predataset.csv ---")
    return df

# -------------------- VISUALIZATION HELPERS --------------------
def plot_confusion_and_roc(y_true, y_pred, y_proba, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not CKD", "CKD"],
                yticklabels=["Not CKD", "CKD"])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"models/{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.show()

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.title(f"{model_name} - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"models/{model_name.lower().replace(' ', '_')}_roc_curve.png")
    plt.show()

# -------------------- MAIN FUNCTION --------------------
def main():
    try:
        df = pd.read_csv("ckd_dataset1.csv")
    except FileNotFoundError:
        print("Error: ckd_dataset1.csv not found.")
        return

    df_clean = preprocess_data(df)

    X = df_clean.drop("classification", axis=1)
    y = df_clean["classification"]

    # Save feature columns for later use
    feature_columns = X.columns.tolist()
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    print(f"--- Saved feature columns to models/feature_columns.pkl ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("--- Saved scaler to models/scaler.pkl ---")

    smote = SMOTE(sampling_strategy={0: 500, 1: 500}, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"\n--- Data ready. Resampled training shape: {X_train_resampled.shape} ---")
        # Display before and after SMOTE values
    print("\n--- Class distribution before SMOTE ---")
    print(df_clean["classification"].value_counts())

    print("\n--- Class distribution after SMOTE ---")
    print(y_train_resampled.value_counts())

    class_counts = Counter(y_train_resampled)

# 2. Prepare data for plotting
# Convert the Counter object to a DataFrame
    class_df = pd.DataFrame(
    list(class_counts.items()),
    columns=['Class', 'Count']
)

# Map numeric classes to meaningful labels for the plot
# Assuming 0=CKD and 1=Not CKD based on prior analysis
    class_df['Class_Label'] = class_df['Class'].map({
    0: 'CKD (Resampled)',
    1: 'Not CKD (Resampled)'
})

# 3. Create the visualization (Count Plot)
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x='Class_Label', y='Count', data=class_df, palette='viridis')

    plt.title('Class Distribution After SMOTE Oversampling', fontsize=14)
    plt.xlabel('Diagnosis Class', fontsize=12)
    plt.ylabel('Count (Number of Samples)', fontsize=12)
    plt.xticks(rotation=0)

# Add counts on top of the bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.tight_layout()
    plt.savefig('class_distribution_after_smote.png')
    plt.show()




# -------------------- VISUALIZATIONS --------------------



    # --- Visualization 2: Relationship between Serum Creatinine ('sc') and CKD ('class') ---

    plt.figure(figsize=(8, 6))
    # Use a box plot to visualize the distribution of Serum Creatinine (sc) across the two classes
    sns.boxplot(x='classification', y='serum_creatinine', data=df, palette='Set2')

    plt.title('Serum Creatinine (sc) Distribution by CKD Status', fontsize=14)
    plt.xlabel('Diagnosis', fontsize=12)
    plt.ylabel('Serum Creatinine (mg/dL)', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('creatinine_vs_ckd.png')
    
    # --- Visualization 2: Relationship between hemoglobin ('sc') and CKD ('class') ---

    plt.figure(figsize=(8, 6))
    # Use a box plot to visualize the distribution of Serum Creatinine (sc) across the two classes
    sns.boxplot(x='classification', y='hemoglobin', data=df, palette='Set2')

    plt.title('Hemoglobin (sc) Distribution by CKD Status', fontsize=14)
    plt.xlabel('Diagnosis', fontsize=12)
    plt.ylabel('Hemoglobin (mg/dL)', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('hemoglobin_vs_ckd.png')
    plt.show()








    model_results = []

    # ---------------- Random Forest ----------------
    print("\n--- Training Random Forest with 5-fold CV ---")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42
    )
    cv_scores = cross_val_score(rf_model, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy')
    print(f"Random Forest 5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    rf_model.fit(X_train_resampled, y_train_resampled)
        # Calculate and print training accuracy for Random Forest
    y_pred_rf_train = rf_model.predict(X_train_resampled)
    acc_rf_train = accuracy_score(y_train_resampled, y_pred_rf_train)
    print(f"Random Forest Training Accuracy: {acc_rf_train:.4f}")
# Evaluate Random Forest
    y_pred_rf_test = rf_model.predict(X_test_scaled)
    y_proba_rf_test = rf_model.predict_proba(X_test_scaled)[:, 1]
    acc_rf = accuracy_score(y_test, y_pred_rf_test)
    print(f"Random Forest Test Accuracy: {acc_rf:.4f}")
    

        # Classification report for Random Forest
    print("\n--- Random Forest Classification Report ---")
    print(classification_report(y_test, y_pred_rf_test, target_names=["Not CKD", "CKD"]))


    model_results.append(("Random Forest", acc_rf))
    
    # Save Random Forest model
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("--- Saved Random Forest model to models/random_forest_model.pkl ---")
    
    plot_confusion_and_roc(y_test, y_pred_rf_test, y_proba_rf_test, "Random Forest")

    # ---------------- LightGBM (OPTIMIZED) ----------------
    print("\n--- Training LightGBM with Optimized Parameters ---")
    # Split training into train/validation for early stopping
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_resampled, y_train_resampled, test_size=0.1, random_state=42, stratify=y_train_resampled
    )

    # OPTIMIZED PARAMETERS for small dataset
    best_lgb = lgb.LGBMClassifier(
        n_estimators=200,           # Reduced from 500
        learning_rate=0.05,         # Slightly increased for faster convergence
        num_leaves=15,              # Reduced from 40 (key change!)
        max_depth=5,                # Reduced from 8 (key change!)
        subsample=0.8,              # Slightly reduced
        colsample_bytree=0.8,       # Slightly reduced
        reg_alpha=0.01,             # Much lower regularization
        reg_lambda=0.01,            # Much lower regularization
        min_child_samples=5,        # Reduced from 15 (key change!)
        min_split_gain=0.0,         # Allow any positive gain
        random_state=42,
        verbosity=-1                # Suppress warnings
    )
    
    # Fit with early stopping
    best_lgb.fit(
        X_train_final, y_train_final,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )

    # Calculate and print training accuracy for LightGBM
    y_pred_lgb_train = best_lgb.predict(X_train_final)
    acc_lgb_train = accuracy_score(y_train_final, y_pred_lgb_train)
    print(f"LightGBM Training Accuracy: {acc_lgb_train:.4f}")


    # Evaluate LightGBM
    y_pred_lgb_test = best_lgb.predict(X_test_scaled)
    y_proba_lgb_test = best_lgb.predict_proba(X_test_scaled)[:, 1]
    acc_lgb = accuracy_score(y_test, y_pred_lgb_test)
    print(f"LightGBM Test Accuracy: {acc_lgb:.4f}")
       

        # Classification report for LightGBM
    print("\n--- LightGBM Classification Report ---")
    print(classification_report(y_test, y_pred_lgb_test, target_names=["Not CKD", "CKD"]))


    print(f"Best iteration: {best_lgb.best_iteration_}")
    model_results.append(("LightGBM", acc_lgb))
    
    # Save LightGBM model
    with open('models/lightgbm_model.pkl', 'wb') as f:
        pickle.dump(best_lgb, f)
    print("--- Saved LightGBM model to models/lightgbm_model.pkl ---")
    
    plot_confusion_and_roc(y_test, y_pred_lgb_test, y_proba_lgb_test, "LightGBM")

   

    # Feature importance
    feature_importances = pd.Series(best_lgb.feature_importances_, index=X.columns)
    top_features = feature_importances.sort_values(ascending=False).head(10)
    
    # Save feature importance
    feature_importances.to_csv('models/feature_importances.csv')
    print("--- Saved feature importances to models/feature_importances.csv ---")
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
    plt.title("Top 10 Important Features - LightGBM")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig('models/feature_importance_plot.png')
    plt.show()

   

    # ---------------- Comparison Summary ----------------
    print("\n\n========== MODEL ACCURACY COMPARISON ==========")
    comparison_df = pd.DataFrame(model_results, columns=["Model", "Test Accuracy"])
    print(comparison_df.sort_values(by="Test Accuracy", ascending=False).to_string(index=False))
    
    # Save comparison results
    comparison_df.to_csv('models/model_comparison.csv', index=False)
    print("\n--- Saved model comparison to models/model_comparison.csv ---")

    plt.figure(figsize=(8, 4))
    sns.barplot(x="Model", y="Test Accuracy", data=comparison_df.sort_values(by="Test Accuracy", ascending=False))
    plt.title("Model Accuracy Comparison")
    plt.ylim(0.9, 1.0)
    plt.tight_layout()
    plt.savefig('models/model_comparison_plot.png')
    plt.show()
    
    print("\n" + "="*50)
    print("ALL MODELS AND ARTIFACTS SAVED TO 'models' FOLDER")
    print("="*50)
    print("Saved files:")
    print("  - models/lightgbm_model.pkl")
    print("  - models/random_forest_model.pkl")
    print("  - models/scaler.pkl")
    print("  - models/feature_columns.pkl")
    print("  - models/feature_importances.csv")
    print("  - models/model_comparison.csv")
    print("  - models/*_confusion_matrix.png")
    print("  - models/*_roc_curve.png")
    print("  - models/feature_importance_plot.png")
    print("  - models/model_comparison_plot.png")
    print("="*50)


if __name__ == "__main__":
    main()