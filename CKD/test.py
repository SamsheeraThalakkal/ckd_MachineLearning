import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, 
    confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== LOAD TRAINED MODEL ====================
def load_model():
    """Load the trained model, scaler, and feature columns"""
    try:
        with open('models/lightgbm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        print("✅ Model, scaler, and features loaded successfully!")
        return model, scaler, feature_columns
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run 'train_model.py' first to train and save the model.")
        exit(1)

# ==================== PREPROCESS UNSEEN DATA ====================
def preprocess_unseen_data(df, feature_columns):
    """
    Preprocess unseen data the same way as training data
    """
    # Column mapping (if using original column names)
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
    
    # Check if we need to rename columns
    if 'bp' in df.columns:
        df.rename(columns=column_mapping, inplace=True)
    
    numerical_cols = [
        'age', 'blood_pressure', 'blood_glucose_random', 'blood_urea',
        'serum_creatinine', 'sodium', 'potassium', 'hemoglobin',
        'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count'
    ]
    categorical_cols = [col for col in df.columns if col not in numerical_cols and col != 'classification']

    # Convert numerical columns
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clean categorical columns
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Handle special cases
    if 'diabetes_mellitus' in df.columns:
        df['diabetes_mellitus'].replace({'\tyes': 'yes', ' yes': 'yes'}, inplace=True)
    if 'coronary_artery_disease' in df.columns:
        df['coronary_artery_disease'].replace({'\tno': 'no', ' no': 'no'}, inplace=True)
    if 'classification' in df.columns:
        df['classification'].replace({'ckd\t': 'ckd'}, inplace=True)

    # Fill missing values
    for col in numerical_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_cols:
        if col in df.columns and len(df[col].mode()) > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Binary mapping
    binary_map = {
        'normal': 1, 'abnormal': 0, 'present': 1, 'notpresent': 0,
        'yes': 1, 'no': 0, 'good': 1, 'poor': 0, 'ckd': 1, 'notckd': 0
    }

    binary_cols = [
        'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
        'hypertension', 'diabetes_mellitus', 'coronary_artery_disease',
        'appetite', 'pedal_edema', 'anemia', 'classification'
    ]
    
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].replace(binary_map)

    # Handle categorical encoding
    categorical_encode = ['specific_gravity', 'albumin', 'sugar']
    for col in categorical_encode:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_encode, drop_first=True)

    # Ensure all training features are present
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reorder columns to match training data
    X = df_encoded[feature_columns]
    
    # Get target if exists
    y = df_encoded['classification'] if 'classification' in df_encoded.columns else None
    
    return X, y

# ==================== EVALUATE MODEL ====================
def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate model performance on unseen data
    """
    # Scale the data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION ON UNSEEN DATA")
    print("="*60)
    print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✅ ROC-AUC Score: {roc_auc:.4f}")
    
    print("\n📊 Classification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, 
                                target_names=['Not CKD', 'CKD'],
                                digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n📈 Confusion Matrix:")
    print(f"                Predicted Not CKD    Predicted CKD")
    print(f"Actual Not CKD        {cm[0][0]}                  {cm[0][1]}")
    print(f"Actual CKD            {cm[1][0]}                  {cm[1][1]}")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not CKD', 'CKD'],
                yticklabels=['Not CKD', 'CKD'])
    plt.title('Confusion Matrix - Unseen Data', fontsize=16)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/unseen_data_confusion_matrix.png', dpi=300)
    print("\n💾 Confusion matrix saved to: models/unseen_data_confusion_matrix.png")
    plt.show()
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Unseen Data', fontsize=16)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('models/unseen_data_roc_curve.png', dpi=300)
    print("💾 ROC curve saved to: models/unseen_data_roc_curve.png")
    plt.show()
    
    return accuracy, roc_auc, y_pred, y_proba

# ==================== PREDICT ON UNSEEN DATA ====================
def predict_unseen_data(model, X, scaler):
    """
    Make predictions on unseen data without labels
    """
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    results = pd.DataFrame({
        'Prediction': ['CKD' if p == 1 else 'Not CKD' for p in predictions],
        'CKD_Probability': probabilities[:, 1],
        'Not_CKD_Probability': probabilities[:, 0],
        'Confidence': [max(p) for p in probabilities]
    })
    
    return results

# ==================== CREATE SAMPLE UNSEEN DATA ====================
def create_sample_unseen_data():
    """
    Create sample unseen test data
    """
    unseen_data = pd.DataFrame({
        'age': [72, 28, 55, 45, 60],
        'bp': [95, 70, 85, 80, 90],
        'sg': [1.010, 1.020, 1.015, 1.020, 1.010],
        'al': [4.0, 0.0, 2.0, 0.0, 3.0],
        'su': [3.0, 0.0, 1.0, 0.0, 2.0],
        'rbc': ['abnormal', 'normal', 'normal', 'normal', 'abnormal'],
        'pc': ['abnormal', 'normal', 'normal', 'normal', 'abnormal'],
        'pcc': ['present', 'notpresent', 'notpresent', 'notpresent', 'present'],
        'ba': ['present', 'notpresent', 'notpresent', 'notpresent', 'present'],
        'bgr': [150, 90, 110, 100, 135],
        'bu': [90, 20, 40, 30, 70],
        'sc': [4.2, 0.9, 1.5, 1.1, 3.0],
        'sod': [128, 140, 138, 142, 132],
        'pot': [6.0, 4.0, 4.5, 4.2, 5.5],
        'hemo': [8.5, 15.5, 12.0, 14.0, 10.0],
        'pcv': [28, 45, 38, 42, 32],
        'wbcc': [10000, 7000, 8000, 7500, 9500],
        'rbcc': [3.0, 5.2, 4.5, 4.8, 3.5],
        'htn': ['yes', 'no', 'yes', 'no', 'yes'],
        'dm': ['yes', 'no', 'no', 'no', 'yes'],
        'cad': ['yes', 'no', 'no', 'no', 'no'],
        'appet': ['poor', 'good', 'good', 'good', 'poor'],
        'pe': ['yes', 'no', 'no', 'no', 'yes'],
        'ane': ['yes', 'no', 'no', 'no', 'yes'],
        'class': ['ckd', 'notckd', 'notckd', 'notckd', 'ckd']
    })
    
    unseen_data.to_csv('unseen_test_data.csv', index=False)
    print("✅ Sample unseen data created: unseen_test_data.csv")
    return unseen_data

# ==================== MAIN FUNCTION ====================
def main():
    print("="*60)
    print("CKD PREDICTION - TESTING ON UNSEEN DATA")
    print("="*60)
    
    # Load model
    model, scaler, feature_columns = load_model()
    
    # Check if unseen data file exists
    try:
        print("\n📂 Loading unseen test data...")
        df_unseen = pd.read_csv('unseen_test_data.csv')
        print(f"✅ Loaded {len(df_unseen)} samples from unseen_test_data.csv")
    except FileNotFoundError:
        print("⚠️  No unseen_test_data.csv found. Creating sample data...")
        df_unseen = create_sample_unseen_data()
    
    print(f"\n📊 Dataset shape: {df_unseen.shape}")
    print(f"📋 Columns: {list(df_unseen.columns)}")
    
    # Preprocess data
    print("\n🔄 Preprocessing unseen data...")
    X_test, y_test = preprocess_unseen_data(df_unseen.copy(), feature_columns)
    print(f"✅ Preprocessed data shape: {X_test.shape}")
    
    if y_test is not None:
        # If we have labels, evaluate the model
        print("\n🧪 Evaluating model performance...")
        accuracy, roc_auc, predictions, probabilities = evaluate_model(
            model, X_test, y_test, scaler
        )
        
        # Create detailed results
        results_df = pd.DataFrame({
            'True_Label': ['CKD' if y == 1 else 'Not CKD' for y in y_test],
            'Predicted_Label': ['CKD' if p == 1 else 'Not CKD' for p in predictions],
            'CKD_Probability': probabilities,
            'Confidence': [max(p) for p in model.predict_proba(scaler.transform(X_test))],
            'Correct': y_test.values == predictions
        })
        
        # Save results
        results_df.to_csv('models/unseen_data_predictions.csv', index=False)
        print("\n💾 Detailed predictions saved to: models/unseen_data_predictions.csv")
        
        # Show sample predictions
        print("\n📋 Sample Predictions:")
        print("-" * 80)
        print(results_df.head(10).to_string(index=False))
        
    else:
        # If no labels, just make predictions
        print("\n🔮 Making predictions (no labels provided)...")
        results = predict_unseen_data(model, X_test, scaler)
        results.to_csv('models/unseen_data_predictions.csv', index=False)
        print("\n💾 Predictions saved to: models/unseen_data_predictions.csv")
        
        print("\n📋 Sample Predictions:")
        print("-" * 80)
        print(results.head(10).to_string(index=False))
    
    print("\n" + "="*60)
    print("✅ TESTING COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()