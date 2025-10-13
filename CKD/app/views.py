from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

# Global variables to store model and scaler
MODEL = None
SCALER = None
FEATURE_COLUMNS = None

def load_or_train_model():
    """
    Load existing model or train a new one if not found.
    """
    global MODEL, SCALER, FEATURE_COLUMNS
    
    model_path = 'models/lightgbm_model.pkl'
    scaler_path = 'models/scaler.pkl'
    features_path = 'models/feature_columns.pkl'
    
    # Try to load existing model
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
        try:
            with open(model_path, 'rb') as f:
                MODEL = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                SCALER = pickle.load(f)
            with open(features_path, 'rb') as f:
                FEATURE_COLUMNS = pickle.load(f)
            print("Model, scaler, and features loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
    
    # Train new model if loading failed
    print("Training new model...")
    try:
        # Load and preprocess data
        df = pd.read_csv("ckd_dataset1.csv")
        df_clean = preprocess_data(df)
        
        X = df_clean.drop("classification", axis=1)
        y = df_clean["classification"]
        
        # Store feature columns
        FEATURE_COLUMNS = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale data
        SCALER = StandardScaler()
        X_train_scaled = SCALER.fit_transform(X_train)
        
        # Apply SMOTE
        smote = SMOTE(sampling_strategy={0: 500, 1: 500}, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        # Split for early stopping
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_resampled, y_train_resampled, test_size=0.1, 
            random_state=42, stratify=y_train_resampled
        )
        
        # Train LightGBM model
        MODEL = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.01,
            reg_lambda=0.01,
            min_child_samples=5,
            min_split_gain=0.0,
            random_state=42,
            verbosity=-1
        )
        
        MODEL.fit(
            X_train_final, y_train_final,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        
        # Save model, scaler, and features
        with open(model_path, 'wb') as f:
            pickle.dump(MODEL, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(SCALER, f)
        with open(features_path, 'wb') as f:
            pickle.dump(FEATURE_COLUMNS, f)
        
        print("Model trained and saved successfully!")
        return True
        
    except Exception as e:
        print(f"Error training model: {e}")
        return False


def preprocess_data(df):
    """
    Cleans and preprocesses the CKD dataset.
    """
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
    return df


def preprocess_input(form_data):
    """
    Preprocesses form input data to match training data format.
    """
    # Create dataframe from form data
    input_dict = {
        'age': float(form_data.get('age', 0)),
        'blood_pressure': float(form_data.get('bp', 0)),
        'blood_glucose_random': float(form_data.get('bgr', 0)),
        'blood_urea': float(form_data.get('bu', 0)),
        'serum_creatinine': float(form_data.get('sc', 0)),
        'hemoglobin': float(form_data.get('hemo', 0)),
        'packed_cell_volume': float(form_data.get('pcv', 0)),
        'white_blood_cell_count': float(form_data.get('wbcc', 0)),
        'red_blood_cell_count': float(form_data.get('rbcc', 0)),
        'sodium': 135.0,  # Default median value
        'potassium': 4.5,  # Default median value
        'red_blood_cells': int(form_data.get('rbc', 1)),
        'pus_cell': int(form_data.get('pc', 1)),
        'pus_cell_clumps': int(form_data.get('pcc', 0)),
        'bacteria': int(form_data.get('ba', 0)),
        'hypertension': int(form_data.get('htn', 0)),
        'diabetes_mellitus': int(form_data.get('dm', 0)),
        'coronary_artery_disease': int(form_data.get('cad', 0)),
        'appetite': int(form_data.get('appet', 1)),
        'pedal_edema': int(form_data.get('pe', 0)),
        'anemia': int(form_data.get('ane', 0)),
    }
    
    # Handle categorical variables with one-hot encoding
    sg_val = float(form_data.get('sg', 1.020))
    al_val = float(form_data.get('al', 0.0))
    su_val = float(form_data.get('su', 0.0))
    
    # Add dummy variables for specific_gravity
    for val in [1.010, 1.015, 1.020, 1.025]:
        input_dict[f'specific_gravity_{val}'] = 1 if sg_val == val else 0
    
    # Add dummy variables for albumin
    for val in [1.0, 2.0, 3.0, 4.0, 5.0]:
        input_dict[f'albumin_{val}'] = 1 if al_val == val else 0
    
    # Add dummy variables for sugar
    for val in [1.0, 2.0, 3.0, 4.0, 5.0]:
        input_dict[f'sugar_{val}'] = 1 if su_val == val else 0
    
    # Create DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Ensure all training features are present
    for col in FEATURE_COLUMNS:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data
    input_df = input_df[FEATURE_COLUMNS]
    
    return input_df


@csrf_protect
def predict_ckd(request):
    """
    Main view function for CKD prediction.
    """
    global MODEL, SCALER, FEATURE_COLUMNS
    
    # Load or train model on first request
    if MODEL is None:
        load_or_train_model()
    
    prediction_text = None
    
    if request.method == 'POST':
        try:
            # Preprocess input
            input_df = preprocess_input(request.POST)
            
            # Scale input
            input_scaled = SCALER.transform(input_df)
            
            # Make prediction
            prediction = MODEL.predict(input_scaled)[0]
            probability = MODEL.predict_proba(input_scaled)[0]
            
            # Format result
            if prediction == 1:
                prediction_text = f"CKD Positive (Confidence: {probability[1]*100:.2f}%)"
            else:
                prediction_text = f"CKD Negative (Confidence: {probability[0]*100:.2f}%)"
                
        except Exception as e:
            prediction_text = f"Error making prediction: {str(e)}"
    
    return render(request, 'app/predict.html', {
        'prediction_text': prediction_text
    })