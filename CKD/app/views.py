from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect
from django.http import JsonResponse
import pandas as pd
import numpy as np
import pickle
import os
import json


# ---------- Global Variables ----------
MODEL = None
RF_MODEL = None
SCALER = None
FEATURE_COLUMNS = None


# ---------- Home Page ----------
def home(request):
    """Simple home page"""
    return render(request, 'app/home.html')


# ---------- Model Loading ----------
def load_models():
    """Load existing pre-trained models and scaler."""
    global MODEL, RF_MODEL, SCALER, FEATURE_COLUMNS

    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(BASE_DIR, 'models')

        MODEL = pickle.load(open(os.path.join(model_dir, 'LightGBM_model.pkl'), 'rb'))
        RF_MODEL = pickle.load(open(os.path.join(model_dir, 'random_forest_model.pkl'), 'rb'))
        SCALER = pickle.load(open(os.path.join(model_dir, 'scaler.pkl'), 'rb'))
        FEATURE_COLUMNS = pickle.load(open(os.path.join(model_dir, 'feature_columns.pkl'), 'rb'))

        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")


# ---------- Data Preprocessing ----------
def preprocess_data(df):
    """Cleans CKD dataset and ensures valid medical ranges."""
    df = df.copy()

    df.replace('?', np.nan, inplace=True)
    df = df.dropna(subset=['age', 'bp', 'bgr', 'sg', 'al', 'su'])

    numeric_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot',
                    'hemo', 'pcv', 'wbcc', 'rbcc']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.fillna(df.median(numeric_only=True), inplace=True)

    # Medical logic validation
    df = df[
        (df['age'] >= 0) & (df['age'] <= 120) &
        (df['bp'] >= 40) & (df['bp'] <= 200) &
        (df['bgr'] >= 50) & (df['bgr'] <= 400) &
        (df['hemo'] >= 3) & (df['hemo'] <= 18) &
        (df['sc'] >= 0)
    ]

    # Encode categorical columns if present
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'classification':
            df[col] = df[col].astype('category').cat.codes

    if 'classification' in df.columns:
        df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0}).fillna(0).astype(int)

    return df


# ---------- Single Input Validation ----------
def preprocess_input(data):
    """Validates and prepares input for single prediction."""
    fields = [
        'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot',
        'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
    ]

    validated = {}

    # Numeric validations
    for f in fields:
        val = float(data.get(f, 0))
        if f in ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']:
            if val < 0:
                raise ValueError(f"{f} cannot be negative.")
        if f == 'bp' and not (40 <= val <= 200):
            raise ValueError("Blood pressure must be between 40 and 200.")
        if f == 'bgr' and not (50 <= val <= 400):
            raise ValueError("Blood glucose random must be between 50 and 400.")
        if f == 'hemo' and not (3 <= val <= 18):
            raise ValueError("Hemoglobin must be between 3 and 18 g/dL.")
        if f == 'age' and not (0 <= val <= 120):
            raise ValueError("Age must be between 0 and 120.")
        validated[f] = val

    input_df = pd.DataFrame([validated])
    return input_df


# ---------- Prediction View ----------
@csrf_protect
def predict_ckd(request):
    """Handles both manual input and CSV upload prediction."""
    global MODEL, RF_MODEL, SCALER, FEATURE_COLUMNS

    if MODEL is None or RF_MODEL is None or SCALER is None:
        load_models()

    prediction_text = None

    if request.method == 'POST':
        try:
            # --- CSV Upload ---
            if 'csv_file' in request.FILES:
                model_choice = request.POST.get('model_choice', 'lightgbm')
                model = MODEL if model_choice == 'lightgbm' else RF_MODEL

                df = pd.read_csv(request.FILES['csv_file'])
                df_clean = preprocess_data(df)

                X = df_clean.drop('classification', axis=1, errors='ignore')
                for col in FEATURE_COLUMNS:
                    if col not in X.columns:
                        X[col] = 0
                X = X[FEATURE_COLUMNS]

                X_scaled = SCALER.transform(X)
                preds = model.predict(X_scaled)

                pos, neg = int(sum(preds)), len(preds) - int(sum(preds))
                prediction_text = f"✅ File processed: {pos} CKD Positive, {neg} Negative."

            # --- Manual Input ---
            else:
                try:
                    data = json.loads(request.body.decode('utf-8')) if request.body else request.POST
                except Exception:
                    data = request.POST

                model_choice = data.get('model_choice', 'lightgbm')
                model = MODEL if model_choice == 'lightgbm' else RF_MODEL

                input_df = preprocess_input(data)
                input_scaled = SCALER.transform(input_df)

                pred = model.predict(input_scaled)[0]
                prob = model.predict_proba(input_scaled)[0]

                if pred == 1:
                    prediction_text = f"CKD Positive (Confidence: {prob[1]*100:.2f}%)"
                else:
                    prediction_text = f"CKD Negative (Confidence: {prob[0]*100:.2f}%)"

        except Exception as e:
            prediction_text = f"⚠️ Error: {str(e)}"

    # --- Response handling ---
    if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.content_type == 'application/json':
        if prediction_text:
            return JsonResponse({'status': 'success', 'prediction': prediction_text})
        return JsonResponse({'status': 'error', 'message': 'No prediction made.'})
    else:
        return render(request, 'app/predict.html', {'prediction_text': prediction_text})
