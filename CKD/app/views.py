from django.shortcuts import render
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from imblearn.over_sampling import SMOTE


# =====================================================
# GLOBAL VARIABLES
# =====================================================
MODEL = None
SCALER = None
FEATURE_COLUMNS = None

# Ensure 'models' directory exists
os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/lightgbm_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_columns.pkl"


# =====================================================
# LOAD / TRAIN MODEL
# =====================================================
def load_or_train_model():
    """
    Loads pre-trained model OR trains a new one if not found.
    """
    global MODEL, SCALER, FEATURE_COLUMNS

    print("\n🔍 Checking for saved model...")

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH):
        try:
            MODEL = pickle.load(open(MODEL_PATH, "rb"))
            SCALER = pickle.load(open(SCALER_PATH, "rb"))
            FEATURE_COLUMNS = pickle.load(open(FEATURES_PATH, "rb"))

            print("✅ Model loaded successfully.")
            return
        except Exception as e:
            print(f"⚠ Model loading failed: {e}")

    # ---------------- TRAIN NEW MODEL ---------------- #

    print("⚠ No model found. Training a new one...")
    
    try:
        df = pd.read_csv("ckd_dataset1.csv")
    except FileNotFoundError:
        print("❌ ERROR: 'ckd_dataset1.csv' not found. Cannot train new model.")
        print("Please add 'ckd_dataset1.csv' to your project directory.")
        return

    df = preprocess_data(df)

    X = df.drop("classification", axis=1)
    y = df["classification"]

    FEATURE_COLUMNS = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    SCALER = StandardScaler()
    X_train_scaled = SCALER.fit_transform(X_train)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

    MODEL = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )

    MODEL.fit(X_res, y_res)

    # SAVE MODEL
    pickle.dump(MODEL, open(MODEL_PATH, "wb"))
    pickle.dump(SCALER, open(SCALER_PATH, "wb"))
    pickle.dump(FEATURE_COLUMNS, open(FEATURES_PATH, "wb"))

    print("🎉 Model training complete and saved.")


# Auto-load model when Django server starts
load_or_train_model()


# =====================================================
# DATA CLEANING FOR TRAINING & PREDICTION
# =====================================================
def preprocess_data(df):
    """
    Cleans CKD dataset for training or prediction.
    """
    mapping = {
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
    # Only rename columns that exist in the dataframe
    df.rename(columns={k: v for k, v in mapping.items() if k in df.columns}, inplace=True)

    # CLEAN AND FIX CATEGORICAL ERRORS
    if "diabetes_mellitus" in df.columns:
        df["diabetes_mellitus"].replace({'\tyes': 'yes', ' yes': 'yes'}, inplace=True)
    if "coronary_artery_disease" in df.columns:
        df["coronary_artery_disease"].replace({'\tno': 'no', ' no': 'no'}, inplace=True)
    if "classification" in df.columns:
        df["classification"].replace({'ckd\t': 'ckd'}, inplace=True)

    # NUMERIC FIXING
    numeric_cols = [
        'age','blood_pressure','blood_glucose_random','blood_urea',
        'serum_creatinine','sodium','potassium','hemoglobin',
        'packed_cell_volume','white_blood_cell_count','red_blood_cell_count'
    ]
    for c in numeric_cols:
        if c in df.columns: # Check if column exists before processing
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c].fillna(df[c].median(), inplace=True)

    # BINARY MAPPING
    binary_map = {
        "normal": 1, "abnormal": 0,
        "present": 1, "notpresent": 0,
        "yes": 1, "no": 0,
        "good": 1, "poor": 0,
        "ckd": 1, "notckd": 0
    }

    binary_cols = [
        "red_blood_cells","pus_cell","pus_cell_clumps","bacteria",
        "hypertension","diabetes_mellitus","coronary_artery_disease",
        "appetite","pedal_edema","anemia"
    ]
    
    for c in binary_cols:
        if c in df.columns: # Check if column exists
            df[c] = df[c].replace(binary_map)
    
    # Handle classification separately
    if "classification" in df.columns:
        df["classification"] = df["classification"].replace(binary_map)


    # OHE
    ohe_cols = ['specific_gravity', 'albumin', 'sugar']
    cols_to_ohe = [c for c in ohe_cols if c in df.columns]
    if cols_to_ohe:
        df = pd.get_dummies(df, columns=cols_to_ohe, drop_first=True)

    return df


# =====================================================
# MANUAL INPUT PREPROCESSING
# =====================================================
def preprocess_input_manual(data):
    """
    Aligns manual input with training feature columns.
    """
    global FEATURE_COLUMNS

    input_dict = {col: 0 for col in FEATURE_COLUMNS}

    # Direct mapped numeric values
    field_map = {
        "age": "age",
        "bp": "blood_pressure",
        "bgr": "blood_glucose_random",
        "bu": "blood_urea",
        "sc": "serum_creatinine",
        "sod": "sodium",
        "pot": "potassium",
        "hemo": "hemoglobin",
        "pcv": "packed_cell_volume",
        "wbcc": "white_blood_cell_count",
        "rbcc": "red_blood_cell_count",
        "rbc": "red_blood_cells",
        "pc": "pus_cell",
        "pcc": "pus_cell_clumps",
        "ba": "bacteria",
        "htn": "hypertension",
        "dm": "diabetes_mellitus",
        "cad": "coronary_artery_disease",
        "appet": "appetite",
        "pe": "pedal_edema",
        "ane": "anemia"
    }

    for key, col in field_map.items():
        if col in input_dict:
            input_dict[col] = float(data.get(key, 0))

    # Handle OHE (sg, al, su)
    sg = float(data.get("sg"))
    al = float(data.get("al"))
    su = float(data.get("su"))

    # SG
    for val in [1.010, 1.015, 1.020, 1.025]:
        colname = f"specific_gravity_{val}"
        if colname in input_dict:
            input_dict[colname] = 1 if sg == val else 0

    # Albumin
    for val in [1, 2, 3, 4, 5]:
        colname = f"albumin_{val}"
        if colname in input_dict:
            input_dict[colname] = 1 if al == val else 0

    # Sugar
    for val in [1, 2, 3, 4, 5]:
        colname = f"sugar_{val}"
        if colname in input_dict:
            input_dict[colname] = 1 if su == val else 0

    return pd.DataFrame([input_dict])[FEATURE_COLUMNS]


# =====================================================
# HOME PAGE
# =====================================================
def home(request):
    return render(request, "app/home.html")


# =====================================================
# PREDICT FUNCTION
# =====================================================
def predict_ckd(request):
    global MODEL, SCALER, FEATURE_COLUMNS

    prediction_text = None
    csv_html = None

    # Check if model is loaded
    if MODEL is None or SCALER is None or FEATURE_COLUMNS is None:
        load_or_train_model() # Try to load/train again
        if MODEL is None:
            # If still None, render error
            return render(request, "app/predict.html", {
                "prediction_text": "ERROR: Model could not be loaded. Check logs.",
            })

    # ================= CSV UPLOAD =================
    if request.method == "POST" and "csv_submit" in request.POST:
        csv_file = request.FILES.get("csv_file")
        if not csv_file:
             return render(request, "app/predict.html", {"csv_html": "<div class='alert alert-danger'>No file uploaded.</div>"})

        try:
            df_display = pd.read_csv(csv_file)
            # Make a copy for processing
            df_processed = preprocess_data(df_display.copy()) 

            # Align processed df with the model's feature columns
            # Create a new DataFrame with all feature columns, initialized to 0
            df_model_input = pd.DataFrame(0, index=df_processed.index, columns=FEATURE_COLUMNS)

            # Fill in the values from df_processed for columns that match
            # This handles both alignment and adding missing OHE columns as 0
            common_cols = [col for col in df_processed.columns if col in FEATURE_COLUMNS]
            if common_cols:
                df_model_input[common_cols] = df_processed[common_cols]

            df_scaled = SCALER.transform(df_model_input)
            predictions = MODEL.predict(df_scaled)

            # Add prediction to the *original* display DataFrame
            df_display["Predicted_Risk"] = ["CKD Positive" if p == 1 else "CKD Negative" for p in predictions]
            csv_html = df_display.to_html(classes="table table-bordered table-striped", index=False)

        except Exception as e:
            # Handle potential errors during file processing
            csv_html = f"<div class='alert alert-danger'>Error processing CSV file: {e}. Make sure the CSV format is correct.</div>"

        return render(request, "app/predict.html", {"csv_html": csv_html})

    # ================= MANUAL FORM =================
    if request.method == "POST":
        try:
            df = preprocess_input_manual(request.POST)
            df_scaled = SCALER.transform(df)
            result = MODEL.predict(df_scaled)[0]

            prediction_text = "CKD Positive" if result == 1 else "CKD Negative"
        
        except Exception as e:
            prediction_text = f"Error during prediction: {e}"


    return render(request, "app/predict.html", {
        "prediction_text": prediction_text,
        "csv_html": csv_html
    })