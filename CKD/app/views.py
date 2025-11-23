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

    # Check if 'classification' column exists after preprocessing
    if 'classification' not in df.columns:
        print("❌ ERROR: 'classification' column missing from ckd_dataset1.csv after preprocessing.")
        return
        
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

    # Ensure feature columns are loaded
    if FEATURE_COLUMNS is None:
        print("Error: FEATURE_COLUMNS is None. Model may not be loaded.")
        load_or_train_model()
        if FEATURE_COLUMNS is None:
             raise ValueError("Model features are not loaded. Cannot process input.")


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
            value = data.get(key)
            if value == '' or value is None:
                value = 0
            input_dict[col] = float(value)

   # Handle OHE (sg, al, su) with closest value mapping
    sg = float(data.get("sg", 0))
    al = float(data.get("al", 0))
    su = float(data.get("su", 0))

    # Define known categories
    sg_values = [1.010, 1.015, 1.020, 1.025]
    al_values = [1, 2, 3, 4, 5]
    su_values = [1, 2, 3, 4, 5]

    # Map to closest known value
    def closest(val, valid_list):
        return min(valid_list, key=lambda x: abs(x - val))

    sg_mapped = closest(sg, sg_values)
    al_mapped = closest(al, al_values)
    su_mapped = closest(su, su_values)

    # SG
    for val in sg_values:
        colname = f"specific_gravity_{val}"
        if colname in input_dict:
            input_dict[colname] = 1 if val == sg_mapped else 0

    # Albumin
    for val in al_values:
        colname = f"albumin_{val}"
        if colname in input_dict:
            input_dict[colname] = 1 if val == al_mapped else 0

    # Sugar
    for val in su_values:
        colname = f"sugar_{val}"
        if colname in input_dict:
            input_dict[colname] = 1 if val == su_mapped else 0


    return pd.DataFrame([input_dict])[FEATURE_COLUMNS]


# =====================================================
# HOME PAGE
# =====================================================
def home(request):
    return render(request, "app/home.html")


# =====================================================
# PREDICT FUNCTION (Handles BOTH Manual and CSV)
# =====================================================
def predict_ckd(request):
    global MODEL, SCALER, FEATURE_COLUMNS
    if 'report_data' in request.session:
        del request.session['report_data']
    # Check if model is loaded
    if MODEL is None or SCALER is None or FEATURE_COLUMNS is None:
        load_or_train_model() # Try to load/train again
        if MODEL is None:
            return render(request, "app/predict.html", {
                "error_message": "ERROR: Model could not be loaded. Check logs.",
            })

    # ================= CSV UPLOAD (POST request) =================
    if request.method == "POST" and "csv_submit" in request.POST:
        csv_file = request.FILES.get("csv_file")
        prediction_text = None

        if not csv_file:
            return render(request, "app/predict.html", {"csv_error": "No file uploaded."})

        try:
            df_display = pd.read_csv(csv_file)
            report_data = df_display.iloc[0].to_dict() 
            
            # --- NEW LOGIC: Preprocess and get FIRST ROW ---
            df_processed = preprocess_data(df_display.copy()) 
            
            # Create DataFrame for 1 row based on model columns
            df_model_input = pd.DataFrame(0, index=[0], columns=FEATURE_COLUMNS) 
            common_cols = [col for col in df_processed.columns if col in FEATURE_COLUMNS]
            
            if common_cols:
                # Get just the first row's data
                df_model_input[common_cols] = df_processed[common_cols].iloc[0]

            df_scaled = SCALER.transform(df_model_input)
            prediction = MODEL.predict(df_scaled)[0] # Get the single prediction
            
            # CORRECTED LOGIC: Set text result instead of HTML table
            prediction_text = "CKD Positive" if prediction == 1 else "CKD Negative"
            report_data['Impression'] = prediction_text
            request.session['report_data'] = report_data

        except Exception as e:
            prediction_text = f"Error processing CSV file: {e}. Make sure the CSV format is correct."

        # Render result using prediction_text so the template shows the colored card
        return render(request, "app/result.html", {"prediction_text": prediction_text})

    # ================= MANUAL FORM (POST request) =================
    if request.method == "POST":
        report_data = {}
        prediction_text = None
        report_data = {
                'Age': request.POST.get('age'),
                'Blood Pressure': request.POST.get('bp'),
                'Blood Glucose': request.POST.get('bgr'),
                'Blood Urea': request.POST.get('bu'),
                'Serum Creatinine': request.POST.get('sc'),
                'Sodium': request.POST.get('sod'),
                'Potassium': request.POST.get('pot'),
                'Hemoglobin': request.POST.get('hemo'),
                'PCV': request.POST.get('pcv'),
                'WBC Count': request.POST.get('wbcc'),
                'RBC Count': request.POST.get('rbcc'),
                'Specific Gravity': request.POST.get('sg'),
                'Albumin': request.POST.get('al'),
                'Sugar': request.POST.get('su'),
                'Red Blood Cells': request.POST.get('rbc'),
                'Pus Cell': request.POST.get('pc'),
                'Pus Cell Clumps': request.POST.get('pcc'),
                'Bacteria': request.POST.get('ba'),
                'Hypertension': request.POST.get('htn'),
                'Diabetes Mellitus': request.POST.get('dm'),
                'Coronary Artery Disease': request.POST.get('cad'),
                'Appetite': request.POST.get('appet'),
                'Pedal Edema': request.POST.get('pe'),
                'Anemia': request.POST.get('ane'),
                # add any additional fields from your form
            }
        try:
            df = preprocess_input_manual(request.POST)
            df_scaled = SCALER.transform(df)
            result = MODEL.predict(df_scaled)[0]
            prediction_text = "CKD Positive" if result == 1 else "CKD Negative"
            report_data['Impression'] = prediction_text
            request.session['report_data'] = report_data
        except Exception as e:
            prediction_text = f"Error during prediction: {e}"
        
        # Render the result
        return render(request, "app/result.html", {
            "prediction_text": prediction_text
        })

    # ================= INITIAL GET REQUEST =================
    return render(request, "app/predict.html", {})

# views.py
from django.shortcuts import render
from django.http import HttpResponse
import csv

def patient_report(request):
    # Fetch patient data from session
    report_data = request.session.get('report_data')
    if not report_data:
        return render(request, 'app/report.html', {'error': 'No patient data found.'})

    # Mapping from short CSV/form names to full display names
    feature_name_map = {
        'age': 'Age', 'bp': 'Blood Pressure', 'bgr': 'Blood Glucose',
        'bu': 'Blood Urea', 'sc': 'Serum Creatinine', 'sod': 'Sodium',
        'pot': 'Potassium', 'hemo': 'Hemoglobin', 'pcv': 'PCV',
        'wbcc': 'WBC Count', 'rbcc': 'RBC Count', 'sg': 'Specific Gravity',
        'al': 'Albumin', 'su': 'Sugar', 'rbc': 'Red Blood Cells',
        'pc': 'Pus Cell', 'pcc': 'Pus Cell Clumps', 'ba': 'Bacteria',
        'htn': 'Hypertension', 'dm': 'Diabetes Mellitus',
        'cad': 'Coronary Artery Disease', 'appet': 'Appetite',
        'pe': 'Pedal Edema', 'ane': 'Anemia', 'Impression': 'Impression'
    }

    # Normal ranges and units only for numeric features
    normal_ranges = {
        'Age': ('0-130', 'years'), 'Blood Pressure': ('80 – 120', 'mm/Hg'),
        'Blood Glucose': ('70 – 140', 'mgs/dL'), 'Blood Urea': ('6 – 21(F), 8 – 24(M)', 'mgs/dL'),
        'Serum Creatinine': ('0.6 – 1.1(F), 0.7 – 1.3(M)', 'mgs/dL'), 'Sodium': ('135 – 145', 'mEq/L'),
        'Potassium': ('3.5 – 5.2', 'mEq/L'), 'Hemoglobin': ('12 – 15.5(F), 13.5 – 17.5(M)', 'gms'),
        'PCV': ('37 – 47(F), 40 – 54(M)', '%'), 'WBC Count': ('4000 – 11000', 'cells/cumm'),
        'RBC Count': ('4.2 – 5.4(F), 4.7 – 6.1(M)', 'million/cmm'),
        'Specific Gravity': ('1.005 – 1.030', '—'), 'Albumin': ('0-30', 'mg/g'),
        'Sugar': ('0-15', 'mg/dL')
    }

    # Binary/categorical features
    binary_features = ['Red Blood Cells', 'Pus Cell', 'Pus Cell Clumps', 'Bacteria',
                       'Hypertension', 'Diabetes Mellitus', 'Coronary Artery Disease',
                       'Appetite', 'Pedal Edema', 'Anemia']

    # Prepare report list
    report_list = []
    for key, value in report_data.items():
        feature = feature_name_map.get(key, key)

        # Only numeric features get normal range/unit
        if feature in binary_features or feature == 'Impression':
            normal, unit = ('-', '-')
        else:
            normal, unit = normal_ranges.get(feature, ('-', '-'))

        # Map 0/1 to human-readable
        if feature in ['Red Blood Cells', 'Pus Cell']:
            display_value = 'Normal' if str(value) in ['0','0.0'] else 'Abnormal'
        elif feature in ['Pus Cell Clumps', 'Bacteria']:
            display_value = 'Not Present' if str(value) in ['0','0.0'] else 'Present'
        elif feature in ['Hypertension','Diabetes Mellitus','Coronary Artery Disease',
                         'Appetite','Pedal Edema','Anemia']:
            display_value = 'No' if str(value) in ['0','0.0'] else 'Yes'
        else:
            display_value = value

        report_list.append({
            'feature': feature,
            'value': display_value,
            'normal': normal,
            'unit': unit
        })

    # CSV download
    if 'download' in request.GET:
        import csv
        from django.http import HttpResponse
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="patient_report.csv"'
        writer = csv.writer(response)
        writer.writerow(['Feature', 'Value', 'Normal Range', 'Unit'])
        for row in report_list:
            writer.writerow([row['feature'], row['value'], row['normal'], row['unit']])
        return response

    return render(request, 'app/report.html', {'report_list': report_list})





def reference(request):
    return render(request, "app/reference.html")