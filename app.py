import os
import json
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="Sex Prediction from Footprints", version="1.0.0")

# --- Configuration ---
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
FEATURES_PATH = os.getenv("FEATURES_PATH", "./features.json")
SCALER_FILENAME = os.getenv("SCALER_FILENAME", "scaler.pkl")  # optional

# Expected filenames (adjust via env if your names differ)
DEFAULT_MODELS = {
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
    "SVM (RBF Kernel)": "svm_(rbf_kernel).pkl",
    "Best Tuned Random Forest": "best_random_forest.pkl"
}

# --- Schemas ---
class Footprint(BaseModel):
    Age: float = Field(..., ge=0, description="Age in years")
    FootSide: str = Field(..., description="Left or Right (case-insensitive)")
    Foot_Length_mm: float
    Foot_Breadth_mm: float
    Ball_Breadth_mm: float
    Heel_Breadth_mm: float
    Toe1_Length_mm: float
    Toe2_Length_mm: float
    Toe3_Length_mm: float
    Toe4_Length_mm: float
    Toe5_Length_mm: float
    Midfoot_Width_mm: float
    Foot_Index_pct: float
    Arch_Index: float
    Heel_Angle_deg: float
    Toe_Angle_deg: float

class Prediction(BaseModel):
    model: str
    predicted_sex: str
    confidence: Optional[float] = None
    confidence_type: Optional[str] = None  # "proba", "decision_score", or None

class PredictResponse(BaseModel):
    ok: bool
    features_order: List[str]
    used_scaler: bool
    results: List[Prediction]

# --- Helpers ---
def load_features(features_path: str) -> List[str]:
    try:
        with open(features_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load features.json: {e}")

def footside_to_num(side: str) -> int:
    s = side.strip().lower()
    if s in ["left", "l"]:
        return 0
    if s in ["right", "r"]:
        return 1
    # Unknown -> raise
    raise ValueError("FootSide must be 'Left' or 'Right'")

def vectorize(d: Dict[str, Any], ordered_features: List[str]) -> np.ndarray:
    x = []
    for k in ordered_features:
        if k == "FootSide":
            x.append(float(footside_to_num(d[k])))
        else:
            x.append(float(d[k]))
    return np.array(x, dtype=float).reshape(1, -1)

def try_predict_proba(model, X) -> Optional[float]:
    # Return positive class probability if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return float(proba[0, 1])
        # If classes are not [0,1], take the max probability
        return float(proba.max(axis=1)[0])
    return None

def try_decision_score(model, X) -> Optional[float]:
    if hasattr(model, "decision_function"):
        df = model.decision_function(X)
        # Convert to float
        if isinstance(df, (list, tuple)):
            return float(df[0])
        else:
            val = float(df.ravel()[0])
            return val
    return None

# --- Load models & scaler at startup ---
MODELS = {}
FEATURES_ORDER: List[str] = []
SCALER = None

@app.on_event("startup")
def _load_artifacts():
    global MODELS, FEATURES_ORDER, SCALER
    FEATURES_ORDER = load_features(FEATURES_PATH)

    # Load models
    not_found = []
    for name, fname in DEFAULT_MODELS.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            try:
                MODELS[name] = joblib.load(path)
            except Exception as e:
                raise RuntimeError(f"Failed to load model '{name}' from {path}: {e}")
        else:
            not_found.append((name, path))

    if not_found:
        missing = "; ".join([f"{n} -> {p}" for n,p in not_found])
        raise RuntimeError(f"Missing model files: {missing}")

    # Optional scaler
    scaler_path = os.path.join(MODEL_DIR, SCALER_FILENAME)
    if os.path.exists(scaler_path):
        try:
            SCALER = joblib.load(scaler_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load scaler from {scaler_path}: {e}")
    else:
        SCALER = None

# --- Routes ---
@app.get("/health")
def health():
    return {
        "ok": True,
        "models_loaded": list(MODELS.keys()),
        "using_scaler": SCALER is not None,
        "features_order": FEATURES_ORDER
    }

@app.post("/predict", response_model=PredictResponse)
def predict(fp: Footprint):
    # Prepare vector
    d = fp.dict()
    try:
        X = vectorize(d, FEATURES_ORDER)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Scale if scaler available
    used_scaler = False
    if SCALER is not None:
        X = SCALER.transform(X)
        used_scaler = True

    results: List[Prediction] = []
    for name, model in MODELS.items():
        # Prediction (assumes classes map to 0/1 where 1 = Female; adjust if needed)
        y = model.predict(X)
        # Determine label
        # Try to infer class mapping from model.classes_ if available
        label = None
        if hasattr(model, "classes_"):
            # Map the predicted class (e.g., 0/1 or 'Male'/'Female') to 'Male'/'Female'
            pred_raw = y[0]
            # If the model already predicts string labels, pass through
            if isinstance(pred_raw, str):
                label = str(pred_raw)
            else:
                # assume 0 -> Male, 1 -> Female
                label = "Female" if int(pred_raw) == 1 else "Male"
        else:
            # Fallback
            label = "Female" if int(y[0]) == 1 else "Male"

        # Confidence
        proba = try_predict_proba(model, X)
        if proba is not None:
            conf = float(proba if label.lower() == "female" else 1.0 - proba)
            results.append(Prediction(model=name, predicted_sex=label, confidence=conf, confidence_type="proba"))
            continue

        # Otherwise use decision score if available
        score = try_decision_score(model, X)
        if score is not None:
            results.append(Prediction(model=name, predicted_sex=label, confidence=score, confidence_type="decision_score"))
        else:
            results.append(Prediction(model=name, predicted_sex=label, confidence=None, confidence_type=None))

    return PredictResponse(ok=True, features_order=FEATURES_ORDER, used_scaler=used_scaler, results=results)
