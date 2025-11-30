# Sex Prediction API (FastAPI + Docker)

This service loads your trained models (*.pkl) and exposes a `/predict` endpoint
that returns the predicted sex and a confidence for each model.

## Project layout

```
sex-prediction-api/
├─ app.py
├─ features.json             # expected feature order
├─ requirements.txt
├─ Dockerfile
└─ models/                   # drop your .pkl files here (not committed)
```

## Place your models

Copy these files into `models/` (filenames must match, or override via env):

- decision_tree.pkl
- random_forest.pkl
- xgboost.pkl
- svm_(rbf_kernel).pkl
- best_random_forest.pkl
- scaler.pkl  (optional; if present, it's used to standardize inputs)

## Build & run with Docker

```bash
cd docker/

docker build --no-cache -f Dockerfile -t swikritipal09/footprint-prediction ..

docker run --rm -p 8000:8000 --name footprint-api swikritipal09/footprint-prediction:latest
```

Open docs at: http://localhost:8000/docs

## Example request

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{
  "Age": 30,
  "FootSide": "Left",
  "Foot_Length_mm": 250.0,
  "Foot_Breadth_mm": 95.0,
  "Ball_Breadth_mm": 85.0,
  "Heel_Breadth_mm": 60.0,
  "Toe1_Length_mm": 40.0,
  "Toe2_Length_mm": 35.0,
  "Toe3_Length_mm": 32.0,
  "Toe4_Length_mm": 28.0,
  "Toe5_Length_mm": 25.0,
  "Midfoot_Width_mm": 28.0,
  "Foot_Index_pct": 38.0,
  "Arch_Index": 0.12,
  "Heel_Angle_deg": 20.0,
  "Toe_Angle_deg": 10.0
}'
```

## Notes

- If your SVM was not trained with `probability=True`, it won't provide `predict_proba`. The API will return `confidence_type="decision_score"` instead.
- Class mapping defaults to `0 -> Male`, `1 -> Female`. If your models output strings already, the API will pass them through.
- You can change file names or paths using environment variables:
  - `MODEL_DIR`, `FEATURES_PATH`, `SCALER_FILENAME`
