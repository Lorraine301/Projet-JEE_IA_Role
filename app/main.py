# FastAPI entry point
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="Real Estate ML Service",
    description="ML microservice for pricing, tenant risk scoring, and recommendations",
    version="1.0.0"
)

# ==============================
# Load ML models
# ==============================
price_model = joblib.load("app/models/price_model.pkl")
risk_model = joblib.load("app/models/risk_model.pkl")
recommend_model = joblib.load("app/models/recommend_model.pkl")
recommend_scaler = joblib.load("app/models/recommend_scaler.pkl")

# ==============================
# Health check
# ==============================
@app.get("/")
def health_check():
    return {"status": "ML Service is running"}

# ==============================
# Request Schemas
# ==============================
class PriceRequest(BaseModel):
    surface: float
    rooms: int
    location_score: float
    distance_center: float
    season_index: float

class RiskRequest(BaseModel):
    late_payments: int
    disputes: int
    rental_duration: int

class RecommendRequest(BaseModel):
    price: float
    surface: float
    rooms: int
    location_score: float
    lifestyle_score: float

# ==============================
# 1. Price Prediction
# ==============================
@app.post("/predict/price")
def predict_price(request: PriceRequest):
    X = np.array([[
        request.surface,
        request.rooms,
        request.location_score,
        request.distance_center,
        request.season_index
    ]])

    predicted_price = price_model.predict(X)[0]

    return {
        "suggested_price": round(float(predicted_price), 2)
    }

# ==============================
# 2. Tenant Risk Scoring
# ==============================
@app.post("/predict/risk")
def predict_risk(request: RiskRequest):
    X = np.array([[
        request.late_payments,
        request.disputes,
        request.rental_duration
    ]])

    risk_probability = risk_model.predict_proba(X)[0][1]

    return {
        "risk_score": round(risk_probability * 100, 2)
    }

# ==============================
# 3. Property Recommendation
# ==============================
@app.post("/recommend")
def recommend_property(request: RecommendRequest):
    X = np.array([[
        request.price,
        request.surface,
        request.rooms,
        request.location_score,
        request.lifestyle_score
    ]])

    X_scaled = recommend_scaler.transform(X)
    cluster = recommend_model.predict(X_scaled)[0]

    return {
        "recommended_cluster": int(cluster)
    }
