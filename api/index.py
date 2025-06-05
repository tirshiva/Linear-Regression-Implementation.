from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import os

app = FastAPI(title="House Price Predictor", version="1.0.0")

templates = Jinja2Templates(directory="templates")

MODEL_VERSION = "v1"
model = joblib.load(f"models/model_{MODEL_VERSION}.pkl")
scaler = joblib.load(f"models/scaler_{MODEL_VERSION}.pkl")

# ✅ HTML Form Route
@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(request: Request, Size_sqft: float = Form(...)):
    try:
        scaled_input = scaler.transform([[Size_sqft]])
        prediction = model.predict(scaled_input)[0]
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": round(prediction, 2)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ API for Swagger (optional)
@app.post("/predict")
def predict_api(Size_sqft: float):
    scaled = scaler.transform([[Size_sqft]])
    pred = model.predict(scaled)[0]
    return {"Predicted Price (Lakhs)": round(pred, 2)}