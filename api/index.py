from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import os
import joblib
import os
import requests

app = FastAPI(title="House Price Predictor", version="1.0.0")

templates = Jinja2Templates(directory="templates")

MODEL_VERSION = "v1"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        r = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(r.content)

# ✅ Use your direct download links
MODEL_URL = "https://drive.google.com/file/d/1U7GVnQQLVel5Uy9iQmtd7QqYWImNTxv2/view?usp=drive_link"
SCALER_URL = "https://drive.google.com/file/d/1B1yVn9R5vdgkAl7lyN2HTv5h1KUkyUu5/view?usp=drive_link"

download_file(MODEL_URL, "model_v1.pkl")
download_file(SCALER_URL, "scaler_v1.pkl")

# ✅ Load the model
model = joblib.load("model_v1.pkl")
scaler = joblib.load("scaler_v1.pkl")

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