#Configuration / configuraciones
from fastapi import FastAPI, Form, Request #Framework
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
from functools import lru_cache

#Create app / Creación de la aplicación
app = FastAPI(title="Sistema de Predicción de Stroke")
#Configuration of the template folder / Configuracion de la carpeta de plantillas
templates = Jinja2Templates(directory="templates")
#Mount static files to the path static / Monta archivos estaticos en la ruta
app.mount("/static", StaticFiles(directory="static"), name="static")
#Load the model ant the scaler wewith cache / Carga el modelo y scaler con cache
@lru_cache(maxsize=1)
def load_model():
    print("Cargando modelo...")
    return joblib.load('stroke_prediction_model.pkl')


#Prediction function / Función de predicción
def predict_stroke(age, hypertension, heart_disease, avg_glucose_level, bmi):
    """Función de predicción que carga el modelo y scaler solo cuando se necesita"""
    clf = load_model() #Model
    
    # Create array with the input data / Crear array con los datos de entrada
    input_data = np.array([[age, hypertension, heart_disease, avg_glucose_level, bmi]])
    
    # Make the prediction / Hacer la predicción
    prediction = clf.predict(input_data)
    probability = clf.predict_proba(input_data)
    
    return {
        "prediction": int(prediction[0]), #0 / 1
        "probability": float(probability[0][1]), # Probability of stroke / Probabilidad de stroke
        "risk_level": "Alto" if probability[0][1] > 0.7 else "Moderado" if probability[0][1] > 0.3 else "Bajo"
    }

#Routes of API / RUtas del api
#Gets the form / Devuelve el formulartio
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

#Post route 
@app.post("/predict/")
async def predict(
    age: float = Form(...),
    avg_glucose_level: float = Form(...),
    bmi: float = Form(...),
    hypertension: str = Form(...),
    heart_disease: str = Form(...)
):
    # Convert answers to numeric values / Convertir las respuestas a valores numéricos
    hypertension_int = 1 if hypertension.lower() == "si" else 0
    heart_disease_int = 1 if heart_disease.lower() == "si" else 0
    
    # Make prediction / Hacer la predicción
    result = predict_stroke(age, hypertension_int, heart_disease_int, avg_glucose_level, bmi)
    
    return JSONResponse(content=result)

###Complete flow:
#User visits / → Views HTML form
#Fills out form → Submits data to /predict/
#API converts "yes"/"no" to 1/0
#Normalizes data with the scaler
#Makes predictions with the model
#Returns JSON with result and risk level 