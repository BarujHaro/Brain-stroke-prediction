# 🧠 Predicción de Stroke (Brain Stroke Prediction) con XGBoost + FastAPI

Este proyecto implementa un modelo de **Machine Learning** para predecir la probabilidad de sufrir un **brain stroke (accidente cerebrovascular)** a partir de características clínicas y de salud de un paciente. El backend está desarrollado en **FastAPI** y el modelo ha sido entrenado con **XGBoost**, guardado con **Joblib** y desplegado para realizar predicciones.

---

## 📌 ¿Qué es un Brain Stroke?
Un **stroke** (accidente cerebrovascular) ocurre cuando el flujo sanguíneo hacia una parte del cerebro se interrumpe o se reduce, privando al tejido cerebral de oxígeno y nutrientes.  
Si no se trata rápidamente, las células cerebrales comienzan a morir, lo que puede provocar daño cerebral permanente, discapacidad o incluso la muerte.  

La predicción temprana puede ayudar en la prevención y tratamiento oportuno.

---

## 📊 Variables utilizadas
El modelo fue entrenado con las siguientes variables independientes:

- **age** → Edad del paciente  
- **hypertension** → Si padece de hipertensión (0 = No, 1 = Sí)  
- **heart_disease** → Si padece de enfermedad cardíaca (0 = No, 1 = Sí)  
- **avg_glucose_level** → Nivel promedio de glucosa en sangre  
- **bmi** → Índice de masa corporal (Body Mass Index)

La variable dependiente es:

- **stroke** → 0 = No, 1 = Sí (indica si el paciente sufrió un stroke)

---

## ⚖️ Balanceo de clases
Dado que el dataset presentaba un **desbalance de clases**, se aplicó la técnica **scale_pos_weight / Escala de pesos por clase**, mejorando la capacidad del modelo de identificar correctamente los casos positivos (stroke).

```python
# Variables independientes (X)
X = df[['age','hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']].values

# Variable dependiente (y)
y = df["stroke"].values

Resultado del modelo:

Matriz de confusión:
[[1197  216]
 [  25   35]]

Reporte de clasificación:
              precision    recall  f1-score   support

           0       0.98      0.85      0.91      1413
           1       0.14      0.58      0.23        60

    accuracy                           0.84      1473
   macro avg       0.56      0.72      0.57      1473
weighted avg       0.95      0.84      0.88      1473

🚀 Tecnologías utilizadas

Python
FastAPI (para exponer la API REST)
XGBoost (modelo ML)
Joblib (para guardar/cargar el modelo)
SMOTE (imblearn) para balanceo de clases

Dataset obtenido de: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data
