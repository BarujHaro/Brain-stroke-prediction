import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
import joblib

df=pd.read_csv("model\dataset\healthcare-dataset.csv")

df.drop('id', axis=1, inplace=True)
df.dropna(inplace=True)
le=LabelEncoder()
df['gender']=le.fit_transform(df['gender'])
df['ever_married'] = le.fit_transform(df['ever_married'])
df['Residence_type'] = le.fit_transform(df['Residence_type']) 
df = pd.get_dummies(df, columns=['work_type', 'smoking_status'])

X = df[['age','hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']].values
y = df["stroke"].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)


model = XGBClassifier(
    # Parámetros para desbalance
    scale_pos_weight=22,  # Tu ratio calculado
    
    # Parámetros para evitar overfitting
    max_depth=4,          # Reducido para evitar overfitting
    min_child_weight=6,   # Aumentado para casos minoritarios
    subsample=0.8,        # Submuestreo para generalización
    colsample_bytree=0.8, # Submuestreo de features
    
    # Parámetros de regularización
    reg_alpha=1,          # Regularización L1
    reg_lambda=1,         # Regularización L2
    
    # Otros parámetros
    n_estimators=200,     # Más árboles para mejor aprendizaje
    learning_rate=0.1,
    random_state=42,
    eval_metric='aucpr'   # Mejor métrica para datos desbalanceados
)

model.fit(X_train, y_train)


y_proba = model.predict_proba(X_test)[:,1]
y_pred = (y_proba >= 0.04).astype(int)




joblib.dump(model, 'stroke_prediction_model.pkl')


def predict_stroke(age, hypertension, heart_disease, avg_glucose_level, bmi):
    """
    Recibe los parámetros del formulario y devuelve una predicción
    """
    # Crear array con los datos de entrada
    input_data = np.array([[age, hypertension, heart_disease, avg_glucose_level, bmi]])
    

    # Hacer la predicción
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    
    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0][1]),  # Probabilidad de stroke
        "risk_level": "Alto" if probability[0][1] > 0.7 else "Moderado" if probability[0][1] > 0.4 else "Bajo"
    }