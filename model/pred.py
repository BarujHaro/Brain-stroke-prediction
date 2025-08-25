import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from xgboost import XGBClassifier
import joblib

df=pd.read_csv("./dataset/healthcare-dataset.csv")


df.drop('id', axis=1, inplace=True)
df.dropna(inplace=True)
le=LabelEncoder()
df['gender']=le.fit_transform(df['gender'])
df['ever_married'] = le.fit_transform(df['ever_married'])
df['Residence_type'] = le.fit_transform(df['Residence_type']) 
df = pd.get_dummies(df, columns=['work_type', 'smoking_status'])

X = df[['age','hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']].values
y = df["stroke"].values

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.2, random_state = 5)


sca = MinMaxScaler()
X_train = sca.fit_transform(X_train)
X_test = sca.transform(X_test)


clf = XGBClassifier(
    n_estimators=1500,
    learning_rate=0.09,
    subsample=0.7, 
    max_depth=8,
)
clf.fit(X_train , y_train)



y_pred = clf.predict(X_test)




joblib.dump(clf, 'stroke_prediction_model.pkl')
joblib.dump(sca, 'scaler.pkl')


def predict_stroke(age, hypertension, heart_disease, avg_glucose_level, bmi):
    """
    Recibe los parámetros del formulario y devuelve una predicción
    """
    # Crear array con los datos de entrada
    input_data = np.array([[age, hypertension, heart_disease, avg_glucose_level, bmi]])
    
    # Escalar los datos
    input_scaled = sca.transform(input_data)
    
    # Hacer la predicción
    prediction = clf.predict(input_scaled)
    probability = clf.predict_proba(input_scaled)
    
    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0][1]),  # Probabilidad de stroke
        "risk_level": "Alto" if probability[0][1] > 0.7 else "Moderado" if probability[0][1] > 0.3 else "Bajo"
    }