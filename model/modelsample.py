import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler  # Cambio a StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
from xgboost import XGBClassifier
import joblib
from collections import Counter


df=pd.read_csv("model\dataset\healthcare-dataset.csv")
df.drop('id', axis=1, inplace=True)
df.dropna(inplace=True)

print(f"Dataset original: {df.shape}")
print(f"Casos de stroke: {df['stroke'].sum()} ({df['stroke'].mean()*100:.2f}%)")

# Preprocesamiento más cuidadoso
le_gender = LabelEncoder()
le_married = LabelEncoder()
le_residence = LabelEncoder()

df['gender_encoded'] = le_gender.fit_transform(df['gender'])
df['ever_married_encoded'] = le_married.fit_transform(df['ever_married'])  
df['residence_encoded'] = le_residence.fit_transform(df['Residence_type'])

# Crear variables dummy más inteligentes
work_dummies = pd.get_dummies(df['work_type'], prefix='work')
smoke_dummies = pd.get_dummies(df['smoking_status'], prefix='smoke')

# Dataset final con todas las variables
df_processed = pd.concat([
    df[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 
        'gender_encoded', 'ever_married_encoded', 'residence_encoded', 'stroke']],
    work_dummies,
    smoke_dummies
], axis=1)

# Seleccionar características más importantes (basado en conocimiento médico)
medical_features = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'gender_encoded', 'ever_married_encoded', 'residence_encoded'
]

# Agregar las dummy más relevantes
important_dummies = []
for col in df_processed.columns:
    if col.startswith('work_') or col.startswith('smoke_'):
        important_dummies.append(col)

feature_columns = medical_features + important_dummies
X = df_processed[feature_columns].values
y = df_processed['stroke'].values

print(f"Características finales: {len(feature_columns)}")

# División estratificada más inteligente
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ESTRATEGIA HÍBRIDA DE RESAMPLING
# 1. SMOTE más agresivo pero controlado
# 2. Cleaning con Edited Nearest Neighbours
smote = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=3)
enn = EditedNearestNeighbours(sampling_strategy='majority', n_neighbors=3)

# Pipeline de limpieza y balanceo
resampling_pipeline = ImbPipeline([
    ('smote', smote),
    ('enn', enn)
])

X_resampled, y_resampled = resampling_pipeline.fit_resample(X_train, y_train)

print(f"Datos originales - No stroke: {sum(y_train == 0)}, Stroke: {sum(y_train == 1)}")
print(f"Datos balanceados - No stroke: {sum(y_resampled == 0)}, Stroke: {sum(y_resampled == 1)}")

# Escalado más robusto
scaler = StandardScaler()  # Mejor para datos médicos
X_train_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# ENSEMBLE DE MODELOS MÉDICOS
# Cada modelo optimizado para diferentes aspectos

# 1. XGBoost optimizado para recall (detectar más strokes)
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    scale_pos_weight=3,  # Penalizar más los falsos negativos
    random_state=42
)

# 2. Random Forest para estabilidad
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # Balance automático
    random_state=42
)

# 3. Logistic Regression para interpretabilidad
lr_model = LogisticRegression(
    C=0.1,  # Regularización fuerte
    class_weight='balanced',
    solver='liblinear',
    random_state=42
)

# ENSEMBLE VOTING
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model), 
        ('lr', lr_model)
    ],
    voting='soft',  # Usar probabilidades
    weights=[2, 1, 1]  # XGB tiene más peso
)

# Entrenamiento del ensemble
print("Entrenando ensemble de modelos...")
ensemble.fit(X_train_scaled, y_resampled)

# Predicciones
y_pred = ensemble.predict(X_test_scaled)
y_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1]

# EVALUACIÓN ENFOCADA EN MEDICINA
print("\n=== EVALUACIÓN MÉDICA ===")
print("Classification Report:")
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

# Métricas médicas críticas
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)  # Recall para stroke
specificity = tn / (tn + fp)  # Recall para no-stroke
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision para stroke
npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Precision para no-stroke

print(f"\n=== MÉTRICAS MÉDICAS ===")
print(f"Sensibilidad (detecta stroke): {sensitivity:.3f} ({sensitivity*100:.1f}%)")
print(f"Especificidad (detecta no-stroke): {specificity:.3f} ({specificity*100:.1f}%)")
print(f"Valor Predictivo Positivo: {ppv:.3f} ({ppv*100:.1f}%)")
print(f"Valor Predictivo Negativo: {npv:.3f} ({npv*100:.1f}%)")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

print(f"\n=== MATRIZ DE CONFUSIÓN ===")
print(f"Verdaderos Negativos (TN): {tn}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Falsos Negativos (FN): {fn} ← ¡Crítico en medicina!")
print(f"Verdaderos Positivos (TP): {tp}")

# Análisis de umbrales para optimizar detección
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Encontrar umbral óptimo para alta sensibilidad
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Umbral para 90% de sensibilidad
high_sensitivity_idx = np.where(tpr >= 0.90)[0]
if len(high_sensitivity_idx) > 0:
    high_sens_threshold = thresholds[high_sensitivity_idx[0]]
    print(f"\n=== ANÁLISIS DE UMBRALES ===")
    print(f"Umbral óptimo (Youden): {optimal_threshold:.3f}")
    print(f"Umbral para 90% sensibilidad: {high_sens_threshold:.3f}")
else:
    high_sens_threshold = 0.3
    print(f"Usando umbral conservador: {high_sens_threshold}")

# Guardar modelos y metadatos
model_data = {
    'ensemble': ensemble,
    'scaler': scaler,
    'feature_columns': feature_columns,
    'optimal_threshold': optimal_threshold,
    'high_sensitivity_threshold': high_sens_threshold,
    'test_metrics': {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'ppv': ppv,
        'npv': npv
    },
    'label_encoders': {
        'gender': le_gender,
        'married': le_married,
        'residence': le_residence
    }
}

joblib.dump(model_data, 'medical_stroke_model.pkl')

# FUNCIÓN DE PREDICCIÓN MÉDICA
def predict_stroke_medical(age, hypertension, heart_disease, avg_glucose_level, bmi,
                          gender='Male', ever_married='Yes', residence_type='Urban',
                          work_type='Private', smoking_status='never smoked',
                          sensitivity_mode='balanced'):
    """
    Predicción médica de stroke con diferentes modos de sensibilidad
    
    sensitivity_mode:
    - 'high': Prioriza detectar todos los strokes (más falsos positivos)
    - 'balanced': Balance entre sensibilidad y especificidad  
    - 'conservative': Minimiza falsos positivos
    """
    
    # Validaciones médicas
    if not (0 <= age <= 120):
        raise ValueError("Edad inválida")
    if avg_glucose_level < 50 or avg_glucose_level > 400:
        print(f"⚠️ Glucosa {avg_glucose_level} fuera de rango normal")
    if bmi < 15 or bmi > 50:
        print(f"⚠️ BMI {bmi} fuera de rango normal")
    
    # Codificar variables categóricas
    try:
        gender_enc = le_gender.transform([gender])[0]
        married_enc = le_married.transform([ever_married])[0]  
        residence_enc = le_residence.transform([residence_type])[0]
    except ValueError as e:
        print(f"Error en codificación: {e}")
        # Usar valores por defecto
        gender_enc = 1  # Male
        married_enc = 1  # Yes
        residence_enc = 1  # Urban
    
    # Construir vector de características
    base_features = [age, hypertension, heart_disease, avg_glucose_level, bmi,
                    gender_enc, married_enc, residence_enc]
    
    # Agregar dummies (simplificado para demo)
    work_features = [1 if work_type == 'Private' else 0,
                    1 if work_type == 'Self-employed' else 0,
                    1 if work_type == 'Govt_job' else 0,
                    1 if work_type == 'children' else 0,
                    1 if work_type == 'Never_worked' else 0]
    
    smoke_features = [1 if smoking_status == 'formerly smoked' else 0,
                     1 if smoking_status == 'never smoked' else 0,
                     1 if smoking_status == 'smokes' else 0,
                     1 if smoking_status == 'Unknown' else 0]
    
    # Vector completo
    input_vector = base_features + work_features + smoke_features
    
    # Ajustar longitud si es necesario
    while len(input_vector) < len(feature_columns):
        input_vector.append(0)
    
    input_array = np.array([input_vector[:len(feature_columns)]])
    
    # Escalar y predecir
    input_scaled = scaler.transform(input_array)
    probability = ensemble.predict_proba(input_scaled)[0, 1]
    
    # Seleccionar umbral según modo
    if sensitivity_mode == 'high':
        threshold = high_sens_threshold
    elif sensitivity_mode == 'conservative': 
        threshold = 0.7
    else:  # balanced
        threshold = optimal_threshold
        
    prediction = 1 if probability > threshold else 0
    
    # Clasificación de riesgo médico
    if probability > 0.8:
        risk_level = "Crítico"
        action = "Evaluación médica URGENTE"
    elif probability > 0.6:
        risk_level = "Alto"
        action = "Consultar médico pronto"
    elif probability > 0.4:
        risk_level = "Moderado" 
        action = "Monitoreo regular"
    elif probability > 0.2:
        risk_level = "Bajo-Moderado"
        action = "Prevención básica"
    else:
        risk_level = "Bajo"
        action = "Mantener hábitos saludables"
    
    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_level": risk_level,
        "recommended_action": action,
        "threshold_used": round(threshold, 3),
        "sensitivity_mode": sensitivity_mode,
        "model_performance": f"Sensibilidad: {sensitivity*100:.1f}%, Especificidad: {specificity*100:.1f}%"
    }

# EJEMPLOS DE USO
print("\n=== EJEMPLOS DE PREDICCIÓN MÉDICA ===")

# Caso de alto riesgo
high_risk = predict_stroke_medical(
    age=75, hypertension=1, heart_disease=1,
    avg_glucose_level=250, bmi=35,
    gender='Male', ever_married='Yes',
    work_type='Private', smoking_status='formerly smoked',
    sensitivity_mode='high'
)
print("Caso ALTO RIESGO:", high_risk)

# Caso de bajo riesgo
low_risk = predict_stroke_medical(
    age=25, hypertension=0, heart_disease=0,
    avg_glucose_level=90, bmi=22,
    gender='Female', ever_married='No', 
    work_type='Private', smoking_status='never smoked',
    sensitivity_mode='balanced'
)
print("Caso BAJO RIESGO:", low_risk)

print(f"\n=== RESUMEN DEL MODELO MÉDICO ===")
print(f"✅ Sensibilidad: {sensitivity*100:.1f}% (detecta {sensitivity*100:.1f}% de strokes)")
print(f"✅ Especificidad: {specificity*100:.1f}% (evita {specificity*100:.1f}% de falsos positivos)")
print(f"✅ ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
print(f"✅ Falsos Negativos reducidos: {fn} (vs anteriores)")

recommendation = "APTO PARA ASISTENCIA MÉDICA" if sensitivity > 0.7 and roc_auc_score(y_test, y_pred_proba) > 0.75 else "REQUIERE MEJORAS"
print(f"📋 Recomendación: {recommendation}")