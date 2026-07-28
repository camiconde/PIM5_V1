# src/model_traning.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

# Métricas de evaluación
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# Importamos la lógica del pipeline que acabamos de validar
from ft_engineering import X_train, X_test, y_train, y_test, preprocessor


# =====================================================================
# 1. FUNCIONES REQUERIDAS POR LA CONSIGNA
# =====================================================================

def build_model(classifier, preprocessor):
    """
    Ensambla el preprocesador (ColumnTransformer) con el algoritmo de clasificación en un Pipeline único.
    """
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    return model_pipeline


def summarize_classification(y_true, y_pred, y_proba=None, model_name="Modelo"):
    """
    Calcula y retorna un diccionario con las métricas principales de evaluación.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan

    return {
        'Modelo': model_name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(auc, 4)
    }


# =====================================================================
# 2. ENTRENAMIENTO Y EVALUACIÓN DE MULTIPLES MODELOS
# =====================================================================

if __name__ == "__main__":
    print("=== INICIANDO ENTRENAMIENTO DE MODELOS SUPERVISADOS ===\n")

    # Definimos los 3 modelos que vamos a entrenar y comparar
    candidates = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    results = []
    trained_models = {}

    for name, classifier in candidates.items():
        print(f"Entrenando {name}...")
        
        # 1. Construir el pipeline
        model = build_model(classifier, preprocessor)
        
        # 2. Entrenar el modelo
        model.fit(X_train, y_train)
        trained_models[name] = model

        # 3. Predicciones en el conjunto de prueba (X_test)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # 4. Resumen de métricas
        metrics = summarize_classification(y_test, y_pred, y_proba, model_name=name)
        results.append(metrics)

    # =====================================================================
    # 3. TABLA COMPARATIVA Y SELECCIÓN DEL MEJOR MODELO
    # =====================================================================
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("=== TABLA RESUMEN DE EVALUACIÓN ===")
    print("="*50)
    print(df_results.to_string(index=False))
    print("="*50)

    # Seleccionamos el ganador según el F1-Score (idóneo para clases desbalanceadas como mora/pago)
    best_model_row = df_results.sort_values(by='F1-Score', ascending=False).iloc[0]
    print(f"\n🏆 MEJOR MODELO SELECCIONADO: {best_model_row['Modelo']} (F1-Score: {best_model_row['F1-Score']})")

# Guardamos el mejor modelo (Random Forest)
    os.makedirs('models', exist_ok=True)
    best_model = trained_models['Random Forest']

    joblib.dump(best_model, 'models/model.pkl')
    print("\n💾 ¡El mejor modelo ha sido guardado exitosamente en 'models/model.pkl'!")