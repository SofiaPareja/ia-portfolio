---
title: "Regresión Lineal y Logística: Comparación y Aplicación"
date: 2025-09-15
---

# Regresión Lineal y Logística: Comparación y Aplicación

## Contexto
La regresión lineal y la regresión logística son dos modelos fundamentales en aprendizaje supervisado. La primera se orienta a problemas de predicción de valores continuos, mientras que la segunda aborda problemas de clasificación binaria o multiclase. En esta práctica se exploraron ambos enfoques con aplicaciones prácticas:  
- Regresión lineal aplicada a la predicción de valores numéricos.  
- Regresión logística aplicada a la clasificación médica (diagnóstico de tumores).  

## Objetivos
- Comprender el funcionamiento y los casos de uso de la regresión lineal y logística.  
- Analizar las métricas de evaluación más relevantes para cada tipo de modelo.  
- Implementar ejemplos prácticos y reflexionar sobre la elección del modelo según el tipo de problema.  

## Actividades (con tiempos estimados)

| Actividad                          | Tiempo | Resultado esperado                          |
|------------------------------------|:------:|---------------------------------------------|
| Revisión teórica de métricas        | 20m    | Definiciones claras de MAE, RMSE, R², MAPE y métricas de clasificación |
| Implementación de regresión lineal | 30m    | Modelo entrenado y evaluación con métricas  |
| Implementación de regresión logística | 40m  | Modelo aplicado a clasificación médica      |
| Comparación de resultados           | 20m    | Tabla comparativa de diferencias            |
| Reflexión final                     | 15m    | Selección de modelo adecuado según contexto |

## Desarrollo
En la primera parte se revisaron las métricas de error para regresión:  
- **MAE** mide el error promedio en valor absoluto.  
- **MSE** y **RMSE** penalizan más los errores grandes.  
- **R²** evalúa la proporción de varianza explicada.  
- **MAPE** expresa el error en porcentaje, útil en escalas diferentes.  

**Regresión Logística:**  
- Aplicada a la clasificación de tumores.  
- Métricas principales:  
  - **Accuracy:** porcentaje de aciertos.  
  - **Precision:** predicciones positivas correctas.  
  - **Recall:** detección de positivos reales.  
  - **F1-score:** balance entre precision y recall.  
  - **Matriz de confusión:** distribución de aciertos y errores.  
- Se destacó el riesgo de **falsos negativos** en diagnóstico médico (clasificar un tumor maligno como benigno).  

Se elaboró una **tabla comparativa** entre regresión lineal y logística, resaltando:  
- La regresión lineal predice valores continuos (ej. precio de una casa).  
- La regresión logística predice categorías (ej. tumor benigno/maligno).  
- Cada modelo tiene métricas específicas y rangos de salida diferentes.  

## Evidencias
- Tabla comparativa entre regresión lineal y logística:  

| **Aspecto**        | **Regresión Lineal**                | **Regresión Logística**                     |
|---------------------|-------------------------------------|---------------------------------------------|
| Qué predice         | Valores numéricos continuos         | Categorías (clases)                         |
| Ejemplo de uso      | Precio de una casa                  | Tumor maligno/benigno                       |
| Rango de salida     | -∞ a +∞                             | Probabilidades entre 0 y 1                  |
| Métricas principales| MAE, RMSE, R²                       | Accuracy, Precision, Recall, F1, AUC        |


```python linenums="1"
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report

# Ejemplo de regresión lineal
lin_model = LinearRegression().fit(X_train_reg, y_train_reg)
y_pred_reg = lin_model.predict(X_test_reg)
print("MAE:", mean_absolute_error(y_test_reg, y_pred_reg))

# Ejemplo de regresión logística
log_model = LogisticRegression(max_iter=200, solver="liblinear").fit(X_train_clf, y_train_clf)
y_pred_clf = log_model.predict(X_test_clf)
print(classification_report(y_test_clf, y_pred_clf))
```



!!! note "Nota"
    En el contexto médico, los **falsos negativos** (clasificar un tumor maligno como benigno) son más peligrosos que los falsos positivos.  

 

???+ info "Reflexiones principales"
    - La regresión lineal es adecuada para problemas numéricos continuos (ej. salarios).  
    - La regresión logística es adecuada para problemas de clasificación binaria (ej. spam vs no spam).  
    - Separar datos de entrenamiento y prueba es esencial para evaluar la capacidad de generalización.  

## Reflexión
La práctica permitió reforzar la diferencia conceptual y técnica entre regresión lineal y logística. La elección del modelo depende del **tipo de variable objetivo**: continua o categórica. Además, se comprendió la importancia de seleccionar las **métricas de evaluación correctas** según el problema.  

Aprendí que, en problemas sensibles como la salud, no basta con maximizar la exactitud: es crucial analizar falsos positivos y falsos negativos. Como mejora futura, se podría extender el análisis a modelos no lineales y probar regularización (L1/L2) en la regresión logística.  

## Checklist
- [x] Definición de métricas de regresión  
- [x] Implementación de regresión lineal  
- [x] Definición de métricas de clasificación  
- [x] Implementación de regresión logística  
- [x] Comparación de ambos modelos  

## Referencias
- Documentación de scikit-learn: [https://scikit-learn.org](https://scikit-learn.org)  
- Apuntes de clase sobre modelos de regresión y clasificación.  
- Material complementario de métricas de error y clasificación.  
