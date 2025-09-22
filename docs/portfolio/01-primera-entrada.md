---
title: "Tarea 5: Validación y Selección de Modelos - Fill in the Blanks"
date: 2025-08-26
---

# Entrada 01 — Mi primera experiencia

## Contexto
Análisis del dataset **Student Dropout and Academic Success** (UCI).  
Se trabajó un pipeline completo: exploración de datos, entrenamiento de modelos de clasificación (Logistic Regression, Ridge, Random Forest), validación cruzada, optimización de hiperparámetros y análisis de explicabilidad.

## Objetivos
- Aprender a prevenir data leakage usando pipelines
- Implementar validación cruzada (cross-validation) robusta
- Comparar múltiples modelos de forma sistemática
- Interpretar métricas de estabilidad y selección de modelos

## Dataset: Predicción de Éxito Estudiantil
- Exploración de dataset — 40 min  
- Entrenamiento de modelos base — 60 min  
- Validación cruzada y competencia de modelos — 50 min  
- Optimización de hiperparámetros (GridSearch/RandomSearch) — 45 min  
- Análisis de explicabilidad con Random Forest — 60 min  
- Reflexiones y redacción de portafolio — 30 min  

## Desarrollo
1. **EDA inicial:** 4424 estudiantes, 36 variables (académicas, demográficas, económicas).  
   Las clases estaban desbalanceadas: *Graduate > Dropout > Enrolled*.  
2. **Modelos probados:** Logistic Regression, Ridge Classifier, Random Forest.  
   - Validación cruzada (5-fold, estratificada) para estabilidad.  
   - Métrica: Accuracy promedio ± desviación estándar.  
3. **Resultados:**  
   - Logistic Regression: estable y buen rendimiento.  
   - Random Forest: mayor exactitud y buena estabilidad.  
   - Ridge: intermedio.  
4. **Optimización:**  
   - Se usó GridSearchCV (exhaustivo) y RandomizedSearchCV (eficiente).  
   - Random Forest optimizado alcanzó mejores resultados en accuracy.  
5. **Explicabilidad:**  
   - Se identificaron variables clave: calificaciones previas, unidades curriculares aprobadas, edad de ingreso y becas.  
   - Se calcularon importancias por categorías: académicas > económicas > demográficas.  
   - Se exploraron reglas de decisión de árboles individuales y predicciones de estudiantes concretos.  


## Evidencias
- Enlace a material o capturas en `docs/assets/`

## Reflexión
Lo más desafiante, lo más valioso, próximos pasos.
