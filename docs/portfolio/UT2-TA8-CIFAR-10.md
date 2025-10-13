---
title: "Explorando CIFAR-10: optimización de una red neuronal densa con Adam y GELU"
date: 2025-10-05
---

## Contexto
En esta práctica se trabajó con el dataset **CIFAR-10**, un conjunto de imágenes a color de 32x32 píxeles que contiene 10 clases de objetos.  
El objetivo fue **entrenar y optimizar una red neuronal multicapa (MLP)** capaz de clasificar las imágenes, aplicando técnicas de *backpropagation* y comparando diferentes optimizadores, funciones de activación y configuraciones de arquitectura.

---

## Objetivos
- Entrenar un modelo base con Keras sobre CIFAR-10.  
- Probar variaciones en la arquitectura: número de capas, cantidad de neuronas y función de activación.  
- Evaluar el impacto de distintos optimizadores (Adam, RMSprop) y tasas de aprendizaje.  
- Implementar *callbacks* para mejorar la eficiencia del entrenamiento (EarlyStopping, TensorBoard).  
- Seleccionar la configuración más efectiva en términos de **accuracy y generalización**.

---

## Actividades (con tiempos estimados)
- Preparación y normalización del dataset — 30 min  
- Creación del modelo base — 20 min  
- Experimentos con capas y activaciones — 60 min  
- Pruebas de optimizadores y *learning rate scheduling* — 60 min  
- Análisis de resultados y comparación final — 40 min  

---

## Desarrollo
El modelo inicial obtuvo una precisión de **≈ 45 %** en test con dos capas densas.  
A partir de ahí se realizaron iteraciones experimentales:

| Experimento | Configuración principal | Test Acc (%) | Observaciones |
|--------------|------------------------|--------------|----------------|
| Base         | 2×128 ReLU + Adam (1e-3) | 44.9 | Modelo simple, subajustado |
| + Más capas  | 3×128 ReLU + Dropout 0.4 | 50.4 | Ligera mejora |
| + GELU       | 3×128 GELU + Adam (5e-4) | 52.3 | Mejor convergencia |
| + Más neuronas | 256-256-128 GELU + Adam (3e-4) | **55.2** | Mejor resultado global |
| RMSprop      | Igual arquitectura + RMSprop (7e-4) | 52.8 | Peor desempeño, menor estabilidad |

Finalmente, se incorporaron *callbacks*:
- `EarlyStopping` para detener el entrenamiento cuando `val_accuracy` se estancaba.  
- `LearningRateScheduler` con **cosine decay**, que permitió una convergencia más suave.  

Con estas técnicas, el modelo alcanzó un **Training Accuracy ≈ 69 %** y **Test Accuracy ≈ 55 %**, mostrando una mejora consistente sin overfitting severo.

---
## Evidencias

Durante el proceso se llevaron a cabo múltiples iteraciones experimentales.
A continuación se resumen los resultados más relevantes:

### Curvas de entrenamiento

Se analizaron las curvas de loss y accuracy en TensorBoard para observar:

Convergencia progresiva con el optimizador Adam.

Mejora de la val_accuracy hasta estabilizar en torno al 55 %.

Ausencia de sobreajuste marcado gracias a Dropout y EarlyStopping.

**Configuración final del modelo**

```python
model = keras.Sequential([
    keras.Input(shape=(32*32*3,)),
    layers.Dense(256, activation="gelu"),
    layers.Dropout(0.5),
    layers.Dense(256, activation="gelu"),
    layers.Dropout(0.5),
    layers.Dense(128, activation="gelu"),
    layers.Dropout(0.4),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

```
**Resultados finales**

Training Accuracy: 69.3 %

Test Accuracy: 55.2 %

Total parámetros: 886,666
## Reflexión

Este proceso permitió entender cómo pequeñas variaciones en la arquitectura y el optimizador pueden impactar significativamente el rendimiento.
GELU resultó más estable que ReLU en este contexto, y Adam con cosine decay ofreció el mejor equilibrio entre velocidad de convergencia y generalización.
Aprendí a experimentar sistemáticamente, interpretar curvas de loss/accuracy y aplicar callbacks para optimizar entrenamientos de forma automática.

