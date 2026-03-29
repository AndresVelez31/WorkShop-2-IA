<div align="center">
 
  <h1>🤖 Workshop 2 - Inteligencia Artificial </h1>
  

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
</div>

---

## 👥 Información del Equipo

| Nombre | Código |
|--------|--------|
| **Andres Felipe Velez Alvarez** | 1021923619 |
| **Nathalia Valentina Cardoza Azuaje** | 4992531 |
| **Sebastian Salazar Henao** | 1022003377 |

**Universidad:** Universidad EAFIT  
**Curso:** Inteligencia Artificial  
**Semestre:** 5to Semestre  
**Fecha:** Marzo 27, 2026

---
## 📋 Descripción del proyecto

Este proyecto académico integra dos ejercicios de Inteligencia Artificial aplicada:

1. **Ejercicio 1 - Clasificación:** detección de fatiga muscular en ciclismo con señales EMG.
2. **Ejercicio 2 - Regresión:** estimación de edad a partir de imágenes faciales usando una CNN.

Ambos ejercicios siguen un flujo completo de trabajo: comprensión del problema, EDA, preprocesamiento, entrenamiento, evaluación y análisis de resultados.

## 🎯 Objetivos del Workshop

### Objetivo general

- Aplicar el ciclo completo de un proyecto de Machine Learning y Deep Learning.
- Construir pipelines reproducibles para datos tabulares derivados de señales (ejercicio 1) e imágenes (ejercicio 2).
- Evaluar el impacto de modelos clásicos vs redes neuronales.
- Medir desempeño con métricas apropiadas para cada tarea.
- Generar evidencia visual y numérica para soportar decisiones de modelado.

## 📂 Estructura del repositorio

```text
WorkShop-2-IA/
├── README.md
├── Clasificacion/
│   ├── clasificacion.ipynb
│   ├── conclusiones/
│   │   └── reporte_resultados.txt
│   ├── graficas_EDA/
│   │   ├── balance_clases.png
│   │   ├── boxplots_clases.png
│   │   ├── correlaciones.png
│   │   ├── distribucion_mdf.png
│   │   ├── distribucion_rms.png
│   │   └── señales_tiempo.png
│   └── graficas_modelos/
│       ├── tabla_comparativa.png
│       ├── comparacion_f1.png
│       ├── confusion_matrix_final.png
│       ├── curva_kNN.png
│       ├── curva_Decision_Tree.png
│       ├── curva_Random_Forest.png
│       ├── curva_Gradient_Boosting.png
│       ├── curva_DNN.png
│       ├── boxplots_predicciones.png
│       └── muestra_artificial.png
└── CNN_Age_Regression/
    ├── data_split.ipynb
    ├── regression.ipynb
    ├── best_age_model.pth
    └── dataset/
        ├── split_log.csv
        ├── train/
        ├── val/
        └── test/
```

## 🗺️ Ejercicio 1: Clasificación de fatiga muscular

Archivo principal: `Clasificacion/clasificacion.ipynb`

### 🎯 Objetivo del ejercicio

Predecir el estado muscular en dos clases:

- `0`: condición normal
- `1`: desgaste muscular

### 🔑 Características principales

- Problema de **clasificación binaria**.
- Datos de entrada transformados desde señales EMG a características por ventanas.
- Comparación de modelos clásicos y un modelo neuronal tipo DNN (MLP).

### 👀 Características específicas del ejercicio

- Pipeline de preprocesamiento con `scikit-learn`.
- Modelos evaluados:
  - kNN
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - DNN (MLPClassifier)
- Ajuste de hiperparámetros.
- Evaluación con `Accuracy`, `Precision`, `Recall`, `F1`.

### 🚀 Cómo Ejecutar

1. Abrir `Clasificacion/clasificacion.ipynb`.
2. Ejecutar celdas en orden desde el inicio.
3. Revisar salidas en:
   - `Clasificacion/graficas_EDA/`
   - `Clasificacion/graficas_modelos/`
   - `Clasificacion/conclusiones/reporte_resultados.txt`

### 📊 Ejemplos de resultados del ejercicio

- Mejor modelo en test por F1: **Random Forest** (`F1=0.8097`).
- Métricas finales (`train+val -> test`):
  - Accuracy: `0.8891`
  - Precision: `0.8522`
  - Recall: `0.7481`
  - F1: `0.7967`
- Matriz de confusión final:
  - `[[303, 17], [33, 98]]`
- Muestra artificial:
  - Predicción: **condición normal (0)**
  - `P(normal)=0.602`, `P(fatiga)=0.398`


## ⏳ Ejercicio 2: Regresión de edad con CNN

✅ Archivo principal:

- `CNN_Age_Regression/regression.ipynb`

### 🎯 Objetivo del ejercicio

Estimar edad (variable continua) desde imágenes faciales.

### 🔑 Características principales

- Problema de **regresión supervisada**.
- Entrada de alta dimensionalidad (imágenes).
- Uso de red neuronal convolucional (CNN) y entrenamiento con PyTorch.

### 👀 Características específicas del ejercicio

- Parámetros de entrenamiento usados en notebook:
	- `IMG_SIZE = 64`
	- `BATCH_SIZE = 32`
	- `NUM_EPOCHS = 25`
	- `LR = 1e-3`
- Optimizador Adam.
- Scheduler de reducción de LR en meseta.
- Seguimiento de `train/val loss` y `train/val MAE`.
- Checkpoint final: `CNN_Age_Regression/best_age_model.pth`.

### 🚀 Cómo ejecutar

1. Abrir y ejecutar:
   1. `CNN_Age_Regression/regression.ipynb`
2. Verificar que exista `CNN_Age_Regression/dataset/` con `train`, `val`, `test`.
3. Confirmar guardado de modelo en `best_age_model.pth`.

### 📊 Ejemplos de resultados del ejercicio

- Partición documentada en `CNN_Age_Regression/dataset/split_log.csv`:
  - `train`: `16874`
  - `val`: `3615`
  - `test`: `3617`
- El notebook genera curvas de entrenamiento y validación para analizar ajuste del modelo.
- Se obtiene y guarda un checkpoint final (`best_age_model.pth`).
- En el análisis exploratorio del notebook se identifica sesgo de edades hacia rangos jóvenes/adultos y menor representación en edades altas.

## 🛠️ Tecnologías y dependencias utilizadas

### Tecnologias

- Python
- Jupyter Notebook
- Scikit-learn
- PyTorch

### Librerias principales

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `torch`
- `torchvision`

## 🚀 Requerimientos de instalación y ejecución

### Requisitos

- Python 3.10 o superior
- `pip`
- Entorno con soporte de notebooks

### Instalacion sugerida

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install jupyter numpy pandas matplotlib seaborn scikit-learn torch torchvision
```

### Ejecución del proyecto

```bash
jupyter notebook
```

Orden recomendado de ejecución:

1. `Clasificacion/clasificacion.ipynb`
2. `CNN_Age_Regression/data_split.ipynb`
3. `CNN_Age_Regression/age_regression_dataloader (1).ipynb`
4. `CNN_Age_Regression/regression.ipynb`

## 🎓 Aprendizajes Clave
### 🗺️ Ejercicio 1: Clasificación de fatiga muscular
- La calidad del feature engineering en señales impacta fuertemente la separabilidad de clases.
- Modelos de ensamble (Random Forest/Gradient Boosting) muestran mejor balance entre generalización y robustez.
- El F1-score es clave cuando interesa balance entre precisión y recall.

### ⏳ Ejercicio 2: Regresión de edad con CNN
- En imágenes, la calidad del split y del pipeline de carga impacta tanto como la arquitectura.
- El sesgo en la distribución del target puede degradar precisión en extremos del rango de edad.
- El monitoreo simultaneo de loss y MAE evita optimizar solo por una senal parcial.

## 🔬 Análisis comparativo

| Dimensión | Ejercicio 1: Clasificación | Ejercicio 2: Regresión |
|---|---|---|
| Tipo de problema | Clasificación binaria | Regresión continua |
| Tipo de datos | Características derivadas de señales | Imágenes faciales |
| Complejidad de entrada | Media (tabular) | Alta (tensor imagen) |
| Modelos | Clasicos + DNN MLP | CNN |
| Métricas clave | Accuracy, Precision, Recall, F1 | MAE, RMSE, R2 |
| Riesgo principal | Errores de detección de fatiga | Sesgo por distribución de edad |
| Costo computacional | Moderado | Alto |

### 📖 Interpretación comparativa

- El ejercicio de clasificación es más interpretable y rápido de iterar.
- El ejercicio de regresión requiere mayor capacidad de cómputo y mayor cuidado en preprocesamiento visual.
- Ambos ejercicios muestran que la preparación del dato es determinante para el desempeño final.

## ⚡️ Conclusiones finales del proyecto

- Se cubrieron dos escenarios complementarios de IA supervisada, con metodologías distintas pero compatibles.
- El proyecto demuestra trazabilidad de resultados (reportes, gráficas, checkpoint) y capacidad de reproducción en notebooks.
- La comparación entre ejercicios evidencia que no existe un único pipeline universal: la naturaleza del dato y del target define la estrategia de modelado.

### 🗺️ Ejercicio 1: Clasificación de fatiga muscular
- El enfoque más sólido en este trabajo fue Random Forest.
- Existe margen de mejora en recall (detección de fatiga), potencialmente vía re-balanceo, nuevas características y/o ajuste de umbral.
### Ejercicio 2: Regresión de edad con CNN
- La CNN permite capturar patrones visuales útiles para regresión de edad.
- El principal riesgo técnico es el desbalance del target por edad, por lo que estrategias de muestreo o pérdidas ponderadas pueden mejorar resultados.
