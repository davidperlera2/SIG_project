# Detección de Expansión Urbana mediante Machine Learning

## Descripción

Este proyecto implementa un modelo de **Machine Learning** para detectar la expansión urbana mediante el análisis de imágenes satelitales multitemporales.

El modelo combina información óptica proveniente de **Sentinel-2** e información radar **Sentinel-1 (SAR)** para mejorar la clasificación de coberturas terrestres y reducir las limitaciones ocasionadas por la nubosidad.

El proyecto fue desarrollado como trabajo de graduación para optar al título de **Ingeniero Informático** en la **Universidad Centroamericana José Simeón Cañas (UCA)**.

---

## Objetivo

Desarrollar un modelo capaz de detectar automáticamente cambios en la cobertura del suelo mediante técnicas de clasificación supervisada y generar mapas de expansión urbana entre diferentes períodos de tiempo.

---

## Tecnologías utilizadas

- Python 3
- Google Earth Engine
- Rasterio
- NumPy
- Scikit-Learn
- Joblib
- SciPy
- QGIS

---

## Metodología

El flujo general del proyecto es el siguiente:

1. Descarga de imágenes Sentinel-1 y Sentinel-2.
2. Preprocesamiento de imágenes.
3. Generación de índices espectrales.
4. Construcción del Ground Truth.
5. Entrenamiento del modelo Random Forest.
6. Clasificación del territorio.
7. Postprocesamiento del mapa clasificado.
8. Detección de cambios entre años.
9. Generación del mapa de expansión urbana.

---

## Variables utilizadas

### Sentinel-2

- B2 (Blue)
- B3 (Green)
- B4 (Red)
- B8 (NIR)
- B11 (SWIR)

### Índices espectrales

- NDVI
- NDBI
- NDWI

### Sentinel-1

- VV
- VH
- Relación VV/VH

---

## Modelo utilizado

Se empleó un clasificador **Random Forest**, entrenado mediante muestras etiquetadas (Ground Truth) para distinguir tres clases:

- Urbano
- Vegetación
- Agua

---

## Estructura del proyecto

```
SIG_project/
│
├── MODELS/
│   ├── model_sentinel2.py
│   ├── model_hybrid.py
│   └── ...
│
├── PREPROCESS/
│
├── DATA/
│
├── OUTPUT/
│
├── MAPS/
│
└── README.md
```

---

## Resultados


El modelo desarrollado demostró que la integración de imágenes ópticas **Sentinel-2** con datos de radar **Sentinel-1 SAR** mejora significativamente la detección de expansión urbana en comparación con el uso exclusivo de imágenes ópticas.

| Métrica | Resultado |
|---------|----------:|
| Precisión global (Accuracy) | **93.78 %** |
| Coeficiente Kappa | **0.866** |
| Área de estudio | **≈240 km²** |
| Período analizado | **2017 – 2025** |
| Expansión urbana detectada | **1,381.37 hectáreas** |
| Algoritmo | **Random Forest** |
| Datos utilizados | **Sentinel-1 SAR + Sentinel-2** |

---

## Área de estudio

La investigación fue desarrollada sobre una zona de aproximadamente **240 km²** ubicada en El Salvador, comprendiendo sectores de:

- Quezaltepeque
- Colón
- San Juan Opico

Posteriormente se aplicó el modelo para detectar expansión urbana en otras zonas del país.

---

## Cómo ejecutar

Primero descargar las imagenes satelitales ocupando los scripts del directorio DOWNLOAD DATA en la plataforma Google Earth Engine. 

### Instalar dependencias

```bash
pip install rasterio
pip install numpy
pip install scipy
pip install scikit-learn
pip install joblib
```

### Ejecutar el modelo

```bash
python MODELS/model_sentinel2.py
```

o

```bash
python MODELS/model_hybrid.py
```

---


## Autores

- Andrés Felipe Cardona Duarte
- Axel Jared Hernández Servellón
- Moisés Ezequiel Juárez Mejía
- David Misael Perlera Ramírez

Universidad Centroamericana José Simeón Cañas (UCA)

2026

---
