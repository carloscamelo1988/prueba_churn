# Predicción de Churn de Clientes

## Descripción
Proyecto de machine learning para predecir churn (deserción) de clientes a partir de ténicas de machine learning desarrollado en Python.

## Objetivo
Reducir el churn de clientes e identificar las causas que lo genenran.

![Distribución de churn](output_29_0.png)
---
## Metodología
### 🧹 Preprocesamiento
- Limpieza de valores nulos
- Creación de variables derivadas (ej: gasto por envío)
- Separación de variables X e y.

---

### 🤖 Modelos utilizados
- Gradient Boosting.
  
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

![Modelo Gradient Boosting](modelo1.png)

EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE

![Importancia de variables](modelo1_1.png)

- Random Forest
  
IIIIIIIIIIIIIIIIIIIIIIIII

![Random Forest](modelo2.png)

OOOOOOOOOOOOOOOOOOOOOOOOOOOO

![Importancia de variables](modelo2.png)

- Regresión logística (balanced)


Se utilizó validación cruzada estratificada debido al desbalance de clases.

---

###  Resultados
- El mejor modelo fue Random Forest debido a su equilibrio entre recall y F1-score.

---

##  Cómo ejecutar
pip install -r requirements.txt  
jupyter notebook P_Inter.ipynb
