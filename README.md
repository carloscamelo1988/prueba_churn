# Predicción de Churn de Clientes

## Descripción
Proyecto de machine learning para predecir churn (deserción) de clientes a partir de ténicas de machine learning desarrollado en Python.

## Objetivo
Reducir el churn de clientes e identificar las causas que lo genenran.

![Distribución de churn](output_29_0.png)
---
## Metodología
### Preprocesamiento
- Fuente datos:raw_data_customers.csv, consta de 114 registros y 10 varibles. En términos generales se tienen variables que caracterizan a los clientes como por ejemplo fecha de inscripción, fecha de última compra, gasto mensual, total de envíos y la variable churn (con valores de 1: desertó y 0: no desertó).

![Caracterización de la fuente de datos](Estructura1.png)
  
- Limpieza de valores nulos: Se realizó limpieza de valores adicionando 0 las variables monthly_spend y total_shipments. Además, aquellso registros que reportaban con valor de 99999 fue reemplazado por 0.
- Transfomación de datos: 
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
