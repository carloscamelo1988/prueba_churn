# 📊 Predicción de Churn de Clientes

## 📌 Descripción
Proyecto de machine learning para predecir churn de clientes usando variables de comportamiento como gasto mensual, número de envíos y antigüedad.

![Distribución de churn](output_29_0.png)
---

## 🧹 Preprocesamiento
- Limpieza de valores nulos
- Creación de variables derivadas (ej: gasto por envío)
- Separación de variables X e y.

---

## 🤖 Modelos utilizados
- Gradient Boosting
  ![Modelo Gradient Boosting](modelo1.png)
  xxx
  ![Importancia de variables](modelo1_1.png)
  
- Random Forest
- Regresión logística (balanced)


Se utilizó validación cruzada estratificada debido al desbalance de clases.

---

## 📊 Resultados
- El mejor modelo fue Random Forest debido a su equilibrio entre recall y F1-score.

---

## 🚀 Cómo ejecutar
pip install -r requirements.txt  
jupyter notebook churn_analysis.ipynb
