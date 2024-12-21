#!/usr/bin/env python
# coding: utf-8

# # AIML- INCAF-1
# ## Módulo 3: Diseñar e implementar modelos
# ## Actividad 3. Despliegue de un modelo de Machine Learning en un entorno de producción
# #### Elaborado por Gabriel Guzmán
# 
# Se ha preparado un modelo para el despliegue en un entorno de producción, asegurándose de incluir los pasos de limpieza y transformación de un conjunto de datos financieros.

# ### 1. Carga Información, depuración y limpieza de datos

# In[1]:


#Importamos librerias
import logging
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_curve, auc,precision_recall_curve, average_precision_score

# Configuración básica de log
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app_AIMLK_M03T03.log',  # Guardar logs en un archivo
    filemode='w'  # Sobreescribe el archivo en cada ejecución
)

logging.info("# Generando semilla\n")
np.random.seed(42)

logging.info("# Cargar el conjunto de datos\n")

# Cargar el conjunto de datos
train_data = pd.read_csv('./load-data.csv')

datos = train_data

df = pd.DataFrame(datos)

logging.info("#Mostrando datos originales:\n")
logging.info(df)

logging.info('\nInformación de Columnas:\n')
logging.info(df.info())

logging.info('# Creamos una copia para no modificar los datos originales')
df_limpio = df.copy()

logging.info('# 1. Manejo de valores faltantes')
logging.info('# Rellenamos valores faltantes en columnas numéricas con la mediana')
columnas_numericas = df_limpio.select_dtypes(include=['float64', 'int64']).columns
for columna in columnas_numericas:
    if (df_limpio[columna].isnull().sum() > 0) and (columna != 'HasCrCard'):
        logging.info(f"columna= {columna}")
        mediana = df_limpio[columna].median()
        #df_limpio[columna].fillna(mediana, inplace=True)
        df_limpio[columna] = df_limpio[columna].fillna(mediana)
        logging.debug(f"- Valores faltantes en '{columna}' rellenados con la mediana: {mediana}")

logging.info('# 2. Eliminación de duplicados')
duplicados = df_limpio.duplicated().sum()
df_limpio.drop_duplicates(inplace=True)
logging.debug(f"- Se eliminaron {duplicados} registros duplicados")

logging.info('# 3. Estandarización de texto')
columnas_texto = df_limpio.select_dtypes(include=['object']).columns
for columna in columnas_texto:
    # Convertimos a minúsculas y eliminamos espacios externos
    df_limpio[columna] = df_limpio[columna].str.lower().str.strip()
    logging.debug(f"- Columna '{columna}' estandarizada a minúsculas y sin espacios externos")

logging.info('# 4. Manejo de valores atípicos (outliers)')
for columna in columnas_numericas:
    if columna != 'HasCrCard':
        Q1 = df_limpio[columna].quantile(0.25)
        Q3 = df_limpio[columna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
    
        # Identificamos outliers
        outliers = df_limpio[(df_limpio[columna] < limite_inferior) | 
                            (df_limpio[columna] > limite_superior)][columna]
        
        if len(outliers) > 0:
            # Recortamos los valores atípicos a los límites
            df_limpio[columna] = df_limpio[columna].clip(limite_inferior, limite_superior)
            logging.debug(f"- Se encontraron y trataron {len(outliers)} valores atípicos en '{columna}'")


logging.info('# 6. Normalización de datos numéricos')
for columna in columnas_numericas:
    # Aplicamos normalización Min-Max
    min_val = df_limpio[columna].min()
    max_val = df_limpio[columna].max()
    df_limpio[f'{columna}_normalizado'] = ((df_limpio[columna] - min_val) / 
                                          (max_val - min_val))
    logging.debug(f"- Columna '{columna}' normalizada entre 0 y 1")

logging.info('# 7. Conversión de tipos de datos')
logging.info('# Preparando el formato de fecha')
date_format = '%Y-%m-%d'

logging.info('# Convertir columnas de fecha')
for columna in columnas_texto:
    try:
        df_limpio[f'{columna}_fecha'] = pd.to_datetime(df_limpio[columna], format=date_format)
        logging.info(f"- Columna '{columna}' convertida a formato fecha")
    except:
        logging.debug(f"- La Columna '{columna}' no es de tipo fecha")
        continue




# ### 2. Entrenamiento del modelo y valida con datos de prueba.

# In[2]:


logging.info("# 8. Seleccionar columnas necesarias\n")
final_dataset = df_limpio[['Geography', 'Gender', 'Age', 'NumOfProducts', 'EstimatedSalary','HasCrCard']]


logging.info("# 9. Separar las características (X) y la variable objetivo (y)")
print("Distribución de la variable objetivo (HasCrCard):")
print(final_dataset['HasCrCard'].value_counts())
print(final_dataset['HasCrCard'].unique())
X = final_dataset.drop(columns=['HasCrCard'])
y = final_dataset['HasCrCard'].apply(lambda x: 1 if x == 1. else 0) # Convertir 'si' a 1 y 'no' a 0 para SMOTE
#y = final_dataset['HasCrCard']
print(y.value_counts())


logging.info("# 10. Aplicar One-Hot Encoding a las columnas categóricas\n")
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Cambiamos sparse a sparse_output
X_encoded = encoder.fit_transform(X[['Geography', 'Gender', 'Age']])
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['Geography', 'Gender', 'Age']))

logging.info("# 11. Combinar columnas codificadas con las numéricas\n")
X_numeric = X[['NumOfProducts', 'EstimatedSalary']].reset_index(drop=True)
X_final = pd.concat([X_numeric, X_encoded_df], axis=1)

logging.info("# 12. Dividir datos en entrenamiento y prueba\n")
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=42, stratify=y)

logging.info("# 13. Aplicar SMOTE para balancear clases\n")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("# Distribución después de SMOTE:\n")
print(y_train_balanced.value_counts())





# ### 3. Calcula y analiza las métricas de rendimiento (precisión, recall, F1-score, etc.).

# In[3]:


print('# XGBoost import XGBClassifier:\n')
print('# Hyperparametros:')
print('n_estimators=500: Número de árboles en el modelo.')
print('random_state=42: Fija la semilla para reproducibilidad.')
print('max_depth=3:  Profundidad máxima de cada árbol.')
print('eval_metric=''logloss'': Métrica de pérdida logarítmica usada internamente durante el entrenamiento.')
xgb_model = XGBClassifier(n_estimators=500, random_state=42, max_depth=3, eval_metric='logloss')
xgb_model.fit(X_train_balanced, y_train_balanced)

print('# Predicciones en el conjunto de prueba\n')
y_pred = xgb_model.predict(X_test)

print(f"# Evaluación del modelo\n")
print(f"Accuracy del modelo: {accuracy_score(y_test, y_pred)}")
print(f"\nClassification Report:\n {classification_report(y_test, y_pred, zero_division=1)}")


# ### 4. Realiza ajustes de hiperparámetros.

# In[4]:


print(f"# Entrenar el modelo XGBoost Mejores Parametros")
eval_set = [(X_train_balanced, y_train_balanced), (X_test, y_test)]

print('# Ajustando Hyperparametros:')
print('----------------------------')
print('n_estimators=300: Número de árboles en el modelo.')
print('random_state=42: Fija la semilla para reproducibilidad.')
print('max_depth=12:  Profundidad máxima de cada árbol.')
print('eval_metric=''logloss'': Métrica de pérdida logarítmica usada internamente durante el entrenamiento.')
print('learning_rate=0.01: Asegura la tasa de aprendizaje')
print('colsample_bytree=0.8: Parametro de muestreo para delimitar el árbol.')
print('subsample=0.6: Fracción de muestra usada para cada árbol.')

xgb_model = XGBClassifier(n_estimators=300, random_state=42, max_depth=8, eval_metric='logloss', 
                          learning_rate=0.01, colsample_bytree=0.8, subsample=0.6)

xgb_model.fit(X_train_balanced, y_train_balanced, eval_set=eval_set, verbose=False)

print(f"Predicciones")
y_pred = xgb_model.predict(X_test)

print(f"Accuracy del modelo: {accuracy_score(y_test, y_pred)}")
print(f"\nClassification Report:\n {classification_report(y_test, y_pred, zero_division=1)}")


# ### Conclusión
# 
# Nuestro modelo predecir un 75% de las veces si el cliente aceptará la tarjeta de crédito, sumado a una precisión de 75% y 100% en la clase 1 (recall) la cual es la que nos interesa precedir para que tome la tarjeta de crédito.
