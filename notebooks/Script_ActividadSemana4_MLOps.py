Script_ActividadSemana4_MLOps
# %% [markdown]
# Operaciones de aprendizaje automático
# Semana 3/4 | Actividad | Dataset - Notebook | Individual
# 
# Daniela Hernández López
# A01840424

# %%
#Se importan todas las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

%pip install seaborn
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, OrdinalEncoder, OneHotEncoder, StandardScaler

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, classification_report, f1_score, precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay





# %%
#Se configura la URI para MLflow
#Primero iniciar la interfaz de usuario de MLflow en la terminal con el comando: mlflow ui
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Notebook-Baseline-Sem4")

# %%
# Se utilizará el dataset de South German Credit para este proyecto. 
# Se puede descargar desde el siguiente enlace: https://archive.ics.uci.edu/dataset/522/south+german+credit
df = pd.read_csv(
    r"c:\Users\ingda\Documents\5_Tec Maestría IA\7_MLOps\Semana 4\Actividad_Semana4\southgermancredit_project\data\raw\SouthGermanCredit.asc",
    sep=r"\s+",
    engine="python"
)

print("Forma del dataset")
print(df.shape)

# %%
# Renombramos los nombres de las columnas del alemán al inglés y desplegamos de
# nuevo el DataFrame para ver el resultado obtenido:

column_mapping = {
    "laufkont": "status",
    "laufzeit": "duration",
    "moral": "credit_history",
    "verw": "purpose",
    "hoehe": "amount",
    "sparkont": "savings",
    "beszeit": "employment_duration",
    "rate": "installment_rate",
    "famges": "personal_status_sex",
    "buerge": "other_debtors",
    "wohnzeit": "present_residence",
    "verm": "property",
    "alter": "age",
    "weitkred": "other_installment_plans",
    "wohn": "housing",
    "bishkred": "number_credits",
    "beruf": "job",
    "pers": "people_liable",
    "telef": "telephone",
    "gastarb": "foreign_worker",
    "kredit": "credit_risk"
}

df.rename(columns=column_mapping, inplace=True)
df.head().T

# %%
# Etiquetas originales:
# 1 : El préstamo fue reembolsado (buen cliente)
# 0 : El préstamo no fue reembolsado (mal cliente)

# Se cambiará la etiqueta a la variable objetivo para que sea más intuitiva, de la siguiente manera:
# 1 : El préstamo no fue reembolsado (mal cliente)
# 0 : El préstamo fue reembolsado (buen cliente)

df["credit_risk"] = np.where(df["credit_risk"] == 1, 0, 1)

print(df["credit_risk"].value_counts())

# %%
print("\n -- INFO --")
print(df.info())

# %%
# Valores faltantes
print("\nValores faltantes")
print(df.isnull().sum())

# %%
#Valores duplicados
print("Duplicados:", df.duplicated().sum())

# %% [markdown]
# Aunque se indica realizar la división entrenamiento/prueba en el Paso 4, se realiza previamente una partición estratificada de los datos para evitar data leakage. El conjunto de prueba se mantiene aislado de las decisiones de limpieza, transformación y selección del modelo. El EDA descriptivo inicial se realizó sobre el dataset completo, mientras que cualquier análisis utilizado para definir transformaciones o parámetros del modelo se realiza exclusivamente sobre el conjunto de entrenamiento.

# %%
# Particionamos los datos de entrada (X) y de salida (y):
X,y = df.iloc[:,:-1], df.iloc[:,-1]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, 
                                                y, 
                                                train_size=0.7, 
                                                stratify=y, 
                                                shuffle=True,
                                                random_state=42)

# Mostramos las dimensiones de la partición generada:
print('X, y Train:')
print(Xtrain.shape)
print('\nX, y Test:')
print(Xtest.shape)

# El umbral de referencia sería el valor de la clase mayoritaria, en este caso 70%.
# La clase mayoritaria constituye un baseline ingenuo.
# Si la clase mayoritaria representa aproximadamente el 70% de los casos, un clasificador que siempre predijera dicha clase tendría un accuracy
# de aproximadamente 70%. Este valor sirve como referencia para comparar el desempeño del modelo.

# %%
# Separamos las variables para poder usar diferentes pipelines de preprocesamiento para cada tipo de variable. 

# Variables numéricas:
list_paper_num = 'duration', 'amount', 'age', 'people_liable'

# Variables ordinales:
list_paper_ord = 'employment_duration', 'installment_rate', 'present_residence', 'property', 'number_credits', 'job'

# Variables nominales/categóricas:
list_paper_cat = 'status', 'credit_history', 'purpose', 'savings', 'personal_status_sex', 'other_debtors', 'other_installment_plans', 'housing', 'telephone', 'foreign_worker'
#no se incluye credit_risk por ser la variable de salida y

# %%
#-- EDA --
# Numéricas
print('\nNuméricas:')

for col in list_paper_num:
    plt.figure(figsize=(3, 2))
    sns.histplot(Xtrain[col], kde=True)
    plt.title(f"Distribución de {col}")
    plt.show()

    print(f"\n{col}")
    print(Xtrain[col].describe().T)
    print('\n')

    counts = Xtrain[col].value_counts()
    perc = Xtrain[col].value_counts(normalize=True) * 100
    summary = pd.DataFrame({"count": counts, "percentage": perc.round(2)})
    print(summary)
    print("-"*150)

# %%
# Ordinales
print("\n Variables Ordinales")
for col in list_paper_ord:
    plt.figure(figsize=(3, 2))
    plt.title(f"Frecuencia de {col}")
    sns.countplot(x=Xtrain[col])
    plt.show()

    counts = Xtrain[col].value_counts()
    perc = Xtrain[col].value_counts(normalize=True) * 100
    summary = pd.DataFrame({"count": counts, "percentage": perc.round(2)})
    print(summary)
    print("-"*150)


# %%
# Nominales / Categóricas
print('\nCategóricas:')

for col in list_paper_cat:
    plt.figure(figsize=(3, 2))
    sns.countplot(y=Xtrain[col], order=Xtrain[col].value_counts().index)
    plt.title(f"Frecuencia de {col}")
    plt.show()

    counts = Xtrain[col].value_counts()
    perc = Xtrain[col].value_counts(normalize=True) * 100
    summary = pd.DataFrame({"count": counts, "percentage": perc.round(2)})
    print(summary)
    print("-"*150)

# %%
# Unimos las variables predictoras de TRAIN con la variable objetivo
train_eda = Xtrain.copy()
train_eda["credit_risk"] = ytrain

# 0 = buen crédito / 1 = mal crédito
train_eda.head()

# %%
# Boxplots de variables numéricas respecto a la variable objetivo

fig, axes = plt.subplots(2, 2, figsize=(7, 7))

for ax, col in zip(axes.flatten(), list_paper_num):
    sns.boxplot(
        data=train_eda,
        x="credit_risk",
        y=col,
        ax=ax
    )
    
    ax.set_title(f"{col} vs. credit_risk")
    ax.set_xlabel("Riesgo de crédito (0 = bueno, 1 = malo)")
    ax.set_ylabel(col)

plt.tight_layout()
plt.show()

# %%
# Proporción de créditos de riesgo (credit_risk = 1)
# dentro de cada categoría

for col in list_paper_cat:
    
    riesgo_por_categoria = (
        train_eda.groupby(col)["credit_risk"]
        .mean()
        .sort_values(ascending=False)
    )
    
    plt.figure(figsize=(5, 3))
    
    sns.barplot(
        x=riesgo_por_categoria.index,
        y=riesgo_por_categoria.values
    )
    
    plt.title(f"Proporción de mal crédito según {col}")
    plt.xlabel(col)
    plt.ylabel("Proporción de credit_risk = 1")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

# %%
# Promedio de variables numéricas según la clase de riesgo

list_num = ["duration", "amount", "age", "people_liable"]

train_eda.groupby("credit_risk")[list_num].agg(
    ["mean", "median", "std"]
)

# %%
# Proporción de mal crédito por categoría

for col in list_paper_cat:
    print(f"\n--- {col} ---")
    
    resumen = (
        train_eda.groupby(col)["credit_risk"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )
    
    resumen["percentage_risk"] = resumen["mean"] * 100
    
    print(resumen)

# %%
# Transformaciones y limpieza

# Variables numéricas:
num_pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")), #para rellenar los valores faltantes con la mediana de cada variable
                                                                   # se usa la mediana en lugar de la media para evitar que los valores atípicos influyan en el valor de imputación
                    ("yeojohnson", PowerTransformer(method="yeo-johnson")), #modifica la distribución original para aproximarla a una distribución más simétrica
                                                                            #aquí los outliers no son un problema, ya que el PowerTransformer es robusto frente a ellos
                    ("scaler", StandardScaler()) # coloca los valores de cada variable en una distribución con media 0 y desviación estándar 1
                    ])
num_pipe_nombres = ['duration', 'amount', 'age', 'people_liable']


#-------------------------------------------------------------------------------------------------------------
# Variables categóricas/nominales/binarias:

# Función para agrupar categorías raras (<5%)
def find_rare_categories(df, variables, threshold=0.05):
    """
    Identifica las categorías raras usando únicamente el conjunto de entrenamiento.
    """
    rare_categories = {}
    for col in variables:
        freqs = df[col].value_counts(normalize=True)
        rare_categories[col] = freqs[freqs < threshold].index.tolist()
    return rare_categories

def group_rare_categories(df, rare_categories):
    """
    Agrupa como 'other' las categorías que fueron identificadas como raras en el conjunto de entrenamiento.
    """
    df_copy = df.copy()
    for col, categories in rare_categories.items():
        df_copy[col] = df_copy[col].replace(categories, "other")
    return df_copy

# Variables categóricas donde aplicamos agrupación
rare_vars = ["credit_history", "purpose", "other_debtors"]

# Identificamos las categorías raras SOLO utilizando el conjunto de entrenamiento
rare_categories = find_rare_categories(
    Xtrain,
    rare_vars,
    threshold=0.05
)
# Aplicamos las mismas categorías raras identificadas en train
# tanto a train como a test
Xtrain_grouped = group_rare_categories(
    Xtrain,
    rare_categories
)
Xtest_grouped = group_rare_categories(
    Xtest,
    rare_categories
)

# Asegurar que todas las columnas categóricas y ordinales sean de tipo string
# para evitar el TypeError en los codificadores debido a tipos int/str mixtos.
for col in list_paper_cat + list_paper_ord:
    Xtrain_grouped[col] = Xtrain_grouped[col].astype(str)
    Xtest_grouped[col] = Xtest_grouped[col].astype(str)


nom_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")), #para rellenar los valores faltantes con la moda de cada variable
    ("onehot", OneHotEncoder(handle_unknown="ignore"))      #convierte las variables categóricas en variables binarias (0/1) para cada categoría
                                                            #se usa One-Hot Encoder en lugar de Ordinal Encoder para evitar que el modelo interprete un orden en las categorías
])
nom_pipe_nombres = ['status', 'credit_history', 'purpose', 'savings', 'personal_status_sex', 'other_debtors', 'other_installment_plans', 'housing', 'telephone', 'foreign_worker']


#-------------------------------------------------------------------------------------------------------------
# Variables ordinales:
ord_pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")), #para rellenar los valores faltantes con la moda de cada variable
                    ("encoder", OrdinalEncoder()) #convierte las variables ordinales en valores numéricos
                    ])
ord_pipe_nombres = ['employment_duration', 'installment_rate', 'present_residence', 'property', 'number_credits', 'job']



# Conjuntemos las transformaciones de todo tipo de variable y
# dejamos sin procesar aquellas que hayas decidido no transformar:

columnasTransformer = ColumnTransformer([
                    ("num", num_pipe, num_pipe_nombres),
                    ("ord", ord_pipe, ord_pipe_nombres),
                    ("cat", nom_pipe, nom_pipe_nombres)
                     ])



# %% [markdown]
# Recall como métrica, es la tasa de verdaderos positivos
# 
# Recall = VP / (VP + FN)
# 
# donde los VP son clientes riesgosos (Credit_risk=1) correctamente identificados y FN clientes riesgosos que el modelo dejó pasar. Para este ejercicio nos interesa identificar a aquellas personas que en credit_risk = 1, es decir que no hay que otorgarles el crédito porque de lo contrario se le estaría otorgando un crédito a una persona que probablemente no pagará, lo que significa una pérdida financiera.

# %%
# Baseline - Referencia ingenua: siempre predecir la clase mayoritaria (0 = buen crédito)
# Métrica principal: Recall de la clase 1 (mal crédito)
mi_metrica = make_scorer(recall_score)

# Validación cruzada estratificada
cv = RepeatedStratifiedKFold(
    n_splits=3,
    n_repeats=3,
    random_state=6
)

# Baseline ingenuo:
modeloD = DummyClassifier(
    strategy="most_frequent"
    #strategy="stratified"
)

# Pipeline para mantener el preprocesamiento dentro de cada fold
pipelineD = Pipeline(
    steps=[
        ("ct", columnasTransformer),
        ("Dummy", modeloD)
    ]
)

# Evaluación mediante validación cruzada
scores = cross_val_score(
    pipelineD,
    Xtrain,
    ytrain,
    scoring=mi_metrica,
    cv=cv,
    n_jobs=-1
)

print("Desempeños obtenidos con DummyClassifier y Recall:")
print(scores)

print("\nPromedio y desviación estándar:")
print(
    "Recall = %.3f, std = %.3f"
    % (np.mean(scores), np.std(scores))
)

# %%
#Modelo Baseline Logistic Regression
# Métrica principal: Recall de la clase 1 (mal crédito)
mi_metrica = make_scorer(recall_score)
valorC = 10  # Valor de regularización para Logistic Regression
valor_max_iter = 1000  # Número máximo de iteraciones para la convergencia del modelo
valor_random_state = 6  # Semilla para la reproducibilidad del modelo

# Validación cruzada estratificada
cv = RepeatedStratifiedKFold(
    n_splits=3,
    n_repeats=3,
    random_state=6
)

# Modelo baseline
modeloLR = LogisticRegression(
    C=valorC,
    max_iter=valor_max_iter,
    random_state=valor_random_state
)

# Pipeline: preprocesamiento + modelo
pipelineLR = Pipeline(
    steps=[
        ("ct", columnasTransformer),
        ("LogisticRegression", modeloLR)
    ]
)

# Validación cruzada sobre TRAIN
scoresLR = cross_val_score(
    pipelineLR,
    Xtrain_grouped,
    ytrain,
    scoring=mi_metrica,
    cv=cv,
    n_jobs=-1
)

print("Desempeños obtenidos con Logistic Regression y la métrica Recall:")
print(scoresLR)

print("\nPromedio y desviación estándar:")
print(
    "Recall = %.3f, std = %.3f"
    % (np.mean(scoresLR), np.std(scoresLR))
)

# %% [markdown]
# El Recall obtenido por Logistic Regression se compara con el Recall del baseline ingenuo. 
# Un valor superior al baseline indica que el modelo aporta capacidad adicional para identificar casos de mal crédito.

# %%
# ENTRENAMIENTO DEL MODELO BASE + REGISTRO EN MLflow

# 1. Entrenar el modelo con todo el conjunto de entrenamiento
#from zipfile import Path

pipelineLR.fit(Xtrain_grouped, ytrain)

# Resultados obtenidos anteriormente con cross_val_score
recall_mean = np.mean(scoresLR)
recall_std = np.std(scoresLR)
precision = precision_score(
    ytrain,
    pipelineLR.predict(Xtrain_grouped)
)
print(f"Recall promedio CV: {recall_mean:.3f}")
print(f"Desviación estándar CV: {recall_std:.3f}")
print(f"Precisión promedio CV: {precision:.3f}")


# REGISTRO DEL EXPERIMENTO EN MLflow

with mlflow.start_run(run_name="LogisticRegression_Baseline"):

    # Parámetros
    mlflow.log_param("modelo", "Logistic Regression")
    mlflow.log_param("valor_C", valorC)
    mlflow.log_param("max_iter", valor_max_iter)
    mlflow.log_param("random_state", valor_random_state)
    mlflow.log_param("cv_n_splits", 3)
    mlflow.log_param("cv_n_repeats", 3)
    mlflow.log_param("metrica", "Recall")

    # Métricas
    mlflow.log_metric("recall_mean", recall_mean)
    mlflow.log_metric("recall_std", recall_std)
    mlflow.log_metric("precision", precision)


    # Artefacto: gráfica de los resultados de validación cruzada
    plt.figure(figsize=(6, 4))
    plt.bar(
        range(1, len(scoresLR) + 1),
        scoresLR
    )
    plt.xlabel("Fold")
    plt.ylabel("Recall")
    plt.title(f"Recall por fold - Logistic Regression (C={valorC})")
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig("recall_cv.png", dpi=300, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact("recall_cv.png")

    print("Experimento registrado correctamente en MLflow.")

# %%
modelo_final = LogisticRegression(
    C=10,
    max_iter=1000,
    random_state=6
)

pipeline_final = Pipeline([
    ("ct", columnasTransformer),
    ("LogisticRegression", modelo_final)
])

pipeline_final.fit(Xtrain_grouped, ytrain)

#Evaluación final con Xtest_grouped
ypred = pipeline_final.predict(Xtest_grouped)
recall_test = recall_score(ytest, ypred)
precision = precision_score(ytest, ypred)
print("Recall en test:", recall_test)

#Matriz de confusión
cm = confusion_matrix(ytest, ypred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Matriz de confusión - Logistic Regression")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# %%
#Registrar la métrica final en MLflow
with mlflow.start_run(run_name=f"LogisticRegression_C_{valorC}_Final"):

    # Parámetros del modelo
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("C", valorC)
    mlflow.log_param("max_iter", valor_max_iter)
    mlflow.log_param("random_state", valor_random_state)

    # Métrica obtenida durante la validación cruzada
    mlflow.log_metric("recall_cv_mean", np.mean(scoresLR))
    mlflow.log_metric("precision", precision)
   

    # Métrica final sobre el conjunto de prueba
    mlflow.log_metric("recall_test", recall_test)

    # Registrar matriz de confusión como artefacto
    mlflow.log_artifact("confusion_matrix.png")

    # Registrar el modelo entrenado
    mlflow.sklearn.log_model(
        sk_model=pipelineLR,
        name="model",
        skops_trusted_types=["numpy.dtype"]
    )

print("Evaluación final registrada en MLflow.")

# %% [markdown]
# Se realizan 4 runs dentro del mismo experimento cambiando 1 de los parámetros
# ![image.png](attachment:image.png)

# %% [markdown]
# Comparación de los 4 runs
# ![image.png](attachment:image.png)

# %% [markdown]
# Se incluye un artefacto a cada run que es la gráfica de los resultados de validación cruzada
# ![image.png](attachment:image.png)

# %% [markdown]
# Finalmente se hace un run adicional con los parámetros del mejor modelo
# ![image.png](attachment:image.png)

# %% [markdown]
# y se incluye la matriz de confusión en los artefactos
# ![image.png](attachment:image.png)

# %% [markdown]
# Reflexión:
# A partir de este ejercicio, considero que el ciclo de un proyecto de Machine Learning mejora significativamente mediante prácticas de MLOps, principalmente para hacerlo más reproducible, trazable y fácil de mantener.
# Es muy importante mantener un registro estructurado de experimentos, hiperparámetros, métricas y artefactos, como se realizó con MLflow durante esta actividad. De esta manera, es posible identificar qué configuración produjo cada resultado y reproducir el experimento posteriormente.
# 
# Una de las principales mejoras que incorporaría a este flujo sería automatizar en mayor medida el proceso de entrenamiento y evaluación. Actualmente, parte del flujo se realiza manualmente, por ejemplo, al modificar los parámetros, ejecutar nuevamente el modelo y posteriormente comparar los resultados en MLflow. Mediante prácticas de MLOps sería posible automatizar la ejecución de diferentes configuraciones, registrar sus resultados y seleccionar de manera más sistemática el mejor modelo de acuerdo con la métrica definida. También sería conveniente mantener un control de versiones tanto del código como de los datos y de los modelos generados, de manera que cada resultado pueda relacionarse con una versión específica del proceso.
# 
# Aunque actualmente no participo directamente en proyectos de MLOps, estos aprendizajes tienen una relación clara con mi área profesional en ABB. En el diseño de celdas robóticas de paletizado, la confiabilidad, trazabilidad y mantenibilidad de los sistemas son aspectos importantes. Por ejemplo, ante una futura integración de Machine Learning con sistemas de visión o robótica, prácticas como monitoreo y control de configuraciones podrían facilitar la implementación de soluciones consistentes entre diferentes celdas/versiones y reducir riesgos durante cambios o actualizaciones.
# Por ello, aunque mi experiencia actual con MLOps es de principiante, este ejercicio me permite identificar cómo sus principios pueden trasladarse a un entorno industrial. Más que enfocarse únicamente en desarrollar un modelo que funcione, MLOps plantea una visión de ciclo de vida completo, donde también importan la operación, el seguimiento, la documentación y la capacidad de mantener la solución a largo plazo.


