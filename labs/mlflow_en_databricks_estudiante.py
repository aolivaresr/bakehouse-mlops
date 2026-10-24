# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # MLflow en Databricks, paso a paso
# MAGIC
# MAGIC **Objetivo:** al terminar, sabrás registrar experimentos, comparar modelos, versionarlos en
# MAGIC Unity Catalog y consumirlos en producción por alias.
# MAGIC
# MAGIC **Caso:** clasificar vinos en 3 variedades a partir de 13 mediciones químicas
# MAGIC (dataset `wine` de scikit-learn, 178 muestras). No hay que cargar archivos.
# MAGIC
# MAGIC ### Vocabulario de MLflow
# MAGIC
# MAGIC | Concepto | Qué es | Ejemplo en este notebook |
# MAGIC |---|---|---|
# MAGIC | **Experimento** | Carpeta que agrupa intentos sobre un mismo problema | `curso-mlflow-vinos` |
# MAGIC | **Run** | Un intento: un entrenamiento con su configuración y resultados | `regresion_logistica` |
# MAGIC | **Parámetro** | Una entrada que eliges antes de entrenar | `max_depth = 3` |
# MAGIC | **Métrica** | Un número que mide el resultado | `accuracy = 0.94` |
# MAGIC | **Artefacto** | Un archivo producido por el run | la matriz de confusión |
# MAGIC | **Tag** | Una etiqueta libre para organizar | `autor = ana` |
# MAGIC | **Modelo registrado** | Un modelo con nombre y versiones en Unity Catalog | `workspace.default.clasificador_vinos` |
# MAGIC | **Alias** | Un nombre móvil que apunta a una versión | `@champion` |
# MAGIC
# MAGIC Las celdas marcadas con 👉 indican un buen momento para ir a la interfaz.
# MAGIC Los bloques **Ejercicio** son para que los resuelvas tú.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuración
# MAGIC Instalamos una versión reciente de MLflow (`-U` actualiza la que trae el cómputo) y reiniciamos Python.

# COMMAND ----------

# MAGIC %pip install -q -U "mlflow>=3.1" "scikit-learn>=1.3" matplotlib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalogo", "workspace")
dbutils.widgets.text("esquema", "default")

CATALOGO = dbutils.widgets.get("catalogo")
ESQUEMA = dbutils.widgets.get("esquema")
NOMBRE_MODELO = f"{CATALOGO}.{ESQUEMA}.clasificador_vinos"

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Los modelos registrados viven en Unity Catalog.
mlflow.set_registry_uri("databricks-uc")

# El experimento se crea en tu carpeta de usuario si no existe.
usuario = spark.sql("SELECT current_user()").first()[0]
experimento = mlflow.set_experiment(f"/Users/{usuario}/curso-mlflow-vinos")
print(f"Experimento: {experimento.name}")
print(f"Modelo:      {NOMBRE_MODELO}")

# Los árboles de scikit-learn guardan su estructura en este tipo interno. MLflow guarda los modelos
# en formato seguro (skops) y solo carga tipos declarados como confiables. Lo declaramos porque
# los modelos los entrenamos nosotros: nunca se confía en un modelo de origen desconocido.
TIPOS_CONFIABLES = ["sklearn.tree._tree.Tree"]

# COMMAND ----------

X, y = load_wine(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Entrenamiento: {X_train.shape} · Prueba: {X_test.shape}")
display(X_train.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Tu primer run
# MAGIC Todo lo que ocurre dentro de `with mlflow.start_run()` queda guardado en un run:
# MAGIC parámetros con `log_param`, métricas con `log_metric`, etiquetas con `set_tag` y
# MAGIC archivos con `log_figure` o `log_text`.

# COMMAND ----------

def matriz_confusion(y_real, y_pred, titulo):
    figura, eje = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_real, y_pred, ax=eje, colorbar=False)
    eje.set_title(titulo)
    plt.close(figura)
    return figura


with mlflow.start_run(run_name="regresion_logistica") as run:
    C = 1.0
    modelo = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    mlflow.log_param("algoritmo", "LogisticRegression")
    mlflow.log_param("C", C)
    mlflow.log_metric("accuracy", accuracy_score(y_test, pred))
    mlflow.set_tag("etapa", "linea_base")
    mlflow.log_figure(matriz_confusion(y_test, pred, "Regresión logística"), "matriz_confusion.png")
    mlflow.log_text("Primer run del curso: modelo lineal como línea base.", "notas.txt")

print(f"Run: {run.info.run_id} · accuracy {accuracy_score(y_test, pred):.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC 👉 **En la interfaz:** abre **Experiments › curso-mlflow-vinos › regresion_logistica**.
# MAGIC Revisa las pestañas *Overview* (parámetros, métricas, tags) y *Artifacts* (la matriz y las notas).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ejercicio 1
# MAGIC Crea un run llamado `regresion_logistica_c01` igual al anterior, pero con `C = 0.1`.
# MAGIC Además de `accuracy`, registra la métrica `f1_macro` (usa `f1_score(..., average="macro")`)
# MAGIC y un tag `autor` con tu nombre.

# COMMAND ----------

with mlflow.start_run(run_name="regresion_logistica_c01") as run:
    C = 0.1
    modelo = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    mlflow.log_param("algoritmo", "LogisticRegression")
    mlflow.log_param("C", C)
    mlflow.log_metric("accuracy", accuracy_score(y_test, pred))
    mlflow.log_metric("f1_macro", f1_score(y_test, pred,average="macro"))
    mlflow.set_tag("etapa", "linea_base")
    mlflow.set_tag("autor", "aolivare")
    mlflow.log_figure(matriz_confusion(y_test, pred, "Regresión logística"), "matriz_confusion.png")
    mlflow.log_text("ejecricio 1 del curso: modelo lineal como línea base.", "notas.txt")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Métricas que cambian en el tiempo
# MAGIC Con el argumento `step`, una métrica guarda una serie y la interfaz la dibuja como curva.
# MAGIC El *gradient boosting* agrega árboles uno a uno: medimos el accuracy después de cada árbol.

# COMMAND ----------

with mlflow.start_run(run_name="gradient_boosting_curva"):
    modelo = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    modelo.fit(X_train, y_train)
    mlflow.log_params({"algoritmo": "GradientBoosting", "n_estimators": 100, "learning_rate": 0.1})

    for paso, pred_parcial in enumerate(modelo.staged_predict(X_test), start=1):
        mlflow.log_metric("accuracy_prueba", accuracy_score(y_test, pred_parcial), step=paso)

    mlflow.log_metric("accuracy", accuracy_score(y_test, modelo.predict(X_test)))

# COMMAND ----------

# MAGIC %md
# MAGIC 👉 **En la interfaz:** en el run `gradient_boosting_curva`, abre la pestaña *Model metrics*.
# MAGIC ¿Desde cuántos árboles deja de mejorar el accuracy?

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Búsqueda de hiperparámetros con runs anidados
# MAGIC Un run **padre** agrupa la búsqueda y cada combinación es un run **hijo** (`nested=True`).
# MAGIC Así el experimento no se llena de runs sueltos.

# COMMAND ----------

combinaciones = [
    {"n_estimators": n, "max_depth": d} for n in [10, 50, 150] for d in [2, 4, None]
]

with mlflow.start_run(run_name="busqueda_random_forest") as padre:
    mlflow.set_tag("etapa", "busqueda")
    for params in combinaciones:
        nombre = f"rf_{params['n_estimators']}_prof_{params['max_depth']}"
        with mlflow.start_run(run_name=nombre, nested=True):
            modelo = RandomForestClassifier(random_state=42, **params).fit(X_train, y_train)
            pred = modelo.predict(X_test)
            mlflow.log_params({"algoritmo": "RandomForest", **params})
            mlflow.log_metric("accuracy", accuracy_score(y_test, pred))
            mlflow.log_metric("f1_macro", f1_score(y_test, pred, average="macro"))

print(f"Búsqueda terminada: {len(combinaciones)} runs hijos")

# COMMAND ----------

# MAGIC %md
# MAGIC 👉 **En la interfaz:** expande `busqueda_random_forest`, selecciona los hijos y haz clic en
# MAGIC **Compare**. Prueba la vista de *coordenadas paralelas* con `n_estimators`, `max_depth` y `f1_macro`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ejercicio 2
# MAGIC Agrega el hiperparámetro `min_samples_leaf` con valores `[1, 5]` a la búsqueda y córrela
# MAGIC de nuevo en un padre llamado `busqueda_random_forest_v2`. ¿Mejora algún resultado?

# COMMAND ----------

combinaciones = [
    {"n_estimators": n, "max_depth": d, "min_samples_leaf": l} for n in [10, 50, 150] for d in [2, 4, None] for l in [1, 5]]

with mlflow.start_run(run_name="busqueda_random_forest_v2") as padre:
    mlflow.set_tag("etapa", "busqueda")
    for params in combinaciones:
        nombre = f"rf_{params['n_estimators']}_prof_{params['max_depth']}"
        with mlflow.start_run(run_name=nombre, nested=True):
            modelo = RandomForestClassifier(random_state=42, **params).fit(X_train, y_train)
            pred = modelo.predict(X_test)
            mlflow.log_params({"algoritmo": "RandomForest", **params})
            mlflow.log_metric("accuracy", accuracy_score(y_test, pred))
            mlflow.log_metric("f1_macro", f1_score(y_test, pred, average="macro"))

print(f"Búsqueda terminada: {len(combinaciones)} runs hijos")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Autolog: menos código
# MAGIC `mlflow.sklearn.autolog()` registra solo los hiperparámetros, las métricas de entrenamiento y
# MAGIC algunos gráficos. Útil para explorar; para producción conviene decidir explícitamente qué se guarda.

# COMMAND ----------

mlflow.sklearn.autolog(log_models=False)

with mlflow.start_run(run_name="autolog_random_forest"):
    modelo = RandomForestClassifier(n_estimators=80, max_depth=4, random_state=42)
    modelo.fit(X_train, y_train)
    mlflow.log_metric("accuracy", accuracy_score(y_test, modelo.predict(X_test)))

mlflow.sklearn.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC 👉 **En la interfaz:** compara cuántos parámetros registró `autolog_random_forest` contra los
# MAGIC runs que hicimos a mano.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Consultar runs desde código
# MAGIC `mlflow.search_runs` devuelve un DataFrame de pandas: filtras, ordenas y eliges sin abrir la interfaz.

# COMMAND ----------

runs = mlflow.search_runs(
    experiment_ids=[experimento.experiment_id],
    filter_string="params.algoritmo = 'RandomForest' AND metrics.f1_macro > 0",
    order_by=["metrics.f1_macro DESC", "metrics.accuracy DESC"],
)
columnas = ["tags.mlflow.runName", "params.n_estimators", "params.max_depth", "metrics.f1_macro"]
display(runs[columnas].head(5))

mejor = runs.iloc[0]
mejores_params = {
    "n_estimators": int(mejor["params.n_estimators"]),
    "max_depth": None if mejor["params.max_depth"] == "None" else int(mejor["params.max_depth"]),
}
# Si hiciste el ejercicio 2, el mejor run puede traer también min_samples_leaf.
if pd.notna(mejor.get("params.min_samples_leaf")):
    mejores_params["min_samples_leaf"] = int(mejor["params.min_samples_leaf"])
print(f"Mejor configuración: {mejores_params} · f1_macro {mejor['metrics.f1_macro']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC 🤔 **Para discutir:** probablemente varias configuraciones empatan con `f1_macro = 1.0`.
# MAGIC ¿Significa que el modelo es perfecto? El set de prueba tiene solo 45 vinos: con tan pocos datos,
# MAGIC un acierto de más o de menos cambia mucho la métrica. Cuando hay empate, preferimos el modelo
# MAGIC más simple (menos árboles, menos profundidad). ¿Cómo cambiarías el `order_by` para lograrlo?

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Guardar el modelo
# MAGIC Un modelo de MLflow empaqueta tres cosas:
# MAGIC - el objeto entrenado,
# MAGIC - la **firma**: qué columnas y tipos recibe y qué devuelve (se infiere del `input_example`),
# MAGIC - las **dependencias** de Python para reproducirlo.
# MAGIC
# MAGIC Unity Catalog exige la firma para registrar un modelo.

# COMMAND ----------

with mlflow.start_run(run_name="random_forest_final") as run_final:
    modelo_final = RandomForestClassifier(random_state=42, **mejores_params).fit(X_train, y_train)
    pred = modelo_final.predict(X_test)
    mlflow.log_params({"algoritmo": "RandomForest", **mejores_params})
    mlflow.log_metric("accuracy", accuracy_score(y_test, pred))
    mlflow.log_metric("f1_macro", f1_score(y_test, pred, average="macro"))
    mlflow.log_figure(matriz_confusion(y_test, pred, "Random forest final"), "matriz_confusion.png")
    info_rf = mlflow.sklearn.log_model(
        modelo_final,
        name="modelo",
        input_example=X_train.head(3),
        skops_trusted_types=TIPOS_CONFIABLES,
    )

print(f"Modelo guardado en: {info_rf.model_uri}")
print(f"Firma: {info_rf.signature}")

# COMMAND ----------

# MAGIC %md
# MAGIC Cualquier modelo de MLflow se carga igual con `pyfunc`, sin importar la librería con que se entrenó.

# COMMAND ----------

cargado = mlflow.pyfunc.load_model(info_rf.model_uri)
print(cargado.predict(X_test.head(5)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Registro en Unity Catalog
# MAGIC Registrar le da al modelo un nombre de tres niveles (`catálogo.esquema.modelo`), **versiones**
# MAGIC numeradas y linaje hasta el run que lo produjo.

# COMMAND ----------

cliente = MlflowClient()

v_rf = mlflow.register_model(info_rf.model_uri, NOMBRE_MODELO)
cliente.update_registered_model(
    NOMBRE_MODELO, description="Clasificador de variedades de vino. Modelo de práctica del curso."
)
cliente.set_model_version_tag(NOMBRE_MODELO, v_rf.version, "f1_macro", f"{mejor['metrics.f1_macro']:.3f}")
cliente.set_registered_model_alias(NOMBRE_MODELO, "champion", v_rf.version)

print(f"Versión {v_rf.version} registrada y marcada como @champion")

# COMMAND ----------

# MAGIC %md
# MAGIC 👉 **En la interfaz:** **Catalog › workspace › default › clasificador_vinos**.
# MAGIC Revisa la versión, el alias `@champion` y el enlace al run de origen.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Champion y challenger
# MAGIC Llega un modelo candidato. Lo registramos como **nueva versión** con alias `@challenger` y lo
# MAGIC comparamos contra el champion **con los mismos datos**. Si gana, se mueve el alias.

# COMMAND ----------

with mlflow.start_run(run_name="gradient_boosting_candidato"):
    candidato = GradientBoostingClassifier(n_estimators=60, random_state=42).fit(X_train, y_train)
    mlflow.log_params({"algoritmo": "GradientBoosting", "n_estimators": 60})
    mlflow.log_metric("f1_macro", f1_score(y_test, candidato.predict(X_test), average="macro"))
    info_gb = mlflow.sklearn.log_model(
        candidato, name="modelo", input_example=X_train.head(3), skops_trusted_types=TIPOS_CONFIABLES
    )

v_gb = mlflow.register_model(info_gb.model_uri, NOMBRE_MODELO)
cliente.set_registered_model_alias(NOMBRE_MODELO, "challenger", v_gb.version)


def f1_del_alias(alias: str) -> float:
    modelo = mlflow.pyfunc.load_model(f"models:/{NOMBRE_MODELO}@{alias}")
    return f1_score(y_test, modelo.predict(X_test), average="macro")


comparacion = pd.DataFrame(
    [
        {"alias": "champion", "version": v_rf.version, "f1_macro": f1_del_alias("champion")},
        {"alias": "challenger", "version": v_gb.version, "f1_macro": f1_del_alias("challenger")},
    ]
)
display(comparacion)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ejercicio 3
# MAGIC Escribe la **regla de promoción**: si el challenger tiene un `f1_macro` estrictamente mayor que
# MAGIC el champion, mueve el alias `@champion` a la versión del challenger. Imprime qué versión quedó
# MAGIC como champion. Pregunta para discutir: ¿qué pasa si empatan?

# COMMAND ----------

# DBTITLE 1,Ejercicio 3: regla de promoción
# Comparar el desempeño de ambos modelos
f1_champion = f1_del_alias("champion")
f1_challenger = f1_del_alias("challenger")

print(f"F1-macro Champion:   {f1_champion:.4f}")
print(f"F1-macro Challenger: {f1_challenger:.4f}")

# Regla de promoción: mover @champion solo si el challenger gana estrictamente
if f1_challenger > f1_champion:
    version_ganadora = cliente.get_model_version_by_alias(NOMBRE_MODELO, "challenger").version
    cliente.set_registered_model_alias(NOMBRE_MODELO, "champion", version_ganadora)
    print(f"Promoción: @champion ahora apunta a la versión {version_ganadora}")
else:
    version_champion = cliente.get_model_version_by_alias(NOMBRE_MODELO, "champion").version
    print(f"Sin promoción: @champion sigue en la versión {version_champion}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Consumir el modelo en producción
# MAGIC Quien usa el modelo apunta al **alias**, nunca a un número de versión. Si mañana se promueve otra
# MAGIC versión, este código sigue igual.
# MAGIC
# MAGIC Para muchos datos se usa `spark_udf`: el modelo se aplica en paralelo sobre un DataFrame de Spark.

# COMMAND ----------

version_actual = cliente.get_model_version_by_alias(NOMBRE_MODELO, "champion").version
print(f"@champion apunta hoy a la versión {version_actual}")

datos_nuevos = spark.createDataFrame(X_test)
predecir = mlflow.pyfunc.spark_udf(spark, f"models:/{NOMBRE_MODELO}@champion", result_type="long")
predicciones = datos_nuevos.withColumn("variedad_predicha", predecir(*X_test.columns))
display(predicciones.select("alcohol", "color_intensity", "proline", "variedad_predicha").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Adelanto de GenAIOps: tracing
# MAGIC En aplicaciones con LLMs no basta una métrica final: hay que ver **cada paso**. Con
# MAGIC `@mlflow.trace` cada llamada deja una traza con sus entradas, salidas y tiempos.
# MAGIC Aquí lo aplicamos a funciones normales para ver la idea.

# COMMAND ----------

@mlflow.trace(span_type="RETRIEVER")
def buscar_muestra(indice: int) -> dict:
    return X_test.iloc[[indice]].to_dict(orient="records")[0]


@mlflow.trace(span_type="TOOL")
def clasificar(muestra: dict) -> int:
    modelo = mlflow.pyfunc.load_model(f"models:/{NOMBRE_MODELO}@champion")
    return int(modelo.predict(pd.DataFrame([muestra]))[0])


@mlflow.trace(span_type="CHAIN")
def responder(indice: int) -> str:
    muestra = buscar_muestra(indice)
    return f"La muestra {indice} es de la variedad {clasificar(muestra)} (alcohol {muestra['alcohol']})."


print(responder(0))

# COMMAND ----------

# MAGIC %md
# MAGIC 👉 **En la interfaz:** la traza aparece debajo de la celda y en la pestaña **Traces** del experimento.
# MAGIC
# MAGIC ## Repaso
# MAGIC 1. ¿Qué diferencia hay entre un parámetro, una métrica y un artefacto?
# MAGIC 2. ¿Para qué sirven los runs anidados?
# MAGIC 3. ¿Por qué Unity Catalog exige la firma del modelo?
# MAGIC 4. ¿Por qué el código de producción usa `@champion` y no `versions/2`?
# MAGIC 5. Tu modelo final usa los mismos datos de prueba con los que elegiste los hiperparámetros.
# MAGIC    ¿Qué riesgo tiene eso y cómo lo evitarías?
# MAGIC
# MAGIC ## Limpieza (opcional)
# MAGIC Para borrar el modelo registrado y empezar de cero, descomenta y ejecuta la siguiente celda.

# COMMAND ----------

# cliente.delete_registered_model(NOMBRE_MODELO)