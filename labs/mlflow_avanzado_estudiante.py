# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow avanzado en Databricks
# MAGIC
# MAGIC Continuación de *MLflow en Databricks, paso a paso*. Aquí organizamos experimentos grandes y
# MAGIC llevamos un modelo hasta producción con evidencia.
# MAGIC
# MAGIC | Sección | Tema |
# MAGIC |---|---|
# MAGIC | 1 | Datasets como entradas del run (linaje) |
# MAGIC | 2 | Runs anidados en tres niveles: estudio › familia › configuración |
# MAGIC | 3 | Validación cruzada: un run por fold y métricas agregadas |
# MAGIC | 4 | Optimización con Optuna y runs anidados |
# MAGIC | 5 | Artefactos ricos: tablas, diccionarios, notas y métricas del sistema |
# MAGIC | 6 | Evaluación formal con `mlflow.models.evaluate`, métrica de negocio y umbrales de validación |
# MAGIC | 7 | Modelo personalizado (`pyfunc`) con parámetros en la inferencia |
# MAGIC | 8 | Registro avanzado: validación, tags, búsqueda de versiones y LoggedModels |
# MAGIC
# MAGIC **Caso:** detectar tumores **malignos** a partir de 30 mediciones de imágenes de biopsias
# MAGIC (dataset `breast_cancer` de scikit-learn, 569 casos). Es un problema donde los errores
# MAGIC no cuestan lo mismo: no detectar un tumor maligno es mucho más grave que una falsa alarma.
# MAGIC
# MAGIC > Datos de práctica para aprender MLflow. Este modelo no es una herramienta médica.

# COMMAND ----------

# MAGIC %pip install -q -U "mlflow>=3.1" "scikit-learn>=1.3" optuna matplotlib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

#prueba para probar que se pueda acceder a la base de datos
dbutils.widgets.text("catalogo", "workspace")
dbutils.widgets.text("esquema", "default")

CATALOGO = dbutils.widgets.get("catalogo")
ESQUEMA = dbutils.widgets.get("esquema")
NOMBRE_MODELO = f"{CATALOGO}.{ESQUEMA}.detector_maligno"

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import MetricThreshold, infer_signature, make_metric
from mlflow.metrics import MetricValue
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

mlflow.set_registry_uri("databricks-uc")
usuario = spark.sql("SELECT current_user()").first()[0]
experimento = mlflow.set_experiment(f"/Users/{usuario}/curso-mlflow-avanzado")
cliente = MlflowClient()

# Modelos de árboles: tipo interno que declaramos confiable al guardarlos (formato skops).
TIPOS_CONFIABLES = ["sklearn.tree._tree.Tree"]

# Costo de negocio: no detectar un maligno (falso negativo) cuesta 10; una falsa alarma cuesta 1.
COSTO_FALSO_NEGATIVO = 10
COSTO_FALSO_POSITIVO = 1

optuna.logging.set_verbosity(optuna.logging.WARNING)
print(f"Experimento: {experimento.name}")

# COMMAND ----------

datos = load_breast_cancer(as_frame=True)
X = datos.data
# En el dataset original 0 = maligno. Lo invertimos para que la clase positiva (1) sea "maligno".
y = (1 - datos.target).rename("maligno")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Entrenamiento: {X_train.shape} · Prueba: {X_test.shape} · % malignos: {y.mean():.0%}")


def costo_negocio(y_real, y_pred) -> float:
    """Costo promedio por caso según la matriz de confusión."""
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()
    return (fn * COSTO_FALSO_NEGATIVO + fp * COSTO_FALSO_POSITIVO) / len(y_real)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Datasets como entradas del run
# MAGIC Con `mlflow.data` cada run guarda **qué datos usó**: nombre, esquema, número de filas y una
# MAGIC huella (*digest*). Si alguien cambia los datos, el digest cambia y se nota en la interfaz.

# COMMAND ----------

ds_entrenamiento = mlflow.data.from_pandas(
    pd.concat([X_train, y_train], axis=1), name="biopsias_entrenamiento", targets="maligno"
)
ds_prueba = mlflow.data.from_pandas(
    pd.concat([X_test, y_test], axis=1), name="biopsias_prueba", targets="maligno"
)

with mlflow.start_run(run_name="linea_base_logistica"):
    mlflow.log_input(ds_entrenamiento, context="entrenamiento")
    mlflow.log_input(ds_prueba, context="prueba")
    modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    pred = modelo.predict(X_test)
    mlflow.log_metrics(
        {
            "f1": f1_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "costo_negocio": costo_negocio(y_test, pred),
        }
    )

print(f"Digest de los datos de entrenamiento: {ds_entrenamiento.digest}")

# COMMAND ----------

# MAGIC %md
# MAGIC 👉 **En la interfaz:** en el run `linea_base_logistica`, busca la sección *Datasets used*.
# MAGIC Aparecen los dos datasets con su contexto, su esquema y su digest.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Runs anidados en tres niveles
# MAGIC Un experimento real compara **familias** de modelos y, dentro de cada una, **configuraciones**:
# MAGIC
# MAGIC ```
# MAGIC estudio_comparativo            ← padre: guarda el ganador
# MAGIC ├── familia_logistica          ← hijo: guarda el mejor de su familia
# MAGIC │   ├── logistica_C_0.01       ← nieto: una configuración
# MAGIC │   └── ...
# MAGIC ├── familia_random_forest
# MAGIC └── familia_gradient_boosting
# MAGIC ```
# MAGIC Cada configuración se evalúa con validación cruzada de 5 folds sobre el entrenamiento, así el
# MAGIC set de prueba queda intacto para el final.

# COMMAND ----------

familias = {
    "logistica": (
        lambda p: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **p)),
        [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}],
    ),
    "random_forest": (
        lambda p: RandomForestClassifier(random_state=42, **p),
        [{"n_estimators": 50, "max_depth": 3}, {"n_estimators": 100, "max_depth": 6}, {"n_estimators": 200, "max_depth": None}],
    ),
    "gradient_boosting": (
        lambda p: GradientBoostingClassifier(random_state=42, **p),
        [{"n_estimators": 50, "learning_rate": 0.1}, {"n_estimators": 100, "learning_rate": 0.05}, {"n_estimators": 150, "learning_rate": 0.1}],
    ),
}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def evaluar_configuracion(familia: str, params: dict) -> float:
    """Nieto: una configuración evaluada con validación cruzada."""
    nombre = f"{familia}_" + "_".join(f"{k}_{v}" for k, v in params.items())
    with mlflow.start_run(run_name=nombre, nested=True):
        constructor = familias[familia][0]
        puntajes = cross_val_score(constructor(params), X_train, y_train, cv=folds, scoring="f1")
        mlflow.log_params({"familia": familia, **params})
        mlflow.log_metrics({"cv_f1_media": puntajes.mean(), "cv_f1_desv": puntajes.std()})
        return float(puntajes.mean())


def correr_familia(familia: str) -> dict:
    """Hijo: todas las configuraciones de una familia."""
    with mlflow.start_run(run_name=f"familia_{familia}", nested=True):
        resultados = [(evaluar_configuracion(familia, p), p) for p in familias[familia][1]]
        mejor_f1, mejores_params = max(resultados, key=lambda r: r[0])
        mlflow.log_param("familia", familia)
        mlflow.log_metric("mejor_cv_f1", mejor_f1)
        mlflow.log_dict(mejores_params, "mejor_configuracion.json")
        return {"familia": familia, "cv_f1": mejor_f1, "params": mejores_params}


with mlflow.start_run(run_name="estudio_comparativo") as estudio:
    mlflow.log_input(ds_entrenamiento, context="entrenamiento")
    mejores_por_familia = [correr_familia(f) for f in familias]
    ganador = max(mejores_por_familia, key=lambda r: r["cv_f1"])
    mlflow.set_tags({"familia_ganadora": ganador["familia"], "nivel": "estudio"})
    mlflow.log_metric("mejor_cv_f1", ganador["cv_f1"])

ID_ESTUDIO = estudio.info.run_id
display(pd.DataFrame(mejores_por_familia))
print(f"Ganador: {ganador['familia']} {ganador['params']} · cv_f1 {ganador['cv_f1']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC 👉 **En la interfaz:** expande `estudio_comparativo`: verás los tres hijos y, dentro de cada uno,
# MAGIC sus nietos. Filtra con `tags.mlflow.parentRunId = '<id del estudio>'` para ver solo un nivel.
# MAGIC
# MAGIC Desde código también se recorre el árbol: cada run hijo guarda el id de su padre en el tag
# MAGIC `mlflow.parentRunId`.

# COMMAND ----------

hijos = mlflow.search_runs(
    experiment_ids=[experimento.experiment_id],
    filter_string=f"tags.mlflow.parentRunId = '{ID_ESTUDIO}'",
    order_by=["metrics.mejor_cv_f1 DESC"],
)
display(hijos[["tags.mlflow.runName", "params.familia", "metrics.mejor_cv_f1"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ejercicio 1: reabrir un run
# MAGIC Un run cerrado se puede **reabrir** con `mlflow.start_run(run_id=...)`. Reabre el estudio y
# MAGIC agrega una cuarta familia, `vecinos` (`KNeighborsClassifier` con `StandardScaler`), con
# MAGIC `n_neighbors` en `[3, 7, 15]`. Si gana, actualiza el tag `familia_ganadora` del estudio.

# COMMAND ----------

# TODO: agrega la familia "vecinos" al diccionario familias y reabre el estudio con run_id=ID_ESTUDIO


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Validación cruzada: un run por fold
# MAGIC Para el modelo ganador queremos ver **cada fold** por separado: si un fold sale muy distinto,
# MAGIC algo raro pasa con esos datos. Activamos además las **métricas del sistema** (CPU, memoria),
# MAGIC útiles para saber cuánto cómputo consume un entrenamiento.

# COMMAND ----------

constructor_ganador = familias[ganador["familia"]][0]

try:
    mlflow.enable_system_metrics_logging()
except Exception as error:
    print(f"Métricas del sistema no disponibles aquí: {error}")

with mlflow.start_run(run_name=f"cv_por_fold_{ganador['familia']}"):
    mlflow.log_params({"familia": ganador["familia"], **ganador["params"]})
    puntajes = []
    for numero, (idx_ent, idx_val) in enumerate(folds.split(X_train, y_train), start=1):
        with mlflow.start_run(run_name=f"fold_{numero}", nested=True):
            modelo = constructor_ganador(ganador["params"])
            modelo.fit(X_train.iloc[idx_ent], y_train.iloc[idx_ent])
            pred = modelo.predict(X_train.iloc[idx_val])
            f1 = f1_score(y_train.iloc[idx_val], pred)
            mlflow.log_metrics({"f1": f1, "recall": recall_score(y_train.iloc[idx_val], pred)})
            mlflow.log_param("filas_validacion", len(idx_val))
            puntajes.append(f1)
    mlflow.log_metrics({"cv_f1_media": np.mean(puntajes), "cv_f1_desv": np.std(puntajes), "cv_f1_min": np.min(puntajes)})

mlflow.disable_system_metrics_logging()
print("f1 por fold:", [round(p, 3) for p in puntajes])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Optimización con Optuna
# MAGIC En vez de probar una grilla fija, Optuna **propone** la siguiente combinación según los resultados
# MAGIC anteriores. Cada intento (*trial*) es un run anidado; el padre guarda el mejor y la historia completa.

# COMMAND ----------

N_INTENTOS = 15  # cada intento hace validación cruzada de 5 folds: súbelo con cuidado por la cuota


def objetivo(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }
    with mlflow.start_run(run_name=f"trial_{trial.number:02d}", nested=True):
        puntajes = cross_val_score(
            RandomForestClassifier(random_state=42, **params), X_train, y_train, cv=folds, scoring="f1"
        )
        mlflow.log_params(params)
        mlflow.log_metric("cv_f1_media", puntajes.mean())
    return float(puntajes.mean())


with mlflow.start_run(run_name="optuna_random_forest"):
    estudio_optuna = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    estudio_optuna.optimize(objetivo, n_trials=N_INTENTOS)

    mlflow.log_params({f"mejor_{k}": v for k, v in estudio_optuna.best_params.items()})
    mlflow.log_metric("mejor_cv_f1", estudio_optuna.best_value)
    historia = estudio_optuna.trials_dataframe()[["number", "value", "params_n_estimators", "params_max_depth", "params_min_samples_leaf"]]
    mlflow.log_table(historia, "historia_optuna.json")

    # Curva del mejor valor encontrado a lo largo de los intentos.
    for paso, mejor_hasta_aqui in enumerate(historia["value"].cummax(), start=1):
        mlflow.log_metric("mejor_cv_f1_acumulado", mejor_hasta_aqui, step=paso)

print(f"Mejor de Optuna: {estudio_optuna.best_params} · cv_f1 {estudio_optuna.best_value:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC 👉 **En la interfaz:** en `optuna_random_forest`, mira la curva `mejor_cv_f1_acumulado` y la tabla
# MAGIC `historia_optuna.json` en *Artifacts*. ¿En qué intento encontró Optuna su mejor configuración?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ejercicio 2
# MAGIC Agrega `max_features` a la búsqueda (`trial.suggest_categorical("max_features", ["sqrt", "log2", None])`)
# MAGIC y corre otro estudio en un padre llamado `optuna_random_forest_v2`. ¿Mejora el `mejor_cv_f1`?

# COMMAND ----------

# TODO: define objetivo_v2 con max_features y corre el estudio dentro del padre optuna_random_forest_v2


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. El modelo candidato, con artefactos ricos
# MAGIC Entrenamos con los mejores parámetros de Optuna y dejamos en el run todo lo que alguien
# MAGIC necesitaría para entender el modelo sin preguntarnos:
# MAGIC - una **nota** en Markdown (se ve en la parte superior del run),
# MAGIC - la **configuración** como diccionario,
# MAGIC - una **tabla** con los casos mal clasificados,
# MAGIC - la **curva ROC**.
# MAGIC
# MAGIC Las métricas se asocian al modelo con `model_id`, que es lo que permite buscarlo después
# MAGIC como *LoggedModel* (sección 8).

# COMMAND ----------

with mlflow.start_run(run_name="candidato_random_forest") as run_candidato:
    candidato = RandomForestClassifier(random_state=42, **estudio_optuna.best_params).fit(X_train, y_train)
    pred = candidato.predict(X_test)
    prob = candidato.predict_proba(X_test)[:, 1]

    info_candidato = mlflow.sklearn.log_model(
        candidato,
        name="modelo",
        input_example=X_train.head(3),
        skops_trusted_types=TIPOS_CONFIABLES,
    )
    mlflow.log_params(estudio_optuna.best_params)
    mlflow.log_metrics(
        {"f1": f1_score(y_test, pred), "recall": recall_score(y_test, pred), "costo_negocio": costo_negocio(y_test, pred)},
        model_id=info_candidato.model_id,
        dataset=ds_prueba,
    )
    mlflow.log_input(ds_prueba, context="prueba")

    mlflow.set_tag(
        "mlflow.note.content",
        "### Candidato\nRandom forest con los mejores parámetros de Optuna. "
        f"Costo de negocio: falso negativo = {COSTO_FALSO_NEGATIVO}, falso positivo = {COSTO_FALSO_POSITIVO}.",
    )
    mlflow.log_dict({"origen": "optuna_random_forest", **estudio_optuna.best_params}, "configuracion.json")

    errores = X_test.assign(real=y_test, predicho=pred, prob_maligno=prob.round(3))[lambda d: d.real != d.predicho]
    mlflow.log_table(errores[["real", "predicho", "prob_maligno", "mean radius", "worst area"]], "errores.json")

    figura, eje = plt.subplots(figsize=(4, 4))
    RocCurveDisplay.from_predictions(y_test, prob, ax=eje)
    mlflow.log_figure(figura, "curva_roc.png")
    plt.close(figura)

print(f"Casos mal clasificados en prueba: {len(errores)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Evaluación formal y umbrales de validación
# MAGIC `mlflow.models.evaluate` calcula de una vez las métricas estándar de clasificación y gráficos
# MAGIC (matriz de confusión, ROC, precisión-recall). Le agregamos nuestra **métrica de negocio**.
# MAGIC
# MAGIC Después, `mlflow.validate_evaluation_results` decide si el candidato **puede pasar** a producción:
# MAGIC un mínimo de recall y un costo menor que el de la línea base.

# COMMAND ----------

def _costo_desde_predicciones(predictions, targets, metrics):
    return MetricValue(aggregate_results={"costo_negocio": costo_negocio(targets, predictions)})


metrica_costo = make_metric(
    eval_fn=_costo_desde_predicciones, greater_is_better=False, name="costo_negocio"
)
datos_eval = X_test.assign(maligno=y_test)

modelo_base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)

with mlflow.start_run(run_name="evaluacion_linea_base"):
    info_base = mlflow.sklearn.log_model(modelo_base, name="modelo", input_example=X_train.head(3))
    resultado_base = mlflow.models.evaluate(
        info_base.model_uri, datos_eval, targets="maligno", model_type="classifier", extra_metrics=[metrica_costo]
    )

with mlflow.start_run(run_name="evaluacion_candidato"):
    resultado_candidato = mlflow.models.evaluate(
        info_candidato.model_uri, datos_eval, targets="maligno", model_type="classifier", extra_metrics=[metrica_costo]
    )

metricas = ["recall_score", "precision_score", "f1_score", "roc_auc", "costo_negocio"]
display(
    pd.DataFrame(
        {
            "linea_base": [resultado_base.metrics.get(m) for m in metricas],
            "candidato": [resultado_candidato.metrics.get(m) for m in metricas],
        },
        index=metricas,
    )
)

# COMMAND ----------

# Reglas para aceptar al candidato:
# - recall de al menos 0,90 (detectar 9 de cada 10 malignos),
# - costo de negocio al menos 0,001 MENOR que el de la línea base.
umbrales = {
    "recall_score": MetricThreshold(threshold=0.90, greater_is_better=True),
    "costo_negocio": MetricThreshold(greater_is_better=False, min_absolute_change=0.001),
}

try:
    mlflow.validate_evaluation_results(
        validation_thresholds=umbrales, candidate_result=resultado_candidato, baseline_result=resultado_base
    )
    VALIDACION = "aprobada"
    print("Validación aprobada: el candidato cumple los umbrales.")
except Exception as rechazo:
    VALIDACION = "rechazada"
    print(f"Validación rechazada:\n{rechazo}")

# El modelo que sigue adelante es el candidato si pasó, o la línea base si no.
if VALIDACION == "aprobada":
    modelo_elegido, nombre_elegido = candidato, "random_forest_optuna"
else:
    modelo_elegido, nombre_elegido = modelo_base, "logistica_linea_base"
print(f"Modelo que sigue a la sección 7: {nombre_elegido}")

# COMMAND ----------

# MAGIC %md
# MAGIC 👉 **En la interfaz:** en `evaluacion_candidato`, la pestaña *Artifacts* tiene la matriz de
# MAGIC confusión, la curva ROC y la de precisión-recall generadas automáticamente.
# MAGIC
# MAGIC 🤔 **Para discutir:** es muy posible que el random forest de Optuna, más complejo, **pierda**
# MAGIC contra la regresión logística: Optuna optimizó el f1, pero el negocio castiga los falsos negativos.
# MAGIC Optimizar la métrica equivocada da un modelo "mejor" que no sirve.
# MAGIC - Si la validación rechaza al candidato, ¿qué es más honesto: cambiar los umbrales o volver a entrenar?
# MAGIC - ¿Quién debería definir esos umbrales en una empresa?
# MAGIC - ¿Cómo cambiarías el `scoring` de Optuna para que optimice lo que le importa al negocio?

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Modelo personalizado con parámetros en la inferencia
# MAGIC Un clasificador devuelve 0 o 1 con un umbral fijo de 0,5. Pero el umbral es una **decisión de
# MAGIC negocio**: con un umbral más bajo se detectan más malignos a cambio de más falsas alarmas.
# MAGIC
# MAGIC Con `mlflow.pyfunc.PythonModel` empaquetamos el modelo junto con esa lógica, y con `params` en la
# MAGIC firma el umbral se elige **al momento de predecir**, sin reentrenar ni registrar otra versión.

# COMMAND ----------

class DetectorConUmbral(mlflow.pyfunc.PythonModel):
    def __init__(self, modelo):
        self.modelo = modelo

    def predict(self, context, model_input, params=None):
        umbral = (params or {}).get("umbral", 0.5)
        prob = self.modelo.predict_proba(model_input)[:, 1]
        return pd.DataFrame(
            {
                "prob_maligno": prob,
                "prediccion": (prob >= umbral).astype(int),
                "prioridad": np.select([prob >= 0.8, prob >= umbral], ["alta", "media"], default="baja"),
            }
        )


# Envolvemos el modelo que ganó la validación de la sección 6.
detector = DetectorConUmbral(modelo_elegido)
ejemplo_salida = detector.predict(None, X_train.head(3))
firma = infer_signature(X_train.head(3), ejemplo_salida, params={"umbral": 0.5})

with mlflow.start_run(run_name="detector_con_umbral") as run_detector:
    mlflow.log_param("modelo_base", nombre_elegido)
    info_detector = mlflow.pyfunc.log_model(
        name="modelo", python_model=detector, signature=firma, input_example=X_train.head(3)
    )
    pred_detector = detector.predict(None, X_test)["prediccion"]
    mlflow.log_metrics(
        {"recall": recall_score(y_test, pred_detector), "costo_negocio": costo_negocio(y_test, pred_detector)},
        model_id=info_detector.model_id,
    )

print(firma)

# COMMAND ----------

cargado = mlflow.pyfunc.load_model(info_detector.model_uri)
for umbral in [0.5, 0.3]:
    salida = cargado.predict(X_test, params={"umbral": umbral})
    print(
        f"umbral {umbral}: recall {recall_score(y_test, salida['prediccion']):.3f} · "
        f"costo {costo_negocio(y_test, salida['prediccion']):.3f}"
    )
display(cargado.predict(X_test.head(5), params={"umbral": 0.3}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ejercicio 3: el umbral óptimo
# MAGIC En un run llamado `barrido_umbral`, prueba umbrales de 0,05 a 0,95 en pasos de 0,05 con el
# MAGIC modelo `cargado`. Registra `costo_negocio` y `recall` con `step` (usa el umbral × 100 como paso)
# MAGIC para ver las curvas, y guarda el umbral de menor costo como parámetro `umbral_optimo`.

# COMMAND ----------

# TODO: barre umbrales con cargado.predict(X_test, params={"umbral": u}) y registra costo y recall por paso


# COMMAND ----------

# MAGIC %md
# MAGIC 🤔 **Para discutir:** el umbral de menor costo puede salir muy bajo. Con un falso negativo 10 veces
# MAGIC más caro, al modelo le conviene marcar casi todo como sospechoso. ¿Qué pasaría con el equipo médico
# MAGIC que revisa las alarmas? ¿Qué le falta a nuestra función de costo para reflejar eso?

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Registro avanzado
# MAGIC Registramos el detector y dejamos la **decisión documentada** en tags de versión: qué pasó con la
# MAGIC validación del candidato y qué modelo quedó adentro. Cualquiera que abra la versión en Unity
# MAGIC Catalog entiende por qué está ahí.

# COMMAND ----------

version = mlflow.register_model(info_detector.model_uri, NOMBRE_MODELO)
cliente.update_registered_model(
    NOMBRE_MODELO, description="Detector de tumores malignos con umbral configurable. Modelo de práctica."
)
cliente.update_model_version(
    NOMBRE_MODELO, version.version, description=f"{nombre_elegido} con umbral de decisión como parámetro."
)
cliente.set_model_version_tag(NOMBRE_MODELO, version.version, "modelo_base", nombre_elegido)
cliente.set_model_version_tag(NOMBRE_MODELO, version.version, "validacion_candidato", VALIDACION)
cliente.set_model_version_tag(NOMBRE_MODELO, version.version, "run_origen", run_detector.info.run_id)
cliente.set_registered_model_alias(NOMBRE_MODELO, "champion", version.version)

print(f"Versión {version.version} ({nombre_elegido}) → @champion · validación del candidato: {VALIDACION}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Buscar versiones y modelos
# MAGIC - `search_model_versions` filtra versiones registradas por nombre y tags.
# MAGIC - `search_logged_models` (MLflow 3) busca **todos los modelos guardados** en el experimento,
# MAGIC   estén registrados o no, ordenados por las métricas que les asociamos con `model_id`.

# COMMAND ----------

versiones = cliente.search_model_versions(f"name = '{NOMBRE_MODELO}'")
display(
    pd.DataFrame(
        [
            {
                "version": v.version,
                "modelo_base": v.tags.get("modelo_base"),
                "validacion_candidato": v.tags.get("validacion_candidato"),
                "aliases": v.aliases,
            }
            for v in versiones
        ]
    )
)

modelos_guardados = mlflow.search_logged_models(
    experiment_ids=[experimento.experiment_id],
    order_by=[{"field_name": "metrics.costo_negocio", "ascending": True}],
    output_format="list",
)
display(
    pd.DataFrame(
        [
            {
                "model_id": m.model_id,
                "run": m.source_run_id,
                "costo_negocio": next((x.value for x in m.metrics or [] if x.key == "costo_negocio"), None),
            }
            for m in modelos_guardados
        ]
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Consumir el champion con un umbral de negocio

# COMMAND ----------

en_produccion = mlflow.pyfunc.load_model(f"models:/{NOMBRE_MODELO}@champion")
display(en_produccion.predict(X_test.head(10), params={"umbral": 0.3}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Repaso
# MAGIC 1. ¿Qué ganas al registrar los datasets con `log_input`?
# MAGIC 2. ¿Cuándo conviene anidar en tres niveles y cuándo basta con dos?
# MAGIC 3. ¿Por qué evaluamos las configuraciones con validación cruzada y no con el set de prueba?
# MAGIC 4. ¿Qué diferencia hay entre una métrica estadística (f1) y una de negocio (costo)?
# MAGIC 5. ¿Por qué el umbral es un parámetro de inferencia y no un hiperparámetro de entrenamiento?
# MAGIC 6. ¿Qué diferencia hay entre un *LoggedModel* y un modelo registrado en Unity Catalog?
# MAGIC
# MAGIC ## Limpieza (opcional)

# COMMAND ----------

# cliente.delete_registered_model(NOMBRE_MODELO)