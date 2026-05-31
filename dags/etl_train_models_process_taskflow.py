
import datetime
from airflow.decorators import dag, task
import logging
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, fbeta_score, recall_score, precision_score
import matplotlib.pyplot as plt
from utils.s3_utils import get_s3_client, s3_file_exists, s3_to_dataframe
from utils.model_utils import cv_f2, METRIC_NAME, BETA

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DIR_DATA_PROCESSED = os.getenv("DIR_DATA_PROCESSED", "/processed/final")
BUCKET     = "mlflow-artifacts"

default_args = {
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}

@dag(
    dag_id="etl_train_models_process_taskflow",
    description="ETL para preparar datos de entrenamiento usando TaskFlow API",
    default_args=default_args,
    catchup=False,
    tags=["ETL", "TaskFlow"],
)
def process_etl_taskflow():

    def create_experiment(experiment_name, model):
        """Create a new MLFlow experiment with a specified name.
        Save artifacts to the specified S3 bucket."""
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        logging.info("Conectado a MLflow Tracking Server")

        if not mlflow.get_experiment_by_name(experiment_name):
            logging.info(f"Creando experimento MLflow: {experiment_name}")
            mlflow.create_experiment(name=experiment_name, 
                                    tags={"project":"healthcare-stroke-prediction", "model": model,
                                   "team": "mlops1-fiuba"}) 

        experiment = mlflow.get_experiment_by_name(experiment_name)

        return experiment.experiment_id

    @task.python(task_id="check_data_to_process", multiple_outputs=True)
    def check_data_to_process():
        """
        Verificamos si el dataset ya fue procesado, si no, lo descargamos y preparamos para el siguiente nodo
        """
        s3 = get_s3_client()

        if not s3_file_exists(s3, BUCKET, f"{DIR_DATA_PROCESSED}/X_train.csv"):
            logging.warning(f"Dataset no encontrado localmente: X_train. Verificando en S3/MinIO...")
            raise FileNotFoundError(f"Dataset no encontrado: X_train.csv")

        if not s3_file_exists(s3, BUCKET, f"{DIR_DATA_PROCESSED}/y_train.csv"):
            logging.warning(f"Dataset no encontrado localmente: y_train. Verificando en S3/MinIO...")
            raise FileNotFoundError(f"Dataset no encontrado: y_train.csv")

        if not s3_file_exists(s3, BUCKET, f"{DIR_DATA_PROCESSED}/X_test.csv"):
            logging.warning(f"Dataset no encontrado localmente: X_test. Verificando en S3/MinIO...")
            raise FileNotFoundError(f"Dataset no encontrado: X_test.csv")

        if not s3_file_exists(s3, BUCKET, f"{DIR_DATA_PROCESSED}/y_test.csv"):
            logging.warning(f"Dataset no encontrado localmente: y_test. Verificando en S3/MinIO...")
            raise FileNotFoundError(f"Dataset no encontrado: y_test.csv")

        logging.info("✅ Dataset encontrado localmente. Continuando con el proceso.")

        X_train = s3_to_dataframe(s3, BUCKET, f"{DIR_DATA_PROCESSED}/X_train.csv")
        y_train = s3_to_dataframe(s3, BUCKET, f"{DIR_DATA_PROCESSED}/y_train.csv")
        X_val = s3_to_dataframe(s3, BUCKET, f"{DIR_DATA_PROCESSED}/X_val.csv")
        y_val = s3_to_dataframe(s3, BUCKET, f"{DIR_DATA_PROCESSED}/y_val.csv")

        if X_train.empty or y_train.empty or X_val.empty or y_val.empty:
            logging.error("❌ Uno o más datasets están vacíos. Verifique los archivos en S3/MinIO.")
            raise ValueError("Dataset vacío encontrado")
        
        if len(X_train) != len(y_train) or len(X_val) != len(y_val):
            logging.error("❌ El número de muestras no coincide.")
            raise ValueError("Desbalance entre X e y")
        
        if len(X_val) < 300 or len(y_val) < 300:
            logging.error("❌ El número de muestras en X_val o y_val es menor a 300. Verifique los archivos en S3/MinIO.")
            raise ValueError("Dataset de validación demasiado pequeño")
        
        X_train_path = "./X_train.csv"
        y_train_path = "./y_train.csv"
        X_val_path = "./X_val.csv"
        y_val_path = "./y_val.csv"

        X_train.to_csv(X_train_path, index=False)
        y_train.to_csv(y_train_path, index=False)
        X_val.to_csv(X_val_path, index=False)
        y_val.to_csv(y_val_path, index=False)

        logging.info("✅ Dataset descargado y preparado localmente. Continuando con el proceso.")
        return {
            "X_train_path": X_train_path,
            "y_train_path": y_train_path,
            "X_val_path": X_val_path,
            "y_val_path": y_val_path,
        }

    @task.python(task_id="create_base_model", multiple_outputs=True)
    def create_base_model(X_val_path, y_val_path):
        """
        Creamos un modelo base simple (baseline) para comparar con el modelo entrenado posteriormente.
        En este caso, un clasificador que solo usa la edad como predictor.
        """
        from utils.model_utils import AgeBaselineClassifier
        import mlflow
        from utils.plots import plot_confusion_matrix, plot_roc_curve, plot_precision_recall
        logging.info("Iniciando tarea modelo base (baseline) usando solo la edad como predictor...")

        experiment_name = "baseline_model_evaluation"
        experiment_id = create_experiment(experiment_name, model="baseline")

        logging.info("Creando modelo base...")
        baseline_model = AgeBaselineClassifier(threshold=0.3)

        logging.info(f"Evaluando modelo base con datos de validación...")

        X_val = pd.read_csv(X_val_path)
        y_val = pd.read_csv(y_val_path)
        # convert to numpy arrays because AgeBaselineClassifier expects ndarray indexing (X[:, 0])
        X_val_np = X_val.to_numpy()
        y_val_arr = y_val.to_numpy().ravel()
        y_pred_baseline_val = baseline_model.predict(X_val_np)
        y_prob_baseline_val = baseline_model.predict_proba(X_val_np)

        bl_metrics = {
            'AUC-ROC':   roc_auc_score(y_val_arr, y_prob_baseline_val[:, 1]),
            'PR-AUC':    average_precision_score(y_val_arr, y_prob_baseline_val[:, 1]),
            'F1':        f1_score(y_val_arr, y_pred_baseline_val),
            METRIC_NAME: fbeta_score(y_val_arr, y_pred_baseline_val, beta=BETA),
            'Recall':    recall_score(y_val_arr, y_pred_baseline_val),
            'Precision': precision_score(y_val_arr, y_pred_baseline_val, zero_division=0),
        }

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        logging.info(f"Registrando en MLflow: {experiment_name} (ID: {experiment_id})")
        run_name_base = 'base_model_exp'

        matrix_plot = plot_confusion_matrix(y_val_arr, y_pred_baseline_val, save_path=None)
        roc_plots = plot_roc_curve(y_val_arr, y_prob_baseline_val, save_path=None)
        pr_plots = plot_precision_recall(y_val_arr, y_prob_baseline_val[:, 1], save_path=None)

        with mlflow.start_run(experiment_id = experiment_id, 
                 run_name=run_name_base,
                 tags={"model":"baseline", "type":"evaluation"}):
            
            logging.info("✅ Log de modelos del modelo base en MLflow")
            mlflow.sklearn.log_model(sk_model=baseline_model, 
                                name=experiment_name)

            logging.info("✅ Parámetros del modelo base:")
            mlflow.log_params(baseline_model.get_params())
            
            logging.info("✅ Métricas del modelo base registradas en MLflow")
            mlflow.log_metrics(bl_metrics)

            logging.info("✅ Plots de evaluación del modelo base generados.")
            mlflow.log_figure(matrix_plot, artifact_file="matrix_plot.png")
            mlflow.log_figure(roc_plots[0], artifact_file="roc_curve_1_plot.png")
            mlflow.log_figure(roc_plots[1], artifact_file="roc_curve_2_plot.png")
            mlflow.log_figure(pr_plots, artifact_file="precision_recall_plot.png")

        logging.info("✅ Tarea modelo base finalizada.")
        return bl_metrics

    @task.python(task_id="create_knn_model", multiple_outputs=True)
    def create_knn_model(X_train_path, y_train_path, X_val_path, y_val_path):
        """
        Creamos un modelo KNN para comparar con el modelo entrenado posteriormente.
        En este caso, un clasificador que usa múltiples características como predictors.
        """
        from sklearn.neighbors import KNeighborsClassifier
        import optuna
        import mlflow
        from utils.plots import plot_confusion_matrix, plot_roc_curve, plot_precision_recall

        logging.info("Iniciando tarea modelo KNN...")

        experiment_id = create_experiment("knn_model_evaluation", model="knn")

        logging.info(f"Usando experimento MLflow: knn_model_evaluation (ID: {experiment_id})")
        run_name_base = 'knn_model_exp'

        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path)
        y_train_arr = y_train.to_numpy().ravel()

        def knn_objective(trial):
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
                'weights':     trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p':           trial.suggest_categorical('p', [1, 2]),
            }
            model = KNeighborsClassifier(n_jobs=-1, **params)
            return cv_f2(model, X_train, y_train_arr)

        knn_study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        knn_study.optimize(knn_objective, n_trials=50, show_progress_bar=True)

        knn_best_params = knn_study.best_params
        knn_best = KNeighborsClassifier(n_jobs=-1, **knn_best_params).fit(X_train, y_train_arr)

        logging.info("✅ Modelo KNN creado.")
        logging.info("✅ Parámetros del modelo KNN:")

        X_val = pd.read_csv(X_val_path)
        y_val = pd.read_csv(y_val_path)
        y_val_arr = y_val.to_numpy().ravel()

        y_prob_knn_val = knn_best.predict_proba(X_val)
        y_pred_knn_val = knn_best.predict(X_val)

        bl_metrics = {
            'AUC-ROC':   roc_auc_score(y_val_arr, y_prob_knn_val[:, 1]),
            'PR-AUC':    average_precision_score(y_val_arr, y_prob_knn_val[:, 1]),
            'F1':        f1_score(y_val_arr, y_pred_knn_val),
            METRIC_NAME: fbeta_score(y_val_arr, y_pred_knn_val, beta=BETA),
            'Recall':    recall_score(y_val_arr, y_pred_knn_val),
            'Precision': precision_score(y_val_arr, y_pred_knn_val, zero_division=0),
        }

        matrix_plot = plot_confusion_matrix(y_val_arr, y_pred_knn_val, save_path=None)
        roc_plots = plot_roc_curve(y_val_arr, y_prob_knn_val, save_path=None)
        pr_plots = plot_precision_recall(y_val_arr, y_prob_knn_val[:, 1], save_path=None)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        with mlflow.start_run(experiment_id=experiment_id,
                              run_name=run_name_base,
                              tags={"model":"knn", "type":"evaluation"}):
            logging.info("✅ Log de modelo KNN en MLflow")
            mlflow.log_params(knn_best_params)
            mlflow.sklearn.log_model(sk_model=knn_best, name="knn_model")
            mlflow.log_metrics(bl_metrics)
            logging.info("✅ Modelos y métricas KNN registrados en MLflow")
            mlflow.log_figure(matrix_plot, artifact_file="knn_confusion_matrix.png")
            mlflow.log_figure(roc_plots[0], artifact_file="knn_roc_curve_1.png")
            mlflow.log_figure(roc_plots[1], artifact_file="knn_roc_curve_2.png")
            mlflow.log_figure(pr_plots, artifact_file="knn_precision_recall_curve.png")

        logging.info("✅ Tarea modelo KNN finalizada.")
        return bl_metrics
    
    @task.python(task_id="create_decision_tree_model", multiple_outputs=True)
    def create_decision_tree_model(X_train_path, y_train_path, X_val_path, y_val_path):
        """
        Creamos un modelo de árbol de decisión para comparar con el modelo entrenado posteriormente.
        En este caso, un clasificador que usa múltiples características como predictors.
        """
        from sklearn.tree import DecisionTreeClassifier
        import optuna
        import mlflow
        from utils.plots import plot_confusion_matrix, plot_roc_curve, plot_precision_recall

        logging.info("Iniciando tarea modelo de árbol de decisión...")

        experiment_id = create_experiment("decision_tree_model_evaluation", model="decision_tree")

        logging.info(f"Usando experimento MLflow: decision_tree_model_evaluation (ID: {experiment_id})")
        run_name_base = 'decision_tree_model_exp'

        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path)
        y_train_arr = y_train.to_numpy().ravel()

        def dt_objective(trial):
            params = {
                'max_depth':         trial.suggest_int('max_depth', 3, 20),
                'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'criterion':         trial.suggest_categorical('criterion', ['gini', 'entropy']),
            }
            model = DecisionTreeClassifier(class_weight='balanced', random_state=42, **params)
            return cv_f2(model, X_train, y_train_arr)

        dt_study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        dt_study.optimize(dt_objective, n_trials=50, show_progress_bar=True)

        dt_best_params = dt_study.best_params
        dt_best = DecisionTreeClassifier(**dt_best_params).fit(X_train, y_train_arr)

        logging.info("✅ Modelo de árbol de decisión creado.")
        logging.info("✅ Parámetros del modelo de árbol de decisión:")
        mlflow.log_params(dt_best_params)

        X_val = pd.read_csv(X_val_path)
        y_val = pd.read_csv(y_val_path)
        y_val_arr = y_val.to_numpy().ravel()

        y_prob_dt_val = dt_best.predict_proba(X_val)
        y_pred_dt_val = dt_best.predict(X_val)

        bl_metrics = {
            'AUC-ROC':   roc_auc_score(y_val_arr, y_prob_dt_val[:, 1]),
            'PR-AUC':    average_precision_score(y_val_arr, y_prob_dt_val[:, 1]),
            'F1':        f1_score(y_val_arr, y_pred_dt_val),
            METRIC_NAME: fbeta_score(y_val_arr, y_pred_dt_val, beta=BETA),
            'Recall':    recall_score(y_val_arr, y_pred_dt_val),
            'Precision': precision_score(y_val_arr, y_pred_dt_val, zero_division=0),
        }

        matrix_plot = plot_confusion_matrix(y_val_arr, y_pred_dt_val, save_path=None)
        roc_plots = plot_roc_curve(y_val_arr, y_prob_dt_val, save_path=None)
        pr_plots = plot_precision_recall(y_val_arr, y_prob_dt_val[:, 1], save_path=None)
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        with mlflow.start_run(experiment_id=experiment_id,
                              run_name=run_name_base,
                              tags={"model":"decision_tree", "type":"evaluation"}):
            logging.info("✅ Log de modelo de árbol de decisión en MLflow")
            mlflow.sklearn.log_model(sk_model=dt_best, name="decision_tree_model")
            mlflow.log_metrics(bl_metrics)
            logging.info("✅ Modelo de árbol de decisión y métricas registrados en MLflow")
            mlflow.log_figure(matrix_plot, artifact_file="dt_confusion_matrix.png")
            mlflow.log_figure(roc_plots[0], artifact_file="dt_roc_curve_1.png")
            mlflow.log_figure(roc_plots[1], artifact_file="dt_roc_curve_2.png")
            mlflow.log_figure(pr_plots, artifact_file="dt_precision_recall_curve.png")

        logging.info("✅ Tarea modelo de árbol de decisión finalizada.")
        return bl_metrics

    @task.python(task_id="create_xgboost_model", multiple_outputs=True)
    def create_xgboost_model(X_train_path, y_train_path, X_val_path, y_val_path):
        """
        Creamos un modelo de XGBoost para comparar con el modelo entrenado posteriormente.
        En este caso, un clasificador que usa múltiples características como predictors.
        """
        from xgboost import XGBClassifier
        import optuna
        import mlflow
        import numpy as np
        from utils.plots import plot_confusion_matrix, plot_roc_curve, plot_precision_recall

        logging.info("Iniciando tarea modelo de XGBoost...")

        experiment_id = create_experiment("xgboost_model_evaluation", model="xgboost")

        logging.info(f"Usando experimento MLflow: xgboost_model_evaluation (ID: {experiment_id})")
        run_name_base = 'xgboost_model_exp'

        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path)
        y_train_arr = y_train.to_numpy().ravel()

        neg, pos = np.bincount(y_train_arr)
        scale_pos = neg / pos

        def xgb_objective(trial):
            params = {
                'n_estimators':     trial.suggest_categorical('n_estimators', [100, 200, 300, 500]),
                'max_depth':        trial.suggest_int('max_depth', 3, 10),
                'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
            model = XGBClassifier(
                scale_pos_weight=scale_pos,
                eval_metric='aucpr',
                random_state=42,
                use_label_encoder=False,
                n_jobs=1,
                **params,
            )
            return cv_f2(model, X_train, y_train_arr)

        xgb_study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)

        xgb_best_params = xgb_study.best_params
        xgb_best = XGBClassifier(**xgb_best_params).fit(X_train, y_train_arr)

        logging.info("✅ Modelo de XGBoost creado.")
        logging.info("✅ Parámetros del modelo de XGBoost:")

        X_val = pd.read_csv(X_val_path)
        y_val = pd.read_csv(y_val_path)
        y_val_arr = y_val.to_numpy().ravel()

        y_prob_xgb_val = xgb_best.predict_proba(X_val)
        y_pred_xgb_val = xgb_best.predict(X_val)

        bl_metrics = {
            'AUC-ROC':   roc_auc_score(y_val_arr, y_prob_xgb_val[:, 1]),
            'PR-AUC':    average_precision_score(y_val_arr, y_prob_xgb_val[:, 1]),
            'F1':        f1_score(y_val_arr, y_pred_xgb_val),
            METRIC_NAME: fbeta_score(y_val_arr, y_pred_xgb_val, beta=BETA),
            'Recall':    recall_score(y_val_arr, y_pred_xgb_val),
            'Precision': precision_score(y_val_arr, y_pred_xgb_val, zero_division=0),
        }

        matrix_plot = plot_confusion_matrix(y_val_arr, y_pred_xgb_val, save_path=None)
        roc_plots = plot_roc_curve(y_val_arr, y_prob_xgb_val, save_path=None)
        pr_plots = plot_precision_recall(y_val_arr, y_prob_xgb_val[:, 1], save_path=None)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        with mlflow.start_run(experiment_id=experiment_id,
                              run_name=run_name_base,
                              tags={"model":"xgboost", "type":"evaluation"}):
            logging.info("✅ Log de modelo XGBoost en MLflow")
            mlflow.log_params(xgb_best_params)
            mlflow.sklearn.log_model(sk_model=xgb_best, name="xgboost_model")
            mlflow.log_metrics(bl_metrics)
            logging.info("✅ Modelo XGBoost y métricas registrados en MLflow")
            mlflow.log_figure(matrix_plot, artifact_file="xgb_confusion_matrix.png")
            mlflow.log_figure(roc_plots[0], artifact_file="xgb_roc_curve_1.png")
            mlflow.log_figure(roc_plots[1], artifact_file="xgb_roc_curve_2.png")
            mlflow.log_figure(pr_plots, artifact_file="xgb_precision_recall_curve.png")

        logging.info("✅ Tarea modelo XGBoost finalizada.")
        return bl_metrics

    @task.python(task_id="create_random_forest_model", multiple_outputs=True)
    def create_random_forest_model(X_train_path, y_train_path, X_val_path, y_val_path):
        """
        Creamos un modelo de bosque aleatorio para comparar con el modelo entrenado posteriormente.
        En este caso, un clasificador que usa múltiples características como predictors.
        """
        from sklearn.ensemble import RandomForestClassifier
        import optuna
        import mlflow
        from utils.plots import plot_confusion_matrix, plot_roc_curve, plot_precision_recall

        logging.info("Iniciando tarea modelo de random forest...")

        experiment_id = create_experiment("random_forest_model_evaluation", model="random_forest")

        logging.info(f"Usando experimento MLflow: random_forest_model_evaluation (ID: {experiment_id})")
        run_name_base = 'random_forest_model_exp'

        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path)
        y_train_arr = y_train.to_numpy().ravel()

        def rf_objective(trial):
            params = {
                'n_estimators':     trial.suggest_categorical('n_estimators', [100, 200, 300, 500]),
                'max_depth':        trial.suggest_categorical('max_depth', [5, 10, 15, 20, None]),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features':     trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]),
            }
            model = RandomForestClassifier(
                class_weight='balanced', random_state=42, n_jobs=-1, **params
            )
            return cv_f2(model, X_train, y_train)

        rf_study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        rf_study.optimize(rf_objective, n_trials=50, show_progress_bar=True)

        rf_best_params = rf_study.best_params
        rf_best = RandomForestClassifier(**rf_best_params).fit(X_train, y_train_arr)

        logging.info("✅ Modelo de bosque aleatorio creado.")
        logging.info("✅ Parámetros del modelo de bosque aleatorio:")
        mlflow.log_params(rf_best_params)

        X_val = pd.read_csv(X_val_path)
        y_val = pd.read_csv(y_val_path)
        y_val_arr = y_val.to_numpy().ravel()

        y_prob_rf_val = rf_best.predict_proba(X_val)
        y_pred_rf_val = rf_best.predict(X_val)

        bl_metrics = {
            'AUC-ROC':   roc_auc_score(y_val_arr, y_prob_rf_val[:, 1]),
            'PR-AUC':    average_precision_score(y_val_arr, y_prob_rf_val[:, 1]),
            'F1':        f1_score(y_val_arr, y_pred_rf_val),
            METRIC_NAME: fbeta_score(y_val_arr, y_pred_rf_val, beta=BETA),
            'Recall':    recall_score(y_val_arr, y_pred_rf_val),
            'Precision': precision_score(y_val_arr, y_pred_rf_val, zero_division=0),
        }

        matrix_plot = plot_confusion_matrix(y_val_arr, y_pred_rf_val, save_path=None)
        roc_plots = plot_roc_curve(y_val_arr, y_prob_rf_val, save_path=None)
        pr_plots = plot_precision_recall(y_val_arr, y_prob_rf_val[:, 1], save_path=None)
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        with mlflow.start_run(experiment_id=experiment_id,
                              run_name=run_name_base,
                              tags={"model":"random_forest", "type":"evaluation"}):
            logging.info("✅ Log de modelo de random forest en MLflow")
            mlflow.sklearn.log_model(sk_model=rf_best, name="random_forest_model")
            mlflow.log_metrics(bl_metrics)
            logging.info("✅ Modelo de random forest y métricas registrados en MLflow")
            mlflow.log_figure(matrix_plot, artifact_file="rf_confusion_matrix.png")
            mlflow.log_figure(roc_plots[0], artifact_file="rf_roc_curve_1.png")
            mlflow.log_figure(roc_plots[1], artifact_file="rf_roc_curve_2.png")
            mlflow.log_figure(pr_plots, artifact_file="rf_precision_recall_curve.png")

        logging.info("✅ Tarea modelo de random forest finalizada.")
        return bl_metrics

    # 🧩 Encadenamiento
    paths = check_data_to_process()
    create_base_model(paths["X_val_path"], paths["y_val_path"])
    create_knn_model(paths["X_train_path"], paths["y_train_path"], paths["X_val_path"], paths["y_val_path"])
    create_decision_tree_model(paths["X_train_path"], paths["y_train_path"], paths["X_val_path"], paths["y_val_path"])
    create_xgboost_model(paths["X_train_path"], paths["y_train_path"], paths["X_val_path"], paths["y_val_path"])
    create_random_forest_model(paths["X_train_path"], paths["y_train_path"], paths["X_val_path"], paths["y_val_path"])

dag = process_etl_taskflow()