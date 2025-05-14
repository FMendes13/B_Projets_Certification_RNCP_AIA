# Library
import pandas as pd 
import requests
import json
import numpy as np
import logging
import psycopg2
from datetime import datetime
from airflow import DAG
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
import mlflow
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import os
import pickle
from airflow.models import Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Variables
run_id = Variable.get("MLFLOW_RUN_ID")
mlflow_uri= Variable.get("MLFLOW_URI")
postegresql_uri= Variable.get("POSTGRESQL_URI")
api_url = Variable.get("FRAUD_API_URL")

# Fonctions


def fetch_and_store_raw_data_s3(**context):
    url = 'https://charlestng-real-time-fraud-detection.hf.space/current-transactions'
    response = requests.get(url)

    if response.status_code == 200:
        try:
            # Nettoyage et conversion en JSON
            text_response = response.text.strip().strip('"').replace('\\"', '"')
            data = json.loads(text_response)

            # Vérifier si les clés 'data' et 'columns' existent
            if 'data' in data and 'columns' in data:
                df = pd.DataFrame(data['data'], columns=data['columns'])

                # Générer un nom de fichier unique
                raw_filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_fraud_detection.json"
                local_path = f"/tmp/{raw_filename}"
                
                # Sauvegarde locale
                df.to_json(local_path, orient="records")
                logging.info(f"Données enregistrées localement : {local_path}")

                # Envoi vers S3
                s3_hook = S3Hook(aws_conn_id="aws_s3")
                s3_hook.load_file(
                    filename=local_path, key=f"raw_data/{raw_filename}", bucket_name="fraud-detection-api-airflow"
                )
                logging.info(f"Données envoyées vers S3 : {raw_filename}")

                # Envoi du chemin vers XCom pour les prochaines tâches
                context["task_instance"].xcom_push(key="raw_s3_filename", value=f"raw_data/{raw_filename}")

            else:
                logging.error("Les clés 'data' et 'columns' sont absentes dans la réponse de l'API.")
        except json.JSONDecodeError as e:
            logging.error(f"Erreur lors de la conversion en JSON : {e}")
            logging.error(f"Contenu de la réponse : {response.text[:1000]}")
    else:
        logging.error(f"Échec de la requête API, statut: {response.status_code}, réponse: {response.text[:1000]}")


def cleaning_and_store_neondb(**context):
    conn = None
    cursor = None
    try:
        # 1. Configuration initiale
        postgresql_uri = Variable.get("POSTGRESQL_URI")
        conn = psycopg2.connect(postgresql_uri)
        cursor = conn.cursor()

        # 2. Création de table avec colonnes explicites
        cursor.execute("""
        DROP TABLE IF EXISTS fraud_detection;
        CREATE TABLE fraud_detection (
            cc_num BIGINT,
            merchant TEXT,
            category TEXT,
            amt FLOAT,
            first TEXT,
            last TEXT,
            gender TEXT,
            city TEXT,
            state TEXT,
            zip INTEGER,
            city_pop INTEGER,
            job TEXT,
            merch_lat FLOAT,
            merch_long FLOAT,
            is_fraud INTEGER,
            day SMALLINT,
            month SMALLINT,
            year SMALLINT,
            hour SMALLINT,
            minute SMALLINT,
            model_prediction INTEGER,
            PRIMARY KEY (cc_num, merchant, amt, hour, minute)
        );
        """)
        conn.commit()

        # 3. Traitement des données
        s3_key = context["task_instance"].xcom_pull(key="raw_s3_filename", task_ids='fetch_and_store_raw_data_s3')
        s3_hook = S3Hook(aws_conn_id="aws_s3")
        local_path = s3_hook.download_file(key=s3_key, bucket_name="fraud-detection-api-airflow", local_path="/tmp")
        
        with open(local_path, 'r') as f:
            raw_data = json.load(f)
        
        # Conversion en DataFrame
        if isinstance(raw_data, list):
            df = pd.DataFrame(raw_data)
        elif 'data' in raw_data and 'columns' in raw_data:
            df = pd.DataFrame(raw_data['data'], columns=raw_data['columns'])
        else:
            df = pd.DataFrame(raw_data)

        # Feature engineering
        now = datetime.now()
        df['day'] = now.day
        df['month'] = now.month
        df['year'] = now.year
        df['hour'] = now.hour
        df['minute'] = now.minute
        df['model_prediction'] = None

        # Colonnes à insérer (doivent correspondre exactement à la table)
        columns_to_insert = [
            'cc_num', 'merchant', 'category', 'amt', 'first', 'last', 'gender',
            'city', 'state', 'zip', 'city_pop', 'job', 'merch_lat', 'merch_long',
            'is_fraud', 'day', 'month', 'year', 'hour', 'minute', 'model_prediction'
        ]

        # Vérification des colonnes
        missing_cols = [col for col in columns_to_insert if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes dans DataFrame: {missing_cols}")

        # 4. Insertion avec vérification de structure
        from psycopg2.extras import execute_values
        execute_values(
            cursor,
            """
            INSERT INTO fraud_detection (
                cc_num, merchant, category, amt, first, last, gender,
                city, state, zip, city_pop, job, merch_lat, merch_long,
                is_fraud, day, month, year, hour, minute, model_prediction
            ) VALUES %s
            """,
            df[columns_to_insert].values.tolist(),
            page_size=100
        )
        conn.commit()
        logging.info(f"Insertion réussie: {len(df)} lignes")

    except Exception as e:
        logging.error(f"ERREUR: {str(e)}", exc_info=True)
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def load_model_and_predict():
    try:
        # 1. Chargement du modèle et du préprocesseur
        model_path = '/opt/airflow/data/xgb_model_v2.pkl'
        preprocessor_path = '/opt/airflow/data/preprocessor.pkl'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError("Le fichier modèle est introuvable.")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError("Le fichier preprocessor est introuvable.")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        logging.info("Modèle et pipeline de preprocessing chargés avec succès depuis les fichiers pickle")

        # 2. Connexion PostgreSQL et récupération des données
        postgresql_uri = Variable.get("POSTGRESQL_URI")
        with psycopg2.connect(postgresql_uri) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                SELECT cc_num, merchant, category, amt, gender, city_pop,
                       merch_lat, merch_long, day, month, year, hour, minute
                FROM fraud_detection
                WHERE model_prediction IS NULL
                ORDER BY year DESC, month DESC, day DESC, hour DESC, minute DESC
                LIMIT 1
                """)
                
                transaction = cursor.fetchone()
                if not transaction:
                    logging.info("Aucune nouvelle transaction à prédire")
                    return

                # 3. Préparation des données
                features = pd.DataFrame([transaction], columns=[
                    'cc_num', 'merchant', 'category', 'amt', 'gender',
                    'city_pop', 'merch_lat', 'merch_long',
                    'day', 'month', 'year', 'hour', 'minute'
                ])
                
                # Création d'une copie pour la prédiction (sans cc_num)
                features_for_prediction = features.drop(columns=['cc_num']).copy()
                
                # Vérifier que les colonnes correspondent au préprocesseur
                expected_columns = list(preprocessor.feature_names_in_)
                missing_cols = set(expected_columns) - set(features_for_prediction.columns)
                
                # Ajouter les colonnes manquantes avec des valeurs par défaut
                for col in missing_cols:
                    if col in ['merchant', 'category', 'gender', 'first', 'last', 'job', 'city', 'state']:
                        features_for_prediction[col] = "unknown"  # Valeur par défaut pour les catégoriques
                    else:
                        features_for_prediction[col] = 0  # Valeur par défaut pour les numériques
                
                # Réorganisation des colonnes
                features_for_prediction = features_for_prediction.reindex(columns=expected_columns)
                
                # Correction des types pour éviter l'erreur 'ufunc isnan'
                categorical_cols = ['merchant', 'category', 'gender', 'first', 'last', 'job', 'city', 'state']
                numeric_cols = [col for col in expected_columns if col not in categorical_cols]
                
                for col in categorical_cols:
                    if col in features_for_prediction.columns:
                        features_for_prediction[col] = features_for_prediction[col].astype(str).fillna("unknown")
                
                for col in numeric_cols:
                    if col in features_for_prediction.columns:
                        features_for_prediction[col] = pd.to_numeric(features_for_prediction[col], errors='coerce').fillna(0).astype(float)
                
                # Application du pipeline de preprocessing
                features_transformed = preprocessor.transform(features_for_prediction)
                prediction = model.predict(features_transformed)[0]
                is_fraud = int(prediction >= 0.5)
                
                # 5. Mise à jour de la prédiction dans la base
                cursor.execute("""
                UPDATE fraud_detection
                SET model_prediction = %s
                WHERE cc_num = %s AND merchant = %s AND amt = %s
                  AND hour = %s AND minute = %s
                """, (is_fraud, transaction[0], transaction[1], transaction[3], 
                           transaction[-2], transaction[-1]))
                
                conn.commit()
                logging.info(f"Prédiction enregistrée: {'FRAUDE' if is_fraud else 'NORMAL'}")

    except Exception as e:
        logging.error(f"ERREUR: {str(e)}", exc_info=True)
        raise


# DAG Definition
default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 3, 24),
    "catchup": False,
}

with DAG(
    "fraud_detection_pipeline",
    default_args=default_args,
    schedule_interval="*/5 * * * *",  # Exécution toutes les 5 minutes
) as dag:

    start = EmptyOperator(task_id="start")

    fetch_data_task = PythonOperator(
        task_id='fetch_and_store_raw_data_s3',
        python_callable=fetch_and_store_raw_data_s3,
        provide_context=True,
    )

    clean_data_task = PythonOperator(
        task_id='cleaning_and_store_neondb',
        python_callable=cleaning_and_store_neondb,
        provide_context=True,
    )

    predict_task = PythonOperator(
        task_id='load_and_predict',
        python_callable=load_model_and_predict,
    )

    end = EmptyOperator(task_id="end")

    start >> fetch_data_task >> clean_data_task >> predict_task >> end
