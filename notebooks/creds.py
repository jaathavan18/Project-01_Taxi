import mlflow
import os

# Set MLFlow tracking credentials
MLFLOW_TRACKING_URI = "https://dagshub.com/jaathavan18/new_york_taxi.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = 'jaathavan18'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '59c1c8d77037a8073e3639cae8918ddfc2970cab'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)