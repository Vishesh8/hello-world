import streamlit as st
import requests
import json
import time
import os
import cloudpickle
from urllib.parse import urlparse
import boto3
import mlflow

refresh_seconds = 5


def sklearn_log(model_artifacts_dir, workspace_dir, model):

  mlflow.set_experiment(workspace_dir)
  with mlflow.start_run() as run:
    mlflow.sklearn.log_model(
      sk_model = model,
      artifact_path="model"
      )
    
    mlflow.log_artifacts(model_artifacts_dir, "model")
  
  return run.info.experiment_id, run.info.run_id

def log_experiment(model_artifacts_dir, workspace_dir, model_type):
  
  host = os.environ.get('HOST')
  with st.spinner("Logging Experiment..."):
    url = urlparse(model_artifacts_dir, allow_fragments=False)
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(url.netloc)
    
    files = [file.key for file in bucket.objects.filter(Prefix=url.path[1:])]
    pkl_file = [pkl for pkl in files if pkl.endswith('.pkl')][0]
    
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket.name, Key=pkl_file)
    model = cloudpickle.loads(response['Body'].read())
    experiment_id, experiment_run_id = model_type_functions.get(model_type)(model_artifacts_dir, workspace_dir, model)
    
    time.sleep(refresh_seconds)
  
  # Check the response status
  st.success("Experiment logged successfully!")
  experiment_url = f"https://{host}/ml/experiments/{experiment_id}/runs/{experiment_run_id}"
  st.markdown(f"[Click here to visit the experiment page]({experiment_url})")
  return experiment_id, experiment_run_id

def register_model_version(model_name, experiment_id, experiment_run_id):
  host = os.environ.get('HOST')
  token = os.environ.get('TOKEN')

  with st.spinner("Registering Model Version..."):
    url = f"https://{host}/api/2.0/mlflow/model-versions/create"
    
    payload = json.dumps({
      "name": model_name,
      "source": f"dbfs:/databricks/mlflow-tracking/{experiment_id}/{experiment_run_id}/artifacts/model",
      "run_id": experiment_run_id
    })

    headers = {
      'Content-Type': 'application/json',
      'Authorization': f'Bearer {token}'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    time.sleep(refresh_seconds)

  # Check the response status
  if response.status_code == 200:
    model_version = response.json()['model_version']['version']
    st.success(f"Model version {model_version} registered successfully!")
    model_url = f"https://{host}/ml/models/{model_name}/versions/{model_version}"
    st.markdown(f"[Click here to visit the model version page]({model_url})")
  else:
    st.error(f"Failed to register the model version. Error: {response.text}")

def register_model(model_name, experiment_id, experiment_run_id):
  host = os.environ.get('HOST')
  token = os.environ.get('TOKEN')

  with st.spinner("Registering Model..."):
    url = f"https://{host}/api/2.0/mlflow/registered-models/create"
    
    payload = json.dumps({
      "name": model_name
    })

    headers = {
      'Content-Type': 'application/json',
      'Authorization': f'Bearer {token}'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    time.sleep(refresh_seconds)
    
  # Check the response status
  if response.status_code == 200:
    st.success("Model registered successfully!")
    register_model_version(model_name, experiment_id, experiment_run_id)
  elif response.json()['error_code'] == 'RESOURCE_ALREADY_EXISTS':
    st.warning(f"Skipping model registration as it already exists")
    register_model_version(model_name, experiment_id, experiment_run_id)
  else:
    st.error(f"Failed to register model. Error: {response.text}")


def main():
  #App title
  st.set_page_config(page_title="Model Registration App")
  st.title("🤖 Register ML Models on Databricks")

  #Databricks Authentication
  with st.sidebar:
    st.subheader('Databricks Login Details')
    st.write( 'This app is created using the Databricks Model APIs')
    host = st.text_input('Enter Databricks Host:') #e2-demo-field-eng.cloud.databricks.com
    token = st.text_input('Enter Databricks API token:', type='password')
    os.environ['TOKEN'] = token
    os.environ['HOST'] = host
  
  #Model Artifacts Location
  artifact_location = st.selectbox("Model Artifact Location", ["mlflow", "S3"])

  #Define External Experiment
  if artifact_location == "S3":
    s3_model_name = st.text_input("Model Name:") #sklearn_model_price_api
    model_artifacts_dir = st.text_input("Model Artifacts Directory:") #s3://one-env-uc-external-location/vishesh_ext/model
    workspace_dir = st.text_input("Workspace Directory:") #/Users/vishesh.arya@databricks.com/tests/sklearn-experiment
    model_type = st.selectbox("Model Type:", ['sklearn']) #sklearn

  elif artifact_location == "mlflow":
    model_name = st.text_input("Model Name:") #sklearn_model_price_api
    experiment_id = st.text_input("Experiment ID:") #385797078888197
    experiment_run_id = st.text_input("Experiment Run ID:") #02a2a23836984f3190e0eb8d3ca697c1

  #Register button
  if st.button("Register"):
    if artifact_location == "S3":
      if not host or not token or not s3_model_name or not model_artifacts_dir or not workspace_dir or not model_type:
        st.warning("Please fill in all fields.")
      else:
        experiment_id, experiment_run_id = log_experiment(model_artifacts_dir, workspace_dir, model_type)
        register_model(s3_model_name, experiment_id, experiment_run_id)
      
    elif artifact_location == "mlflow":
      if not host or not token or not model_name or not experiment_id or not experiment_run_id:
        st.warning("Please fill in all fields.")
      else:
        register_model(model_name, experiment_id, experiment_run_id)

if __name__ == "__main__":
  model_type_functions = {'sklearn': sklearn_log}

  main()