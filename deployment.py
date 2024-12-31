from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging  
from glob import glob
import shutil


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_folder_path = os.path.join(config['output_folder_path']) 
logging.info(f"Dataset folder path: {dataset_folder_path}")
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
logging.info(f"Prod deployment path: {prod_deployment_path}")
model_folder_path = os.path.join(config['output_model_path']) 
logging.info(f"Model folder path: {model_folder_path}")
#logging.info(f"Model path: {model_path}")

####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    # Create the deployment directory if it doesn't exist
    os.makedirs(prod_deployment_path, exist_ok=True)

    # Copy the latest pickle file
    logging.info(f"Model path: {model_folder_path}")
    logging.info(f"Model files: {os.listdir(model_folder_path)}")
    model_file_path = glob(os.path.join(model_folder_path, '*.pkl'))
    if not model_file_path:
        logging.error("No pickle files found in model directory")
        return
    model_file_name = os.path.basename(model_file_path[0])  # gets just the filename
    logging.info(f"Model file name: {model_file_name}")

    deployment_model_path = os.path.join(prod_deployment_path, model_file_name)
    logging.info(f"Deployment model path: {deployment_model_path}")
    shutil.copy(model_file_path[0], deployment_model_path)

    # Copy the latestscore.txt file     
    score_file_path = glob(os.path.join(model_folder_path, '*.txt'))
    if not score_file_path:
        logging.error("No score files found in model directory")
        return
    score_file_name = os.path.basename(score_file_path[0])  # gets just the filename    
    logging.info(f"Score file name: {score_file_name}")
    score_file_path = os.path.join(model_folder_path, score_file_name)
    logging.info(f"Score file path: {score_file_path}")
    deployment_score_path = os.path.join(prod_deployment_path, score_file_name)
    logging.info(f"Deployment score path: {deployment_score_path}")
    shutil.copy(score_file_path, deployment_score_path)

    # Copy the ingestedfiles.txt file
    ingested_files_path = glob(os.path.join(dataset_folder_path, '*.txt'))
    logging.info(f"Ingested files path: {ingested_files_path}")

    if not ingested_files_path:
        logging.error("No ingested files found in dataset directory")
        return
        
    ingested_files_name = os.path.basename(ingested_files_path[0])  # gets just the filename
    logging.info(f"Ingested files name: {ingested_files_name}")
    deployment_ingested_path = os.path.join(prod_deployment_path, ingested_files_name)
    logging.info(f"Deployment ingested path: {deployment_ingested_path}")
    shutil.copy(ingested_files_path[0], deployment_ingested_path)

    logging.info(f"Model, score, and ingested files copied to deployment directory")    
        

if __name__ == "__main__":
    store_model_into_pickle()