import os
from ingestion import merge_multiple_dataframe
import scoring
import deployment
import diagnostics
import reporting
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


with open('config.json','r') as f:
    config = json.load(f) 

config['input_folder_path'] = 'sourcedata'
#config['output_folder_path'] = 'ingesteddata'
config['output_model_path'] = 'models'

logging.info(f"Config: {config}")

# Define the paths
input_folder_path = os.path.join(config['input_folder_path'])
logging.info(f"Input folder path: {input_folder_path}")
output_folder_path = os.path.join(config['output_folder_path'])
logging.info(f"Output folder path: {output_folder_path}")
output_model_path = os.path.join(config['output_model_path'])
logging.info(f"Output model path: {output_model_path}")

##################Check and read new data
#first, read ingestedfiles.txt
deployment_path = os.path.join(config['prod_deployment_path'])
logging.info(f"Deployment path: {deployment_path}")
deployment_data = os.path.join(deployment_path, 'ingestedfiles.txt')
logging.info(f"Deployment data: {deployment_data}") 

old_ingested_files = []
with open(deployment_data, 'r') as file:
    first_line = file.readline().strip()
    ingested_data = json.loads(first_line)  # Parse the first line as JSON

logging.info(f"Ingested data: {ingested_data}")
old_csv_files = ingested_data['ingested_files']  # Access the key directly
logging.info(f"Old CSV files: {old_csv_files}")


#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
input_folder_path = os.path.join(config['input_folder_path'])
logging.info(f"Input folder path: {input_folder_path}")
#logging.info(f"Input files: {os.listdir(input_folder_path), "*.csv"}")

new_csv_files = [f for f in os.listdir(input_folder_path) if f.endswith('.csv')]
logging.info(f"New CSV files found: {new_csv_files}")

new_csv = set(new_csv_files) - set(old_csv_files)
logging.info(f"New CSV files: {new_csv}")

if new_csv:
    logging.info("New data found. Proceeding with the process.")
else:
    logging.info("No new data found. Ending the process.")
    exit()

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
merge_multiple_dataframe()

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here



##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model







