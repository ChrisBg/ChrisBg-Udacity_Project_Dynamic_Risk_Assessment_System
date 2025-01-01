# Importing libraries
import os
from ingestion import merge_multiple_dataframe
from diagnostics import model_predictions, dataframe_summary, missing_data, execution_time, outdated_packages_list
from deployment import store_model_into_pickle
from training import train_model
from scoring import score_model
from sklearn import metrics
from reporting import cm_model
import json
import logging
import pandas as pd

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Loading configuration
with open('config.json','r') as f:
    config = json.load(f) 

# Modifying config 
config['input_folder_path'] = 'sourcedata'
config['output_model_path'] = 'models'

# Saving updated config
logging.info(f"Fullprocess: Config: {config}")
with open('config.json','w') as f:
    json.dump(config, f)
logging.info("Fullprocess: Config updated successfully")

# Define the paths
input_folder_path = os.path.join(config['input_folder_path'])
logging.info(f"Fullprocess: Input folder path: {input_folder_path}")
output_folder_path = os.path.join(config['output_folder_path'])
logging.info(f"Fullprocess: Output folder path: {output_folder_path}")
output_model_path = os.path.join(config['output_model_path'])
logging.info(f"Fullprocess: Output model path: {output_model_path}")

# Main function
def main():
    ##################Check and read new data
    #first, read ingestedfiles.txt
    deployment_path = os.path.join(config['prod_deployment_path'])
    logging.info(f"Fullprocess: Deployment path: {deployment_path}")
    deployment_data = os.path.join(deployment_path, 'ingestedfiles.txt')
    logging.info(f"Fullprocess: Deployment data: {deployment_data}") 

    # Reading ingestedfiles.txt
    try:
        with open(deployment_data, 'r') as file:
            first_line = file.readline().strip()
            ingested_data = json.loads(first_line) 
            logging.info(f"Fullprocess: Ingested data: {ingested_data}")
    except Exception as e:
        logging.error(f"Fullprocess: Error reading ingestedfiles.txt: {e}")
        return  

    # Extracting old CSV files
    old_csv_files = ingested_data['ingested_files'] 
    logging.info(f"Fullprocess: Old CSV files: {old_csv_files}")


    # Checking if there are new CSV files
    input_folder_path = os.path.join(config['input_folder_path'])
    logging.info(f"Fullprocess: Input folder path: {input_folder_path}")

    # Extracting new CSV files
    new_csv_files = [f for f in os.listdir(input_folder_path) if f.endswith('.csv')]
    logging.info(f"Fullprocess: New CSV files found: {new_csv_files}")

    # Checking for new CSV files
    new_csv = set(new_csv_files) - set(old_csv_files)
    logging.info(f"Fullprocess: New CSV files: {new_csv}")

    # Proceeding with the process if there are new CSV files
    if new_csv:
        logging.info("Fullprocess: New data found. Proceeding with the process.")
    else:
        logging.info("Fullprocess: No new data found. Ending the process.")
        exit()

    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here
    merge_multiple_dataframe()
    logging.info("Fullprocess: Merge multiple dataframe completed")

    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    deployed_score_file = os.path.join(deployment_path, 'latestscore.txt')
    logging.info(f"Full process: Deployed score file: {deployed_score_file}")

    # Reading the deployed score
    with open(deployed_score_file, 'r') as file:
        deployed_score = float(file.read().strip())
    logging.info(f"Full process: Deployed score: {deployed_score}")

    # Reading the new data
    new_data_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
    logging.info(f"Full process: New data path: {new_data_path}")
    new_data = pd.read_csv(new_data_path)
    logging.info(f"Full process: New data: {new_data.head()}")

    # Making predictions from new data
    predictions = model_predictions(new_data)
    logging.info(f"Full process: Model predictions: {predictions}")

    # Calculating F1 score from new data
    y_test = new_data['exited']
    f1_score = metrics.f1_score(y_test, predictions)
    logging.info(f"Full process: F1 score: {f1_score}")

    # Checking for model drift
    if f1_score < deployed_score:
        logging.info("Full process: Model drift detected. Proceeding with re-deployment.")
    else:
        logging.info("Full process: No model drift detected. Ending the process.")
        exit()

    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here
    
    # Re-training the model
    train_model()
    logging.info("Full process: Model re-training completed")

    # Re-scoring the model
    score_model()
    logging.info("Full process: Model re-scoring completed")

    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
    store_model_into_pickle()
    logging.info("Full process: Model re-deployment completed")

    ##################Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model
    cm_model()
    logging.info("Full process: Confusion matrix completed")
    dataframe_summary()
    logging.info("Full process: Dataframe summary completed")
    missing_data()
    logging.info("Full process: Missing data completed")
    execution_time()
    logging.info("Full process: Execution time completed")
    outdated_packages_list()
    logging.info("Full process: Outdated packages list completed")

if __name__ == "__main__":
    main()





