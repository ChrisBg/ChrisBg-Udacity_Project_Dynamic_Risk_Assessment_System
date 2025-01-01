import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 
logging.info("Ingestion: Config loaded successfully")
logging.info(f"Ingestion: Config: {config}")

input_folder_path = config['input_folder_path']
logging.info(f"Ingestion: Input folder path: {input_folder_path}")
output_folder_path = config['output_folder_path']
logging.info(f"Ingestion: Output folder path: {output_folder_path}")

#  get today's date
today = datetime.now().strftime("%Y-%m-%d")


#############Function for data ingestion
def merge_multiple_dataframe():

    # Create output directory if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)
    
    # check for datasets, compile them together, and write to an output file
    data_files = os.listdir(input_folder_path)
    logging.info(f"Ingestion: Data files: {data_files}")
    files_names = [file for file in data_files if file.endswith('.csv')]
    logging.info(f"Ingestion: Files names: {files_names}")


    if len(files_names) == 0:
        logging.error("Ingestion: No data files found")
        return
    
    # read the first file
    final_data = pd.read_csv(os.path.join(input_folder_path, files_names[0]))
    logging.info(f"Ingestion: Data Shape: {final_data.shape}")

    # read the rest of the files
    for file in files_names[1:]:
        data = pd.read_csv(os.path.join(input_folder_path, file))
        logging.info(f"Ingestion: Data Shape: {data.shape}")
        final_data = pd.concat([final_data, data])
        logging.info(f"Ingestion: Final Data Shape: {final_data.shape}")

    # drop duplicates
    final_data.drop_duplicates(inplace=True)
    logging.info(f"Ingestion: Final Data Shape after dropping duplicates: {final_data.shape}")

    file_name = 'finaldata.csv'
    final_data.to_csv(os.path.join(output_folder_path, file_name), index=False)
    logging.info("Ingestion: Data ingestion completed successfully")

    record_file_name = 'ingestedfiles.txt'
    record = {
        'ingestion_date': today,
        'file_name': file_name,
        'input_folder_path': input_folder_path,
        'output_folder_path': output_folder_path,
        'ingested_files': files_names
    }
    
    record_file_path = os.path.join(output_folder_path, record_file_name)    
    # Write record to file (overwrite mode)
    with open(record_file_path, 'w') as f:  # 'w' mode overwrites existing file
        f.write(json.dumps(record) + '\n')
    logging.info("Ingestion: Record file created/updated successfully")


if __name__ == '__main__':
    merge_multiple_dataframe()
