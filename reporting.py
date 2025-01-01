# Importing libraries
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions
import logging  

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
logging.info(f"Reporting: Dataset CSV path: {dataset_csv_path}")

test_data_path = os.path.join(config['test_data_path']) 
logging.info(f"Reporting: Test data path: {test_data_path}")

output_model_path = os.path.join(config['output_model_path'])
logging.info(f"Reporting: Output model path: {output_model_path}")



##############Function for reporting
def cm_model(df):
    #calculate a confusion matrix using the test data and the deployed model
    y_pred = model_predictions(df)
    y_true = df['exited']
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    
    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', 
                xticklabels=['Predicted:0', 'Predicted:1'],
                yticklabels=['True:0', 'True:1'])
    plt.title('Confusion Matrix')
    
    # Save the plot
    plt.savefig(os.path.join(output_model_path, 'confusion_matrix.png'))
    plt.close()
    
    logging.info(f"Reporting: Confusion matrix saved to {output_model_path}/confusion_matrix.png")



if __name__ == '__main__':
    #load test data
    logging.info(f"Reporting: Files in dataset_csv_path: {os.listdir(test_data_path)}")
    data_filename = 'testdata.csv'
    data_path = os.path.join(test_data_path, data_filename)
    logging.info(f"Reporting: Data path: {data_path}")
    try:
        logging.info("Reporting: Loading test data")
        test_data = pd.read_csv(data_path)
        logging.info(f"Reporting: Test data shape: {test_data.shape}")
    except Exception as e:
        logging.error(f"Reporting: Error loading test data: {e}")

    cm_model(test_data)
    logging.info("Reporting: Confusion matrix completed")
