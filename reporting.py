import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions
import logging  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
logging.info(f"Dataset CSV path: {dataset_csv_path}")

test_data_path = os.path.join(config['test_data_path']) 
logging.info(f"Test data path: {test_data_path}")

output_model_path = os.path.join(config['output_model_path'])
logging.info(f"Output model path: {output_model_path}")






##############Function for reporting
def score_model(df):
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
    
    logging.info(f"Confusion matrix saved to {output_model_path}/confusion_matrix.png")






if __name__ == '__main__':
    #load test data
    logging.info(f"Files in dataset_csv_path: {os.listdir(test_data_path)}")
    data_filename = 'testdata.csv'
    data_path = os.path.join(test_data_path, data_filename)
    logging.info(f"Data path: {data_path}")
    try:
        test_data = pd.read_csv(data_path)
        logging.info(f"Test data shape: {test_data.shape}")
    except Exception as e:
        logging.error(f"Error loading test data: {e}")

    score_model(test_data)
