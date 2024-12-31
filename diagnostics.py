import pandas as pd
import numpy as np
import timeit
import os
import json
import logging
import pickle
import pip
from ingestion import merge_multiple_dataframe
from training import train_model
import subprocess
from pkg_resources import working_set
from tabulate import tabulate


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
logging.info(f"Dataset CSV path: {dataset_csv_path}")

test_data_path = os.path.join(config['test_data_path']) 
logging.info(f"Test data path: {test_data_path}")

deployment_path = os.path.join(config['prod_deployment_path'])
logging.info(f"Deployment path: {deployment_path}")

data_filename = 'testdata.csv'
data_path = os.path.join(test_data_path, data_filename)
logging.info(f"Data path: {data_path}")




##################Function to get model predictions
def model_predictions(df):
    '''
    Function to get the model predictions on the test data
    '''
    #read the deployed model and a test dataset, calculate predictions
    #logging.info(f"Deployment listdir: {os.listdir(deployment_path)}")
    model_name = [name for name in os.listdir(deployment_path) if '.pkl' in name][0]
    #logging.info(f"Model name: {model_name}")
    model_path = os.path.join(deployment_path, model_name)
    logging.info(f"Model path: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded successfully") 
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None
    
    # Split the dataframe into X and y
    X_test = df.drop(['exited', 'corporation'], axis=1)
    y_test = df['exited']
    logging.info(f"X_test shape: {X_test.shape}")
    logging.info(f"y_test shape: {y_test.shape}")

    # Make predictions on the test data
    predictions = model.predict(X_test)
    logging.info(f"Predictions shape: {predictions.shape}")
    #logging.info(f"Predictions: {type(predictions)}")
    try :
        assert df.shape[0] == predictions.shape[0], "Predictions shape does not match test data shape"
        logging.info("Predictions shape matches test data shape")
    except Exception as e:
        logging.error(f"Error matching predictions shape with test data shape: {e}")

    return predictions.tolist() #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    '''
    Function to get the summary statistics (mean, median, std) of the dataset
    '''
    #calculate summary statistics here
    data_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    logging.info(f"Data path: {data_path}")
    try:
        data = pd.read_csv(data_path)
        logging.info(f"Data shape: {data.shape}")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    stats = data[numerical_columns].agg(['mean', 'median', 'std']).transpose() # Only these stats in this order
    logging.info(f"Summary statistics: {stats}")

    #stats_path = os.path.join(deployment_path, 'summary_statistics.csv')
    #   try:
    #    stats.to_csv(stats_path, index=True)
    #    logging.info(f"Summary statistics saved to {stats_path}")
    #    logging.info("Computing & saving summary statistics successfully")
    #except Exception as e:
    #    logging.error(f"Error saving summary statistics: {e}")
    
    return stats



def missing_data():
    '''
    Function to get the percentage of missing data in the dataset
    '''
    #calculate missing data
    data_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    logging.info(f"Data path: {data_path}")
    data = pd.read_csv(data_path)
    logging.info(f"Data shape: {data.shape}")
    missing_data = data.isna().sum()
    #logging.info(f"Missing data: {missing_data}")
    percentage_missing = (missing_data / data.shape[0]) * 100
    #logging.info(f"Percentage missing: {percentage_missing}")
    #logging.info(f"Percentage missing type: {type(percentage_missing)}")
    #logging.info(f"Percentage missing values: {percentage_missing.values}")
    logging.info(f"Percentage missing list: {percentage_missing.tolist()}")
    #missing_data_path = os.path.join(deployment_path, 'missing_data.csv')
    #try:
    #    missing_data.to_csv(missing_data_path, index=True)
    #    logging.info(f"Missing data saved to {missing_data_path}")
    #    logging.info("Computing & saving missing data successfully")
    #except Exception as e:
    #    logging.error(f"Error saving missing data: {e}")
    logging.info("Computing shares of missing data successfully")
    return percentage_missing.tolist()

##################Function to get timings
def execution_time():
    '''
    Function to get the execution time of the ingestion and training processes
    '''
    
    ingestion_time = timeit.timeit(lambda: merge_multiple_dataframe(), number=1)
    logging.info(f"Ingestion time: {ingestion_time}")
    training_time = timeit.timeit(lambda: train_model(), number=1)
    logging.info(f"Training time: {training_time}")
    return [ingestion_time, training_time]

##################Function to check dependencies
def outdated_packages_list():
    '''
    Function to get a formatted table of outdated packages
    '''
    import subprocess
    from tabulate import tabulate
    
    # Get list of outdated packages using pip
    result = subprocess.run(['pip', 'list', '--outdated', '--format=json'], 
                          capture_output=True, text=True)
    packages = json.loads(result.stdout)
    
    # Format data for tabulate
    table_data = [[p['name'], p['version'], p['latest_version']] 
                 for p in packages]
    
    # Create and return table string
    return tabulate(
        table_data,
        headers=['Package', 'Version', 'Latest'],
        tablefmt='grid'
    )


if __name__ == '__main__':
    #load test data
    try:
        test_data = pd.read_csv(data_path)
        logging.info(f"Test data shape: {test_data.shape}")
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
    
    #get model predictions
    model_predictions(test_data)

    #get summary statistics
    dataframe_summary()

    #get missing data
    missing_data()

    #get timing
    execution_time()

    #get outdated packages
    outdated_packages_list()





    
