import pandas as pd
import pickle
import os
from sklearn import metrics
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 
logging.info(f"Scoring: Model path: {model_path}")

#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    # Load the trained model
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)
    logging.info("Scoring: Model loaded successfully")
    
    # Load the test data
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    logging.info(f"Scoring: Test data shape: {test_data.shape}")

    # Split the test data into X and y
    X_test = test_data.drop(['exited', 'corporation'], axis=1)
    y_test = test_data['exited']
    logging.info(f"Scoring: X_test shape: {X_test.shape}")
    logging.info(f"Scoring: y_test shape: {y_test.shape}")

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    logging.info("Scoring: Predictions made successfully")      
    # Calculate the F1 score
    f1 = metrics.f1_score(y_test, y_pred)
    logging.info(f"Scoring: F1 score : {f1}")

    # Write the F1 score to the latestscore.txt file
    score_file_name = 'latestscore.txt'
    score_file_path = os.path.join(model_path, score_file_name)
    with open(score_file_path, 'w') as f:
        f.write(str(f1))
    logging.info("Scoring: F1 score written to latestscore.txt")
    
    return f1

if __name__ == "__main__":
    score_model()