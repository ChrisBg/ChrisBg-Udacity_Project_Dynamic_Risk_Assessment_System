import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for training the model
def train_model():
    # Create model directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    logging.info("Training: Training model")

    # read the data
    data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    logging.info(f"Training: Data shape: {data.shape}")

    logging.info(f"Training:    Infos about the data: {data.info()}")
    # split the data into X and y 
    X = data.drop(['exited', 'corporation'], axis=1)
    y = data['exited']
    logging.info(f"Training: X shape: {X.shape}")
    logging.info(f"Training: y shape: {y.shape}")

    # use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    model.fit(X, y)
    logging.info("Training: Model trained")

    #write the trained model to your workspace in a file called trainedmodel.pkl
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'wb') as f:
        pickle.dump(model, f)   
    logging.info("Training: Model saved")
     
if __name__ == "__main__":
    train_model()
    
