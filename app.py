# Importing necessary libraries
from flask import Flask, jsonify, request
import json
import os
import logging
import pandas as pd
import diagnostics
import scoring

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
logging.info(f"App: Dataset CSV path: {dataset_csv_path}")

prediction_model = None


####################### Welcome Endpoint
@app.route("/", methods=['GET','OPTIONS'])
def welcome():        
    return "Welcome to the API"


#######################Prediction Endpoint
@app.route("/prediction", methods=['GET', 'POST'])
def predict():        
    if request.method == 'GET':
        # For GET requests, use a default test file
        file_path = 'testdata/testdata.csv'
    else:  # POST
        file_path = request.json.get('file_path')
    
    logging.info(f"App: File path: {file_path}")
    data = pd.read_csv(file_path)
    prediction = diagnostics.model_predictions(data)
    return jsonify(prediction) 

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def get_score():        
    #check the score of the deployed model
    f1_score = scoring.score_model()
    logging.info(f"App: F1 score: {f1_score}")
    return {'f1_score': f1_score}

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def get_summary_stats():        
    #check means, medians, and modes for each column
    summary_stats = diagnostics.dataframe_summary()
    # Convert DataFrame to dictionary
    summary_stats_dict = summary_stats.to_dict()
    logging.info(f"App: Summary stats: {summary_stats_dict}")
    return jsonify(summary_stats_dict)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def get_diagnostics():        
    response_data = {
        'missing_data': diagnostics.missing_data(),
        'execution_time': diagnostics.execution_time(),
        'outdated_packages': diagnostics.outdated_packages_list()
    }
    logging.info(f"App: Diagnostics response: {response_data}")
    return jsonify(response_data)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
