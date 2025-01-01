import requests
import os
import json
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

with open('config.json', 'r') as f:
    config = json.load(f)


test_data_path = config['test_data_path']
data_filename = 'testdata.csv'
data_path = os.path.join(test_data_path, data_filename)
logging.info(f"Apicalls: Data path: {data_path}")


#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"
logging.info(f"Apicalls: URL: {URL}")



#Call each API endpoint and store the responses
response1 = requests.post(
    f"{URL}/prediction",
    json={"file_path": data_path}
).json()
response2 = requests.get(f"{URL}/scoring").json()
response3 = requests.get(f"{URL}/summarystats").json()
response4 = requests.get(f"{URL}/diagnostics").json()

#combine all API responses
responses = {'prediction': response1, 
             'scoring': response2, 
             'summary_stats': response3, 
             'diagnostics': response4}

#write the responses to your workspace
save_path = os.path.join(config['output_model_path'], 'apireturns.txt')
with open(save_path, 'w') as f:
    for key, value in responses.items():
        f.write(f"{key}: {value}\n")

logging.info(f"Apicalls: API responses saved to {save_path}")


