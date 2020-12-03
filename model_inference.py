import pandas as pd
import requests
import json
dataset = pd.read_csv('data/final.csv')
dataset = dataset.iloc[:, dataset.columns != 'v6392']
data = dataset.sample(1)


all_cols = list(data.columns)
all_vals = data.values.tolist()

input_data = {"columns":all_cols,
              "data":all_vals}

response = requests.post(url='http://127.0.0.1:2125/invocations',
                         data=json.dumps(input_data),
                         headers={"Content-type": "application/json"}
                         )
response_json = json.loads(response.text)
print(response_json)