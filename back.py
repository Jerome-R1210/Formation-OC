import os
from flask import Flask, request, jsonify
import lightgbm as lgb
import pandas as pd
import joblib
import requests
from io import BytesIO

# Initialize the Flask application
app = Flask(__name__)

# Define the URL to download the model and data from GitHub
data_url = 'https://github.com/Jerome-R1210/Formation-OC/blob/18ba732f3d774bcde4cb1c93f2852615ce44ccc6/app_train_sample.csv'
model_url = 'https://github.com/Jerome-R1210/Formation-OC/blob/18ba732f3d774bcde4cb1c93f2852615ce44ccc6/pipeline_p7.joblib'

# Load the pre-trained LightGBM model from URL
def load_model_from_url(url):
    response = requests.get(url)
    model = joblib.load(BytesIO(response.content))
    return model

# Load the model
model = load_model_from_url(model_url)

# Function to fetch client data based on client_id
def get_client_data(client_id):

    # Load the client data from CSV
    client_data = pd.read_csv(data_url)
    
    # Filter the client data based on client_id
    client_data = client_data[client_data['SK_ID_CURR'] == int(client_id)]
    
    # Check if the client data is found
    if client_data.empty:
        return None
    
    # Drop the unnecessary columns
    client_data = client_data.drop(columns=['TARGET'])
    
    return client_data

@app.route('/', methods=['GET'])
def index():
    return "API is running. Please use /predict to make a prediction."


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the client_id from the request
        client_id = request.json.get('client_id')
        
        if client_id is None:
            return jsonify({'error': 'No client_id provided'}), 400

        # Fetch the client data
        client_data = get_client_data(client_id)
        
        if client_data is None:
            return jsonify({'error': f'No data found for client_id {client_id}'}), 404
        
        # Predict using the model
        prediction_proba = model.predict_proba(client_data)[:, 1][0]

        # Return the decision as a JSON response
        return jsonify({'client_id': client_id,'prediction': prediction_proba})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
