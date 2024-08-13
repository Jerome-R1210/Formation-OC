from flask import Flask, request, jsonify
import lightgbm as lgb
import pandas as pd
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained LightGBM model
model_path = r"C:\Users\jerom\Formation OC\Projet 7\pipeline_p7.joblib"
model = joblib.load(model_path)

# Function to fetch client data based on client_id
def get_client_data(client_id):
    # Load the client data from CSV
    client_data = pd.read_csv(r"C:\Users\jerom\Formation OC\Projet 7\app_train.csv")
    
    # Filter the client data based on client_id
    client_data = client_data[client_data['SK_ID_CURR'] == int(client_id)]
    
    # Check if the client data is found
    if client_data.empty:
        return None
    
    # Drop the unnecessary columns
    client_data = client_data.drop(columns=['TARGET'])
    
    return client_data

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
    app.run(debug=True, host='0.0.0.0', port=8080)

