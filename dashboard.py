import streamlit as st
import pandas as pd
import requests

# URL de l'API MLflow
API_URL = "http://127.0.0.1:5000/invocations"

# Définition de la fonction pour charger les données du client
@st.cache_data
def load_client_data(client_id):
    # Charger les données complètes (adapté à votre chemin de fichier)
    data = pd.read_csv(r"C:\Users\jerom\Formation OC\Projet 7\application_train.csv")
    # Sélectionner les données du client spécifique
    client_data = data[data['SK_ID_CURR'] == int(client_id)]
    # Retirer les colonnes non pertinentes
    client_data = client_data.drop(columns=['SK_ID_CURR', 'TARGET'])
    return client_data

# Fonction pour prédire le score du client et décider d'accepter ou refuser le prêt
def predict_loan_approval(client_id):
    # Chargez les données du client
    client_data = load_client_data(client_id)
    
    if client_data.empty:
        st.error(f"Aucune donnée trouvée pour l'identifiant client {client_id}. Veuillez vérifier l'identifiant.")
        return None, None
    
    # Préparez les données pour la prédiction
    client_data_json = client_data.to_json(orient='split')
    
    # Faire une requête POST à l'API MLflow
    response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=client_data_json)
    
    if response.status_code == 200:
        prediction = response.json()["predictions"][0][1]  # Score de probabilité de défaut
        
        # Décision d'acceptation ou de refus du prêt
        if prediction > 0.5:
            loan_decision = "Refusé"
        else:
            loan_decision = "Accepté"
        
        return prediction, loan_decision
    else:
        st.error("Erreur lors de la requête à l'API MLflow.")
        return None, None

# Code principal de l'application Streamlit
def main():
    st.title("Simulation de scoring client pour l'acceptation de prêt")
    
    # Saisie de l'identifiant client
    client_id = st.text_input("Entrez l'identifiant du client:")
    
    if client_id:
        try:
            client_id = int(client_id)  # Convertir l'entrée en entier
            st.write(f"Identifiant client saisi : {client_id}")
            
            # Prédire le score du client et décider de l'acceptation ou du refus du prêt
            prediction, decision = predict_loan_approval(client_id)
            
            if prediction is not None:
                st.write(f"Score de probabilité de défaut : {prediction:.2f}")
                st.write(f"Décision de prêt : {decision}")
            
        except ValueError:
            st.error("Veuillez entrer un identifiant client valide (nombre entier).")

# Exécuter l'application Streamlit
if __name__ == "__main__":
    main()



