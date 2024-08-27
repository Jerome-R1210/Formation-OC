import streamlit as st
import requests

# URL de votre API Flask
API_URL = 'https://powerful-inlet-94213-f9415709d590.herokuapp.com/'

# Fonction pour faire une requête POST à l'API Flask
def predict_loan_approval(client_id):
    try:
        response = requests.post(API_URL, json={'client_id': client_id})
        
        if response.status_code == 200:
            result = response.json()
            prediction_proba = result.get('prediction')
            return prediction_proba
        else:
            st.error(f"Erreur lors de la prédiction : {response.json()['error']}")
            return None
        
    except Exception as e:
        st.error(f"Erreur lors de la requête à l'API : {str(e)}")
        return None

# Code principal de l'application Streamlit
def main():
    st.title("Simulation de scoring client pour l'acceptation de prêt")
    
    # Saisie de l'identifiant client
    client_id = st.text_input("Entrez l'identifiant du client:")
    
    if st.button("Prédire"):
        if client_id:
            try:
                client_id = int(client_id)  # Convertir l'entrée en entier
                st.write(f"Identifiant client saisi : {client_id}")
                
                # Appel à l'API Flask pour obtenir la probabilité
                prediction_proba = predict_loan_approval(client_id)
                
                if prediction_proba is not None:
                    st.write(f"Prédiction (probabilité de défaut) : {prediction_proba:.2f}")
                    if prediction_proba > 0.5:
                        st.write("Décision : Crédit accordé")
                    else:
                        st.write("Décision : Crédit refusé")
                    
            except ValueError:
                st.error("Veuillez entrer un identifiant client valide (nombre entier).")

# Exécuter l'application Streamlit
if __name__ == "__main__":
    main()
