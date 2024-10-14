import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

# URL de l'API Flask
API_URL = 'https://apiprojet8-87cd68b59ab6.herokuapp.com/predict'
#API_URL = 'http://127.0.0.1:8080/predict'

# URL des données clients
data_url = 'https://raw.githubusercontent.com/Jerome-R1210/Formation-OC/master/app_train_sample.csv?token=GHSAT0AAAAAACVYF4Q6YS547JYJ45SH7TWKZV3ONYA'

# Fonction pour appeler l'API Flask et obtenir la probabilité de crédit
@st.cache_data
def predict_loan_approval(client_id):
    try:
        response = requests.post(API_URL, json={'client_id': client_id})
        if response.status_code == 200:
            result = response.json()
            return result.get('prediction')
        else:
            st.error(f"Erreur lors de la prédiction : {response.json()['error']}")
            return None
    except Exception as e:
        st.error(f"Erreur lors de la requête à l'API : {str(e)}")
        return None

# Fonction pour charger les données du client et des clients ayant un crédit accepté
@st.cache_data
def load_client_data():
    data = pd.read_csv(data_url)
    accepted_clients = data[data['TARGET'] == 1]
    return data, accepted_clients

# Fonction pour obtenir les données du client
def get_client_data(client_id, client_data):
    client_info = client_data[client_data['SK_ID_CURR'] == int(client_id)]
    if client_info.empty:
        return None
    return client_info

# Fonction pour afficher les 10 principales variables et les histogrammes comparatifs
def display_histograms(client_id):
    # Charger les données clients et les clients ayant un crédit accepté
    client_data, accepted_clients = load_client_data()

    # Récupérer les informations du client spécifique
    client_info = get_client_data(client_id, client_data)
    
    if client_info is None:
        st.error("Client non trouvé.")
        return

    # Extraire les 10 variables les plus importantes selon le modèle LightGBM
    top_10_features = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
                       'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'DAYS_ID_PUBLISH', 
                       'DAYS_REGISTRATION', 'AMT_ANNUITY', 'REGION_POPULATION_RELATIVE']

    # Calculer la moyenne des clients acceptés pour ces variables
    accepted_means = accepted_clients[top_10_features].mean()

    # Comparer les valeurs du client avec la moyenne des clients acceptés
    st.subheader(f"Comparaison des 10 variables les plus importantes pour le client {client_id}")
    
    # Afficher un histogramme par variable
    for feature in top_10_features:
        fig = go.Figure()

        # Valeur du client
        fig.add_trace(go.Bar(
            x=[feature],
            y=[client_info[feature].values[0]],
            name='Client spécifique',
            marker_color='blue'
        ))

        # Moyenne des clients acceptés
        fig.add_trace(go.Bar(
            x=[feature],
            y=[accepted_means[feature]],
            name='Moyenne des clients acceptés',
            marker_color='green'
        ))

        # Configuration du layout de l'histogramme
        fig.update_layout(
            title=f'Comparaison pour la variable {feature}',
            yaxis_title='Valeur',
            barmode='group',
            xaxis_tickangle=-45
        )

        # Affichage du graphique dans Streamlit
        st.plotly_chart(fig)

# Interface utilisateur principale Streamlit
def main():
    st.title("Dashboard de scoring client")

    # Saisie de l'identifiant du client
    client_id = st.text_input("Entrez l'identifiant du client:")

    # Bouton pour afficher les scores et les comparaisons
    if st.button("Afficher les informations"):
        if client_id:
            try:
                client_id = int(client_id)  # Convertir l'identifiant en entier

                # Appel à l'API Flask pour obtenir la probabilité de crédit
                prediction_proba = predict_loan_approval(client_id)

                if prediction_proba is not None:
                    st.write(f"Probabilité de crédit accepté : {prediction_proba:.2f}")

                    # Décision : accordé ou refusé
                    if prediction_proba > 0.5:
                        st.success("Crédit accordé")
                    else:
                        st.error("Crédit refusé")

                    # Afficher les histogrammes comparatifs
                    display_histograms(client_id)
                else:
                    st.error("Impossible de récupérer la probabilité pour ce client.")
            except ValueError:
                st.error("Veuillez entrer un identifiant client valide (nombre entier).")

# Lancer l'application Streamlit
if __name__ == "__main__":
    main()
