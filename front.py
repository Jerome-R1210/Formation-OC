import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
from io import BytesIO
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier
import numpy as np

# URL de l'API Flask
API_URL = 'https://apiprojet8-87cd68b59ab6.herokuapp.com/predict'
# Define the URL to download the model and data from GitHub
data_url = 'https://raw.githubusercontent.com/Jerome-R1210/Formation-OC/refs/heads/backend/app_train_sample.csv'
model_url = 'https://raw.githubusercontent.com/Jerome-R1210/Formation-OC/backend/pipeline_p7.joblib'

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

# Fonction pour charger le modèle pré-entraîné à partir d'une URL
def load_model_from_url(url):
    response = requests.get(url)
    model = joblib.load(BytesIO(response.content))
    return model

# Charger le modèle
model = load_model_from_url(model_url)

# Fonction pour charger les données du client
@st.cache_data
def load_client_data():
    data = pd.read_csv(data_url)
    return data

# Fonction pour obtenir les données du client spécifique
def get_client_data(client_id, client_data):
    client_info = client_data[client_data['SK_ID_CURR'] == int(client_id)]
    if client_info.empty:
        return None
    return client_info

# Fonction pour afficher l'importance des caractéristiques locales avec SHAP
def display_local_feature_importance(client_data, model):
    # Extraire le modèle de l'étape du pipeline
    lightgbm_model = model.named_steps['model']
    
    explainer = shap.Explainer(lightgbm_model)
    shap_values = explainer(client_data)

    st.subheader("Importance des caractéristiques locales")

    # Générer le graphique dans une figure Matplotlib
    shap.initjs()
    plt.figure()
    shap.waterfall_plot(shap_values[0])
    plt.title("Importance des caractéristiques locales")

    # Utilisez st.pyplot pour afficher le graphique dans Streamlit
    st.pyplot(plt.gcf())

# Fonction pour afficher les histogrammes comparatifs pour les 5 caractéristiques les plus importantes
def display_top_5_histograms(client_id, client_data, model):
    # Extraire le modèle LightGBM du pipeline
    lightgbm_model = model.named_steps['model']
    
    # Récupérer l'importance des caractéristiques
    importance = lightgbm_model.feature_importances_
    feature_names = client_data.columns.tolist()
    
    # Obtenir les 5 caractéristiques les plus importantes
    top_5_indices = np.argsort(importance)[-5:]
    top_5_features = [feature_names[i] for i in top_5_indices]
    
    # Récupérer les informations du client spécifique
    client_info = get_client_data(client_id, client_data)

    if client_info is None:
        st.error("Client non trouvé.")
        return

    accepted_clients = client_data[client_data['TARGET'] == 1]

    # Comparer les valeurs du client avec la moyenne des clients acceptés
    st.subheader(f"Comparaison des 5 caractéristiques les plus importantes pour le client {client_id}")
    
    # Afficher un histogramme pour chacune des 5 caractéristiques les plus importantes
    for feature in top_5_features:
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
            y=[accepted_clients[feature].mean()],
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

# Fonction pour afficher la jauge du score du client
def display_score_gauge(prediction_proba):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_proba * 100,
        title={'text': "Score de crédit (%)"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 100], 'color': "green"}]
        }
    ))

    st.plotly_chart(fig)

# Fonction pour afficher l'importance globale des caractéristiques
def display_global_feature_importance(model, client_data):
    st.subheader("Importance des caractéristiques globales")

    # Extraire l'importance des features depuis le modèle LightGBM
    lightgbm_model = model.named_steps['model']
    importance = lightgbm_model.feature_importances_

    feature_importance = pd.DataFrame({
        'features': client_data.columns,
        'importance': importance
    }).sort_values(by='importance', ascending=False)

    # Limiter à 10 caractéristiques les plus importantes
    top_10_features = feature_importance.head(10)

    # Création du graphique avec des barres horizontales
    fig = go.Figure([go.Bar(x=top_10_features['importance'], y=top_10_features['features'], orientation='h')])
    fig.update_layout(title="Top 10 des caractéristiques les plus importantes", 
                      xaxis_title="Importance", 
                      yaxis_title="Caractéristiques")
    st.plotly_chart(fig)

# Fonction pour afficher l'analyse bi-variée entre deux features
def display_bivariate_analysis(client_data, feature1, feature2, client_id):
    st.subheader(f"Analyse bi-variée entre {feature1} et {feature2} pour le client {client_id}")

    # Créer un scatter plot avec les deux features et un dégradé basé sur le score (TARGET)
    fig = go.Figure()

    # Ajouter tous les points
    fig.add_trace(go.Scatter(
        x=client_data[feature1],
        y=client_data[feature2],
        mode='markers',
        marker=dict(
            size=10,
            color=client_data['TARGET'],  # Couleur basée sur la variable cible
            colorscale='Viridis',
            showscale=True
        ),
        name='Tous les clients'
    ))

    # Ajouter le point pour le client spécifique
    client_info = get_client_data(client_id, client_data)
    fig.add_trace(go.Scatter(
        x=client_info[feature1],
        y=client_info[feature2],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='x',
        ),
        name='Client spécifique'
    ))

    # Mise en page du graphique
    fig.update_layout(
        title=f"Analyse bi-variée : {feature1} vs {feature2}",
        xaxis_title=feature1,
        yaxis_title=feature2
    )

    st.plotly_chart(fig)

# Fonction principale
def main():
    st.title("Dashboard de scoring client")

    # Saisie de l'identifiant du client
    client_id = st.text_input("Entrez l'identifiant du client:")

    # Sélection des features pour analyse
    feature_list = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
                    'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'DAYS_ID_PUBLISH', 
                    'DAYS_REGISTRATION', 'AMT_ANNUITY', 'REGION_POPULATION_RELATIVE']

    feature1 = st.selectbox("Sélectionnez la première feature pour l'analyse", feature_list)
    feature2 = st.selectbox("Sélectionnez la deuxième feature pour l'analyse", feature_list)

    # Bouton pour afficher les scores et les comparaisons
    if st.button("Afficher les informations"):
        if client_id:
            try:
                client_id = int(client_id)  # Convertir l'identifiant en entier

                # Charger les données clients
                client_data = load_client_data()

                # Appel à l'API Flask pour obtenir la probabilité de crédit
                prediction_proba = predict_loan_approval(client_id)

                if prediction_proba is not None:
                    st.write(f"Probabilité de crédit accepté : {prediction_proba:.2f}")

                    # Afficher la jauge du score
                    display_score_gauge(prediction_proba)

                    # Décision : accordé ou refusé
                    if prediction_proba > 0.5:
                        st.success("Crédit accordé")
                    else:
                        st.error("Crédit refusé")

                    # Récupérer les informations du client
                    client_info = get_client_data(client_id, client_data)

                    # Afficher l'importance des caractéristiques locales
                    display_local_feature_importance(client_info.drop(columns=['TARGET']), model)

                    # Afficher les histogrammes des 5 caractéristiques les plus importantes
                    display_top_5_histograms(client_id, client_data, model)
                   
                    # Afficher l'analyse bi-variée
                    display_bivariate_analysis(client_data, feature1, feature2, client_id)

                     # Afficher l'importance globale des caractéristiques
                    display_global_feature_importance(model, client_info.drop(columns=['TARGET']))
                else:
                    st.error("Impossible de récupérer la probabilité pour ce client.")
            except ValueError:
                st.error("Veuillez entrer un identifiant client valide (nombre entier).")

# Lancer l'application Streamlit
if __name__ == "__main__":
    main()
