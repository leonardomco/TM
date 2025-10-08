import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Leonardo_V1_3_1 import data
st.title("Telemetrie Formule 1")

# Sidebar for input
st.sidebar.header("Sélection de la session")
annee = st.sidebar.selectbox("Année", range(2018, 2026))
circuit = st.sidebar.selectbox("Circuit", ['Monza', 'Silverstone', 'Spa'])
session = st.sidebar.selectbox("Session", ['Qualifying', 'Race', 'Practice 1'])
pilote = st.sidebar.text_input("Abréviation du pilote (ex: VER, LEC)", 'VER')

# Load Data
if st.sidebar.button("Charger les données"):
    with st.spinner("Chargement des données..."):
        try:
            # ✅ Call your function properly
            donnees = data(annee, circuit, session, pilote, 3, 3, 9)
            st.success("✅ Données chargées !")

            # --- TRAJECTOIRE ---
            st.subheader('Trajectoire')
            fig, ax = plt.subplots()
            ax.plot(donnees['X'].values, donnees['Y'].values, linewidth=1)
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_xlabel('X'); ax.set_ylabel('Y')
            ax.grid(True)
            st.pyplot(fig)

            # --- ALTITUDE ---
            st.subheader('Altitude')
            st.line_chart(donnees.set_index('Distance')['Z'])

            # --- VITESSE SCALAIRE ---
            st.subheader('Vitesse scalaire')
            st.line_chart(donnees.set_index('Distance')['Speed'])

            # --- ACCÉLÉRATIONS ---
            st.subheader('Accélération longitudinale')
            st.line_chart(donnees.set_index('Distance')['Accélération longitudinale'])

            st.subheader('Accélération latérale')
            st.line_chart(donnees.set_index('Distance')['Accélération latérale'])

            st.subheader('Accélération verticale')
            st.line_chart(donnees.set_index('Distance')['Accélération verticale'])

            # --- PORTANCE ---
            st.subheader("Portance")
            fig, ax = plt.subplots()
            ax.plot(donnees['Distance'], donnees['Portance_moy'], label='Portance moyenne', color='blue')
            ax.fill_between(donnees['Distance'], donnees['Portance_min'], donnees['Portance_max'],
                            color='blue', alpha=0.2, label='Min–Max')
            ax.set_xlabel('Distance [m]')
            ax.set_ylabel('Portance [N]')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # --- TRAINÉE ---
            st.subheader("Trainée")
            fig, ax = plt.subplots()
            ax.plot(donnees['Distance'], donnees['Trainée_moy'], label='Trainée moyenne', color='red')
            ax.fill_between(donnees['Distance'], donnees['Trainée_min'], donnees['Trainée_max'],
                            color='red', alpha=0.2, label='Min–Max')
            ax.set_xlabel('Distance [m]')
            ax.set_ylabel('Trainée [N]')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # --- FROTTEMENT DE ROULEMENT ---
            st.subheader("Force de frottement de roulement")
            fig, ax = plt.subplots()
            ax.plot(donnees['Distance'], donnees['Force de frottement de roulement moy'],
                    label='Force moyenne', color='green')
            ax.fill_between(donnees['Distance'],
                            donnees['Force de frottement de roulement min'],
                            donnees['Force de frottement de roulement max'],
                            color='green', alpha=0.2, label='Min–Max')
            ax.set_xlabel('Distance [m]')
            ax.set_ylabel('Force de frottement [N]')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # --- FORCE MOTRICE ---
            st.subheader("Force motrice")
            fig, ax = plt.subplots()
            ax.plot(donnees['Distance'], donnees['Force motrice moy '],
                    label='Force motrice moyenne', color='orange')
            ax.fill_between(donnees['Distance'],
                            donnees['Force motrice min '],
                            donnees['Force motrice max '],
                            color='orange', alpha=0.2, label='Min–Max')
            ax.set_xlabel('Distance [m]')
            ax.set_ylabel('Force motrice [N]')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            

        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")



