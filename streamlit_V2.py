import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from Leonardo_V2 import donnees

st.set_page_config(page_title="Télémetrie Formule 1", layout="wide")
st.title("Télémetrie Formule 1")

# Sidebar controls
st.sidebar.header("Sélection de la session")
annee = st.sidebar.selectbox("Année", range(2018, 2026))
circuit = st.sidebar.selectbox("Circuit", ['Monza', 'Silverstone', 'Spa-Francorchamps'])
session = st.sidebar.selectbox("Session", ['Qualifying', 'Race', 'Practice 1'])
pilote = st.sidebar.text_input("Abréviation du pilote (ex: VER, LEC)", "")
show_corners = True


# Cache data loading
@st.cache_data
def load_data(a, c, s, p):
    return donnees(a, c, s, p,)

# Load Data
if st.sidebar.button("Charger les données"):
    with st.spinner("Chargement des données..."):
        try:
            donnees, track, circuit_info, df_corners = load_data(annee, circuit, session, pilote)

            X = track[:, 0]
            Y = track[:, 1]

            st.success("✅ Données chargées avec succès !")

            st.subheader('Trajectoire')
            fig_traj = go.Figure()

            fig_traj.add_trace(go.Scatter(
                x=X,
                y=Y,
                mode='lines',
                line=dict(color='blue', width=2),
            ))

            if show_corners and hasattr(circuit_info, "corners"):
                offset_vector = np.array([500, 0])  

                def rotate(xy, *, angle):
                    rot_mat = np.array([
                        [np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]
                    ])
                    return np.matmul(xy, rot_mat)

                track_angle = circuit_info.rotation / 180 * np.pi

                for _, corner in circuit_info.corners.iterrows():
                    txt = f"{corner['Number']}{corner['Letter']}"
                    offset_angle = corner['Angle'] / 180 * np.pi

                    offset_x, offset_y = rotate(offset_vector, angle=offset_angle)
                    text_x = corner['X'] + offset_x
                    text_y = corner['Y'] + offset_y

                    text_x, text_y = rotate(np.array([text_x, text_y]), angle=track_angle)
                    track_x, track_y = rotate(np.array([corner['X'], corner['Y']]), angle=track_angle)

                    fig_traj.add_trace(go.Scatter(
                        x=[track_x, text_x],
                        y=[track_y, text_y],
                        mode="lines",
                        line=dict(color="gray", width=1),
                        showlegend=False
                    ))

                    fig_traj.add_trace(go.Scatter(
                        x=[text_x],
                        y=[text_y],
                        mode="markers+text",
                        marker=dict(size=14, color="gray"),
                        text=[txt],
                        textposition="middle center",
                        textfont=dict(color="white", size=10),
                        showlegend=False
                    ))

            fig_traj.update_layout(
                xaxis_title="Coordonnées X",
                yaxis_title="Coordonnées Y",
                width=800,
                height=600
            )

            fig_traj.update_yaxes(scaleanchor="x", scaleratio=1)

            st.plotly_chart(fig_traj, use_container_width=True)

            st.subheader("Altitude")

            fig_alt = go.Figure()
            fig_alt.add_trace(go.Scatter(
                x=donnees["Distance"],
                y=donnees["Altitude"],
                mode='lines',
                line=dict(color='blue', width=2),
                name='Altitude'
            ))

            for _, row in df_corners.iterrows():
                fig_alt.add_vline(
                    x=row["Distance"],
                    line=dict(color="white", dash="dash", width=1),
                    opacity=1
                )

                fig_alt.add_annotation(
                    x=row["Distance"],
                    y=min(donnees["Altitude"]) - 20,   
                    text=str(int(row["Number"])),
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",                    
                    textangle=0,                      
                    font=dict(size=16, color="white"),
                    align="center"
                )

            fig_alt.update_layout(
                xaxis_title="Distance (m)",
                yaxis_title="Altitude (m)",
                width=800,
                height=600,
                hovermode='x unified', 
                plot_bgcolor='rgba(0,0,0,0)') 

            fig_alt.update_traces(
                hovertemplate="Distance: %{x:.1f} m<br>Altitude: %{y:.1f} m<extra></extra>"
            )

            st.plotly_chart(fig_alt, use_container_width=True)

            st.subheader("Vitesse")

            fig_vit = go.Figure()

            vx_ms = donnees["vx"]
            fig_vit.add_trace(go.Scatter(
                x=donnees["Distance"],
                y=vx_ms,
                mode="lines",
                line=dict(color="blue", width=2),
                yaxis="y1"
            ))

            for _, row in df_corners.iterrows():
                fig_vit.add_vline(
                    x=row["Distance"],
                    line=dict(color="white", dash="dash", width=1),
                    opacity=1
                )
                fig_vit.add_annotation(
                    x=row["Distance"],
                    y=min(vx_ms) - 10,
                    text=str(int(row["Number"])),
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    font=dict(size=14, color="white"),
                    align="center"
                )

            fig_vit.update_layout(
                xaxis=dict(title="Distance (m)"),
                yaxis=dict(
                    title=dict(text="Vitesse (m/s)", font=dict(color="white")),
                    tickfont=dict(color="white"),
                    title_standoff=30,  
                ),
                width=800,
                height=600,
                hovermode='x unified', 
                plot_bgcolor="rgba(0,0,0,0)",

            )
            fig_vit.update_traces(
                hovertemplate="Distance: %{x:.1f} m<br>Vitesse : %{y:.1f} m/s<extra></extra>"
            )


            st.plotly_chart(fig_vit, use_container_width=True)

            st.subheader("Accélération tangentielle")
            fig_a_t = go.Figure()

            fig_a_t.add_trace(go.Scatter(
                x=donnees["Distance"],
                y=donnees["Accélération tangentielle"],
                mode='lines',
                line=dict(color='blue', width=2),
                name='A_t'
            ))


            for _, row in df_corners.iterrows():
                fig_a_t.add_vline(
                    x=row["Distance"],
                    line=dict(color="white", dash="dash", width=1),
                    opacity=1
                )

                fig_a_t.add_annotation(
                    x=row["Distance"],
                    y=min(donnees["Accélération tangentielle"]) - 20,   
                    text=str(int(row["Number"])),
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",                    
                    textangle=0,                     
                    font=dict(size=16, color="white"),
                    align="center"
                )
            fig_a_t.update_layout(
                xaxis_title="Distance (m)",
                yaxis_title="Accélération tangentielle (m/s^2)",
                width=800,
                height=600,
                hovermode='x unified', 
                plot_bgcolor='rgba(0,0,0,0)',  
            )

            fig_a_t.update_traces(
                hovertemplate="Distance: %{x:.1f} m<br>Accélération tangentielle: %{y:.1f} m/s^2<extra></extra>"
            )

            st.plotly_chart(fig_a_t, use_container_width=True)

            st.subheader("Accélération normale")
            fig_a_n = go.Figure()

            fig_a_n.add_trace(go.Scatter(
                x=donnees["Distance"],
                y=donnees["Accélération normale"],
                mode='lines',
                line=dict(color='blue', width=2),
                name='A_n'
            ))


            for _, row in df_corners.iterrows():
                fig_a_n.add_vline(
                    x=row["Distance"],
                    line=dict(color="white", dash="dash", width=1),
                    opacity=1
                )

                fig_a_n.add_annotation(
                    x=row["Distance"],
                    y=min(donnees["Accélération normale"]) - 20,   
                    text=str(int(row["Number"])),
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",                    
                    textangle=0,                      
                    font=dict(size=16, color="white"),
                    align="center"
                )
                
            fig_a_n.update_layout(
                xaxis_title="Distance (m)",
                yaxis_title="Accélération normale (m/s^2)",
                width=800,
                height=600,
                hovermode='x unified', 
                plot_bgcolor='rgba(0,0,0,0)', 
            )

            fig_a_n.update_traces(
                hovertemplate="Distance: %{x:.1f} m<br>Accélération normale: %{y:.1f} m/s^2<extra></extra>"
            )


            st.plotly_chart(fig_a_n, use_container_width=True)

            st.subheader("Accélération verticale")
            fig_a_v = go.Figure()

            fig_a_v.add_trace(go.Scatter(
                x=donnees["Distance"],
                y=donnees["Accélération verticale"],
                mode='lines',
                line=dict(color='blue', width=2),
                name='A_v'
            ))


            for _, row in df_corners.iterrows():

                fig_a_v.add_vline(
                    x=row["Distance"],
                    line=dict(color="white", dash="dash", width=1),
                    opacity=1
                )

                fig_a_v.add_annotation(
                    x=row["Distance"],
                    y=min(donnees["Accélération verticale"]) - 20,   
                    text=str(int(row["Number"])),
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",                    
                    textangle=0,                      
                    font=dict(size=16, color="white"),
                    align="center"
                )
            # Layout settings
            fig_a_v.update_layout(
                xaxis_title="Distance (m)",
                yaxis_title="Accélération verticale (m/s^2)",
                width=800,
                height=600,
                hovermode='x unified', 
                plot_bgcolor='rgba(0,0,0,0)',  
            )

            fig_a_v.update_traces(
                hovertemplate="Distance: %{x:.1f} m<br>Accélération verticale: %{y:.1f} m/s^2<extra></extra>"
            )


            st.plotly_chart(fig_a_v, use_container_width=True)

            st.subheader("Portance")

            fig_portance = go.Figure()

            fig_portance.add_trace(go.Scatter(
                x=donnees["Distance"],
                y=donnees["Portance_moy"],
                mode='lines',
                line=dict(color='blue', width=2),
                name='Portance moyenne'
            ))

            fig_portance.add_trace(go.Scatter(
                x=pd.concat([donnees["Distance"], donnees["Distance"][::-1]]),
                y=pd.concat([donnees["Portance_max"], donnees["Portance_min"][::-1]]),
                fill='toself',
                fillcolor='rgba(0, 0, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='min–max',
                showlegend=True,
                hoverinfo='text',               
                hovertext='min–max',             
                hovertemplate='%{hovertext}<extra></extra>'  
            ))


            for _, row in df_corners.iterrows():
                fig_portance.add_vline(
                    x=row["Distance"],
                    line=dict(color="white", dash="dash", width=1),
                    opacity=1
                )

                fig_portance.add_annotation(
                    x=row["Distance"],
                    y=min(donnees["Portance_min"]) - 20,
                    text=str(int(row["Number"])),
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    textangle=0,
                    font=dict(size=16, color="white"),
                    align="center"
                )

            fig_portance.update_layout(
                xaxis_title="Distance (m)",
                yaxis_title="Portance (N)",
                width=800,
                height=600,
                hovermode='x unified',
                hoverlabel=dict(
                    bgcolor='rgba(30,30,30,0.8)',
                    font_size=13,
                    font_color='white'
                ),
                xaxis=dict(showspikes=False),  
                showlegend = False
            )
            

            fig_portance.update_traces(
                hovertemplate="Distance: %{x:.1f} m<br>Portance: %{y:.1f} N<extra></extra>"
            )


            st.plotly_chart(fig_portance, use_container_width=True)

            st.subheader("Trainée")
            fig_trainee = go.Figure()


            fig_trainee.add_trace(go.Scatter(
                x=donnees["Distance"],
                y=donnees["Trainée_moy"],
                mode='lines',
                line=dict(color='blue', width=2),
                name='Portance moyenne'
            ))

            fig_trainee.add_trace(go.Scatter(
                x=pd.concat([donnees["Distance"], donnees["Distance"][::-1]]),
                y=pd.concat([donnees["Trainée_max"], donnees["Trainée_min"][::-1]]),
                fill='toself',
                fillcolor='rgba(0, 0, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='min–max',
                showlegend=True,
                hoverinfo='text',               
                hovertext='min–max',             
                hovertemplate='%{hovertext}<extra></extra>'  
            ))

            for _, row in df_corners.iterrows():
                fig_trainee.add_vline(
                    x=row["Distance"],
                    line=dict(color="white", dash="dash", width=1),
                    opacity=1
                )

                fig_trainee.add_annotation(
                    x=row["Distance"],
                    y=min(donnees["Trainée_min"]) - 20,
                    text=str(int(row["Number"])),
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    textangle=0,
                    font=dict(size=16, color="white"),
                    align="center"
                )


            fig_trainee.update_layout(
                xaxis_title="Distance (m)",
                yaxis_title="Trainée (N)",
                width=800,
                height=600,
                hovermode='x unified',
                hoverlabel=dict(
                    bgcolor='rgba(30,30,30,0.8)',
                    font_size=13,
                    font_color='white'
                ),
                xaxis=dict(showspikes=False),  
                showlegend = False,
            )

            fig_trainee.update_traces(
                hovertemplate="Distance: %{x:.1f} m<br>Trainée: %{y:.1f} N<extra></extra>"
            )

            st.plotly_chart(fig_trainee, use_container_width=True)

            st.subheader("Force de frottement au roulement")
            fig_fr = go.Figure()

            fig_fr.add_trace(go.Scatter(
                x=donnees["Distance"],
                y=donnees["Force de frottement de roulement moy"],
                mode='lines',
                line=dict(color='blue', width=2),
                name='Portance moyenne'
            ))

            
            fig_fr.add_trace(go.Scatter(
                x=pd.concat([donnees["Distance"], donnees["Distance"][::-1]]),
                y=pd.concat([donnees["Force de frottement de roulement max"], donnees["Force de frottement de roulement min"][::-1]]),
                fill='toself',
                fillcolor='rgba(0, 0, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='min–max',
                showlegend=True,
                hoverinfo='text',               
                hovertext='min–max',             
                hovertemplate='%{hovertext}<extra></extra>'  
            ))

            for _, row in df_corners.iterrows():
                fig_fr.add_vline(
                    x=row["Distance"],
                    line=dict(color="white", dash="dash", width=1),
                    opacity=1
                )

                fig_fr.add_annotation(
                    x=row["Distance"],
                    y=min(donnees["Force de frottement de roulement min"]) - 20,
                    text=str(int(row["Number"])),
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    textangle=0,
                    font=dict(size=16, color="white"),
                    align="center"
                )

            fig_fr.update_layout(
                xaxis_title="Distance (m)",
                yaxis_title="Force de frottement de roulement (N)",
                width=800,
                height=600,
                hovermode='x unified',
                hoverlabel=dict(
                    bgcolor='rgba(30,30,30,0.8)',
                    font_size=13,
                    font_color='white'
                ),
                xaxis=dict(showspikes=False),  
                showlegend = False,
            )

            fig_fr.update_traces(
                hovertemplate="Distance: %{x:.1f} m<br>Force de frottement: %{y:.1f} N<extra></extra>"
            )

            st.plotly_chart(fig_fr, use_container_width=True)



            st.subheader("Force motrice")
            fig_m = go.Figure()


            fig_m.add_trace(go.Scatter(
                x=donnees["Distance"],
                y=donnees["Force motrice moy"],
                mode='lines',
                line=dict(color='blue', width=2),
                name='Portance moyenne'
            ))

           
            fig_m.add_trace(go.Scatter(
                x=pd.concat([donnees["Distance"], donnees["Distance"][::-1]]),
                y=pd.concat([donnees["Force motrice max"], donnees["Force motrice min"][::-1]]),
                fill='toself',
                fillcolor='rgba(0, 0, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='min–max',
                showlegend=True,
                hoverinfo='text',                
                hovertext='min–max',             
                hovertemplate='%{hovertext}<extra></extra>'  
            ))

            for _, row in df_corners.iterrows():
                fig_m.add_vline(
                    x=row["Distance"],
                    line=dict(color="white", dash="dash", width=1),
                    opacity=1
                )

                fig_m.add_annotation(
                    x=row["Distance"],
                    y=min(donnees["Force motrice min"]) - 20,
                    text=str(int(row["Number"])),
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    textangle=0,
                    font=dict(size=16, color="white"),
                    align="center"
                )

            fig_m.update_layout(
                xaxis_title="Distance (m)",
                yaxis_title="Force motrice (N)",
                width=800,
                height=600,
                hovermode='x unified',
                hoverlabel=dict(
                    bgcolor='rgba(30,30,30,0.8)',
                    font_size=13,
                    font_color='white'
                ),
                xaxis=dict(showspikes=False), 
                showlegend = False,
            )

            fig_m.update_traces(
                hovertemplate="Distance: %{x:.1f} m<br>Force motricet: %{y:.1f} N<extra></extra>"
            )

            st.plotly_chart(fig_m, use_container_width=True)


            st.subheader("Force de freinage")
            fig_f = go.Figure()

            fig_f.add_trace(go.Scatter(
                x=donnees["Distance"],
                y=donnees["Force de freinage moy"],
                mode='lines',
                line=dict(color='blue', width=2),
                name='Portance moyenne'
            ))

            fig_f.add_trace(go.Scatter(
                x=pd.concat([donnees["Distance"], donnees["Distance"][::-1]]),
                y=pd.concat([donnees["Force de freinage max"], donnees["Force de freinage min"][::-1]]),
                fill='toself',
                fillcolor='rgba(0, 0, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='min–max',
                showlegend=True,
                hoverinfo='text',               
                hovertext='min–max',             
                hovertemplate='%{hovertext}<extra></extra>'  
            ))

            for _, row in df_corners.iterrows():
                fig_f.add_vline(
                    x=row["Distance"],
                    line=dict(color="white", dash="dash", width=1),
                    opacity=1
                )

                fig_f.add_annotation(
                    x=row["Distance"],
                    y=min(donnees["Force de freinage min"]) - 20,
                    text=str(int(row["Number"])),
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    textangle=0,
                    font=dict(size=16, color="white"),
                    align="center"
                )

            fig_f.update_layout(
                xaxis_title="Distance (m)",
                yaxis_title="Force de freinage (N)",
                width=800,
                height=600,
                hovermode='x unified',
                hoverlabel=dict(
                    bgcolor='rgba(30,30,30,0.8)',
                    font_size=13,
                    font_color='white'
                ),
                xaxis=dict(showspikes=False), 
                showlegend = False,
            )

            fig_f.update_traces(
                hovertemplate="Distance: %{x:.1f} m<br>Force motricet: %{y:.1f} N<extra></extra>"
            )

            st.plotly_chart(fig_f, use_container_width=True)


        except Exception as e:
            st.error(f"❌ Erreur lors du chargement ou de l'affichage des données : {e}")



