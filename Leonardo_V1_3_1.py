#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.environ['FASTF1_NO_ERGAST'] = '1'
import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import math as m
import numpy as np



# In[3]:


a_gen = 2018
c_gen = 'Monza'
s_gen = 'Q'
p_gen = 'Lec'


# ### Data avec 

# In[21]:


def data (annee,circuit,session,pilote, Nax: int, Nay: int, Naz:int) -> pd.DataFrame:


    epreuve = fastf1.get_session(annee, circuit, session)
    epreuve.load(telemetry=True, laps=True, weather=True)
    circuit_info:pd.DataFrame = epreuve.get_circuit_info()
    tour:pd.DataFrame = epreuve.laps.pick_drivers(pilote).pick_fastest().dropna()

    df_corners = circuit_info.corners
    df_corners = df_corners.loc[:, ["Number", "Distance"]]

    
    #pos = pd.concat([pos, pos.iloc[[0]]], ignore_index=True)          # Permet que qd je plot y(x) le circuit soit fermé mais à 
    #                                                                    revoir pcq ça me nique mon plot des nouvelles accels
    
    telemetry: pd.DataFrame = tour.get_telemetry().copy()

    vx = telemetry["Speed"] / 3.6
    time_float = telemetry["Time"] / np.timedelta64(1, "s")
    dtime = np.gradient(time_float)
    ax = np.gradient(vx) / dtime

    # Clean up outliers
    for i in np.arange(1, len(ax) - 1).astype(int):
        if ax[i] > 25:
            ax[i] = ax[i - 1]

    # Smooth x-acceleration
    a_long_smooth = np.convolve(ax, np.ones((Nax,)) / Nax, mode="same")

    # Get position data
    x = telemetry["X"]
    y = telemetry["Y"]
    z = telemetry["Z"]

    # Calculate gradients
    dx = np.gradient(x)
    dy = np.gradient(y)
    dz = np.gradient(z)

    # Calculate theta (angle in xy-plane)
    theta = np.arctan2(dy, (dx + np.finfo(float).eps))
    theta[0] = theta[1]
    theta_noDiscont = np.unwrap(theta)

    # Calculate distance and curvature
    dist = telemetry["Distance"]
    ds = np.gradient(dist)
    dtheta = np.gradient(theta_noDiscont)

    # Clean up outliers
    for i in np.arange(1, len(dtheta) - 1).astype(int):
        if abs(dtheta[i]) > 0.5:
            dtheta[i] = dtheta[i - 1]

    # Calculate curvature and lateral acceleration
    C = dtheta / (ds + 0.0001)  # To avoid division by 0
    ay = np.square(vx) * C

    # Remove extreme values
    indexProblems = np.abs(ay) > 70
    ay[indexProblems] = 0
    
    # Smooth y-acceleration
    a_lat_smooth = np.convolve(ay, np.ones((Nay,)) / Nay, mode="same")

    # Calculate z-acceleration (similar process)
    z_theta = np.arctan2(dz, (dx + np.finfo(float).eps))
    z_theta[0] = z_theta[1]
    z_theta_noDiscont = np.unwrap(z_theta)

    z_dtheta = np.gradient(z_theta_noDiscont)

    # Clean up outliers
    for i in np.arange(1, len(z_dtheta) - 1).astype(int):
        if abs(z_dtheta[i]) > 0.5:
            z_dtheta[i] = z_dtheta[i - 1]

    # Calculate z-curvature and vertical acceleration
    z_C = z_dtheta / (ds + 0.0001)
    az = np.square(vx) * z_C

    # Remove extreme values
    indexProblems = np.abs(az) > 150
    az[indexProblems] = 0

    # Smooth z-acceleration
    az_smooth = np.convolve(az, np.ones((Naz,)) / Naz, mode="same")

    telemetry['vx'] = vx
    telemetry["Accélération longitudinale"] = a_long_smooth
    telemetry["Accélération latérale"] = a_lat_smooth
    telemetry["Accélération verticale"] = az_smooth



    if annee < 2018 or annee > 2025:
        print ('Erreur: année non accéptée')
        return


    compound = tour['Compound']

    meteo = tour.get_weather_data()
    humidité_relative = meteo["Humidity"]/100   #pour avoir la valeur comprise entre 0 et 1 
    presssion = meteo["Pressure"] * 100     #pour avoir la pression en Pa
    temp_C = meteo["AirTemp"]
    Rs =  287.058   # constante spécifique de l'air sec

    exp_term = np.exp((17.5043 * temp_C) / (241.2 + temp_C))
    rho = (1 / (Rs * (temp_C + 273.15))) * (presssion - 230.617 * humidité_relative * exp_term)
    g = 9.81


    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    m = pd.DataFrame({
        'm_moy': [785.5, 798.0, 801.0, 807.0, 853.0, 853.0, 853.0, 855.0],
        'delta_m': [52.5, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0]
    }, index=years)

    m['m_max'] = m['m_moy'] + m['delta_m']
    m['m_min'] = m['m_moy'] - m['delta_m']


    A = 1.7

    c = pd.DataFrame({
        'cd_moy': [0.896],
        'cl_moy': [-1.523]
    })

    c['cd_max'] = c['cd_moy'] + 0.1 * c['cd_moy']
    c['cd_min'] = c['cd_moy'] - 0.1 * c['cd_moy']

    c['cl_max'] = c['cl_moy'] + 0.1 * c['cl_moy']
    c['cl_min'] = c['cl_moy'] - 0.1 * c['cl_moy']



    m_moy = m.loc[annee, 'm_moy']
    m_min = m.loc[annee, 'm_min']
    m_max = m.loc[annee, 'm_max']

    # pick aero coefficients
    cd_moy = c.loc[0, 'cd_moy']
    cd_min = c.loc[0, 'cd_min']
    cd_max = c.loc[0, 'cd_max']
    cl_moy = c.loc[0, 'cl_moy']
    cl_min = c.loc[0, 'cl_min']
    cl_max = c.loc[0, 'cl_max']


    telemetry['Portance_min'] = 0.5 * rho * (telemetry['Speed']/3.6)**2 * cl_min * A
    telemetry['Portance_moy'] = 0.5 * rho * (telemetry['Speed']/3.6)**2 * cl_moy * A
    telemetry['Portance_max'] = 0.5 * rho * (telemetry['Speed']/3.6)**2 * cl_max * A

    telemetry['Trainée_min'] = 0.5 * rho * (telemetry['Speed']/3.6)**2 * cd_min * A
    telemetry['Trainée_moy'] = 0.5 * rho * (telemetry['Speed']/3.6)**2 * cd_moy * A
    telemetry['Trainée_max'] = 0.5 * rho * (telemetry['Speed']/3.6)**2 * cd_max * A

    telemetry['Force pesanteur_min'] = m_min * g
    telemetry['Force pesanteur_moy'] = m_moy * g
    telemetry['Force pesanteur_max'] = m_max * g


    P_min = abs(telemetry['Portance_min'])
    P_moy = abs(telemetry['Portance_moy'])
    P_max = abs(telemetry['Portance_max'])

    """def coef_f_r(v):
        f0 =  1 * 10**-2
        f1 = 5 * 10**-7
        f2 = 2 * 10**-7
        print(max(f0 + f1 * v + f2 * v**2))
        return f0 + f1 * v + f2 * v**2"""

    Cfr_moy = 0.015
    Cfr_min = 0.01
    Cfr_max = 0.02

    telemetry['Force de frottement de roulement min'] = (
        4 * (Cfr_min) * ((P_min + telemetry['Force pesanteur_min']) / 4)
    )

    telemetry['Force de frottement de roulement moy'] = (
        4 * (Cfr_moy) * ((P_moy + telemetry['Force pesanteur_moy']) / 4)
    )

    telemetry['Force de frottement de roulement max'] = (
        4 * (Cfr_max) * ((P_max + telemetry['Force pesanteur_max']) / 4)
    )






    telemetry['Force motrice max '] = (
        (m_max * telemetry['Accélération longitudinale']) + 
        (telemetry['Force de frottement de roulement min'] + telemetry['Trainée_min']))
    
    telemetry['Force motrice moy '] = (
        (m_moy * telemetry['Accélération longitudinale']) + 
        (telemetry['Force de frottement de roulement moy'] + telemetry['Trainée_moy']))
    

    telemetry['Force motrice min '] = (
        (m_min * telemetry['Accélération longitudinale']) + 
        (telemetry['Force de frottement de roulement max'] + telemetry['Trainée_max']))
    
    





    return telemetry

data(a_gen, c_gen, s_gen,p_gen, 3, 9, 9)


# In[46]:


def dynamics (annee, circuit, session, pilote):
    if annee < 2018 or annee > 2025:
        print ('Erreur: année non accéptée')
        return
    
    epreuve = fastf1.get_session(annee, circuit, session)
    epreuve.load(telemetry=True, laps=True, weather=True)
    circuit_info:pd.DataFrame = epreuve.get_circuit_info()
    tour:pd.DataFrame = epreuve.laps.pick_drivers(pilote).pick_fastest().dropna()

    compound = tour['Compound']
    
    df_corners = circuit_info.corners
    df_corners = df_corners.loc[:, ["Number", "Distance"]]
    
    telemetry: pd.DataFrame = tour.get_telemetry().copy()
    meteo = tour.get_weather_data()
    humidité_relative = meteo["Humidity"]/100   #pour avoir la valeur comprise entre 0 et 1 
    presssion = meteo["Pressure"] * 100     #pour avoir la pression en Pa
    temp_C = meteo["AirTemp"]
    Rs =  287.058   # constante spécifique de l'air sec

    exp_term = np.exp((17.5043 * temp_C) / (241.2 + temp_C))
    rho = (1 / (Rs * (temp_C + 273.15))) * (presssion - 230.617 * humidité_relative * exp_term)
    g = 9.81

    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    m = pd.DataFrame({
        'm_moy': [785.5, 798.0, 801.0, 807.0, 853.0, 853.0, 853.0, 855.0],
        'delta_m': [52.5, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0]
    }, index=years)

    m['m_max'] = m['m_moy'] + m['delta_m']
    m['m_min'] = m['m_moy'] - m['delta_m']


    A = 1.7

    c = pd.DataFrame({
        'cd_moy': [0.896],
        'cl_moy': [-1.523]
    })

    c['cd_max'] = c['cd_moy'] + 0.1 * c['cd_moy']
    c['cd_min'] = c['cd_moy'] - 0.1 * c['cd_moy']

    c['cl_max'] = c['cl_moy'] + 0.1 * c['cl_moy']
    c['cl_min'] = c['cl_moy'] - 0.1 * c['cl_moy']

    print(c)
    p = 1.4 * 100000

    rayon = pd.DataFrame({
        'Dry':   [335, 335, 335, 335, 362.5, 362.5, 362.5, 362.5],
        'Wet':   [340, 340, 340, 340, 367.5, 367.5, 367.5, 367.5],
    }, index=years)

    slick = ['HYPERSOFT', 'ULTRASOFT', 'SUPERSOFT', 'SOFT', 'MEDIUM', 'HARD', 'SUPERHARD']
    wet = ['INTERMEDIATE', 'WET']

    if compound in slick:
        R = rayon.loc[annee, 'Dry']   
    elif compound in wet:
        R = rayon.loc[annee, 'Wet']   
    else:
        R = None     


    lar = pd.DataFrame({
        'lar_f_moy': [377.5, 377.5, 377.5, 377.5, 360, 360, 360, 360,],
        'lar_r_moy': [462.5, 462.5, 462.5, 462.5, 455, 455, 455, 455,],
        'delta_lar': [7.5, 7.5, 7.5, 7.5, 15, 15, 15, 15],
    }, index = years)


    lar['lar_f_max'] = lar['lar_f_moy'] + lar['delta_lar']
    lar['lar_f_min'] = lar['lar_f_moy'] - lar['delta_lar']

    lar['lar_r_max'] = lar['lar_r_moy'] + lar['delta_lar']
    lar['lar_r_min'] = lar['lar_r_moy'] - lar['delta_lar']


    """
    telemetry ['Portance'] = 1/2 * rho * (telemetry['Speed']/3.6) * CL * A
    telemetry ['Trainée'] = 1/2 * rho * (telemetry['Speed']/3.6) * CD * A
    telemetry ['Force pesanteur'] = m * g
    telemetry ['Force de frottement de roulement'] = e/R * (telemtry['Portance'] + telemetry["Force pesanteur"])
    telemetry['Force motrice'] = (m * telemetry['Accélération longitudinale']) - (telemetry['Force de frottoment de roulement'] + telemtery['Trainée'])
    
    """



    print(meteo)

dynamics(a_gen, c_gen, s_gen, p_gen, )


# ### Plots

# In[ ]:

"""
#plt.plot(pos['Distance'], pos['||a_lat_sin||'], linestyle='-', label='||a_lat_sin||', color='blue')
plt.figure(figsize=(10,6))
#plt.plot(telemetry["Distance"], telemetry["Accélération longitudinale"], color='green')

# 👉 ADDED: vertical lines at corners
for _, row in df_corners.iterrows():
    plt.axvline(x=row["Distance"], color="black", linestyle="--", alpha=0.6)
    plt.text(row["Distance"], plt.ylim()[1], str(row["Number"]), 
            rotation=90, va="bottom", ha="center", fontsize=7)
    
plt.title('Accélération longitudinale en fonction de la distance')
plt.xlabel("Distance (m)")
plt.ylabel("Accélération longitudinale (m/s²)")
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(telemetry["Distance"], telemetry["Accélération latérale"], color='red')

# 👉 ADDED: vertical lines at corners
for _, row in df_corners.iterrows():
    plt.axvline(x=row["Distance"], color="black", linestyle="--", alpha=0.6)
    plt.text(row["Distance"], plt.ylim()[1], str(row["Number"]), 
            rotation=90, va="bottom", ha="center", fontsize=7)
    
plt.title('Accélération latérale en fonction de la distance')
plt.xlabel("Distance (m)")
plt.ylabel("Accélération latérale (m/s²)")
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(telemetry["Distance"], telemetry["Accélération verticale"], color='red')

# 👉 ADDED: vertical lines at corners
for _, row in df_corners.iterrows():
    plt.axvline(x=row["Distance"], color="black", linestyle="--", alpha=0.6)
    plt.text(row["Distance"], plt.ylim()[1], str(row["Number"]), 
            rotation=90, va="bottom", ha="center", fontsize=7)
    
plt.title('Accélération verticale en fonction de la distance')
plt.xlabel("Distance (m)")
plt.ylabel("Accélération verticale (m/s²)")
plt.grid(True)
plt.show()


"""




