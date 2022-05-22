
"""
Created on Sun Apr 17 13:59:40 2022

@author: Arroua
"""

import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm


from sklearn import model_selection, preprocessing
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import datetime
from datetime import datetime



with st.sidebar:
    menu = st.radio("SOMMAIRE",
                    ('Projet', 'Dataset', 'Analyse exploratoire',
                     'Etude des corrélations', 'Etude de la consommation',
                     'Prédiction de la consommation')
                    )

    st.markdown(''' *Auteur* : 

**Justine Rougé** 

**Rachid Boujida**

**Boussad ARROUA** 

*Mentor* : **Lara E.**

*Bootcamp Data Scientist Février 2022*.''') 


if menu == 'Projet':
    st.title("PY-EnerConsoProd")
    st.image("https://www.encyclopedie-energie.org/wp-content/uploads/2015/08/besoins-energie-1170x570.jpg")
    st.subheader("**Projet**")
    st.markdown('''Ce projet est réalisé dans le cadre de la formation **Data Scientist** de \
                 [DataScientest.com](https://datascientest.com/en/home-page).

L’objectif principal du projet PY-EnerConsoProd est de constater \
le phasage entre la consommation et la production énergétique au niveau national \
et au niveau régional.
                   
Afin d’atteindre cet objectif, nous allons : 
1)	Analyser de façon exploratoire des données de consommation et de production d’énergie.
2)	Etudier les corrélations entre la consommation énergétique et les températures extérieures.
3)	Etudier la consommation énergétique selon la saisonnalité au niveau national et régional.
4)	Prédire la consommation énergétique pour 2022 à l’échelle nationale et régionale.'''
                )


elif menu == 'Dataset':
    dataset = st.radio("JEUX DE DONNES",
                    ('Dataset principal', 'Dataset secondaire'))
    if dataset == 'Dataset principal':
        st.markdown('''##### 1. DATASET PRINCIPAL
Le jeu de données provient de l'Open Data Réseaux Energies: \
                    [ODRE](https://opendata.reseaux-energies.fr/explore/dataset/eco2mix-regional-cons-def/information/disjunctive.libelle_region&disjunctive.nature&sort=-date_heure)
                    .''')
        

        df_pr = pd.read_csv('Dataset_principal.csv')
        st.dataframe(df_pr)
        st.markdown('''Ce jeu de données, rafraîchi une fois par jour, présente \
                les données régionales consolidées et définitives de **janvier \
                    2013 à décembre 2020**.

On y trouvere au pas demi-heure :
    
    * La consommation réalisée.
    * La production selon les différentes filières composant le mix énergétique.
    * La consommation des pompes dans les Stations de Transfert d'Energie par Pompage (STEP)
    * Le solde des échanges avec les régions limitrophes.
    * le Taux de COuverture (TCO) d'une filière de production au sein d'une région.
    * le Taux de CHarge (TCH) ou facteur de charge (FC) d'une filière.

**La taille du Dataframe** : 1875455 lignes X 28 colonnes.''')

        st.markdown('''##### 2. TRAITEMENT DES DONNEES
Le Dataframe ne contient pas de doublons.

**Valeurs manquantes**

Entre 108 et 1640256 valeurs sont manquantes, soit entre 0 et 87,5% du DataFrame.
Elles concernent 21 variables. l' analyse statistique de la consommation et de la
et production énergétique a révélé que la somme de la consommation est égale à la somme
de la production par type d'énergie. Ce qui signifie qu'on peut remplacer les valeurs
manquantes par des 0.

**Ajout de variables** 

Pour alimenter notre analyse, 3 variables sont ajoutées.

*Périodicité* : Année, mois et jour.''')

    else : 
        st.markdown('''##### 2. DATASET SECONDAIRE

Le jeu de données provient de l'Open Data Réseaux Energies: \
                    [ODRE](https://opendata.reseaux-energies.fr/explore/dataset/eco2mix-regional-cons-def/information/disjunctive.libelle_region&disjunctive.nature&sort=-date_heure)
                    .''')
        df_sec = pd.read_csv('Dataset_secondaire.csv')
        st.dataframe(df_sec)
        st.markdown('''Ce jeu de données présente les températures minimales,
                maximales et moyennes quotidiennes (en degré celsius),
                par région française, **du 1er janvier 2016 au 31 janvier
                2022**.

Il est basé sur les mesures officielles du réseau de stations météorologiques
françaises. La mise à jour de ce jeu de données est mensuelle.''')
    
    

elif menu == 'Analyse exploratoire':
    genre = st.selectbox("ANALYSE DES DONNEES", ('Consommation', 'Production'))
    if genre == 'Consommation':
        st.markdown("##### **1. Etude de la consommation énergétique en France**")

        df_an = pd.read_csv('consommation_an.csv')
        # st.dataframe(data)
        figure = px.line(df_an, x='Année', y='Consommation (MW)',
                         width=800, height=500)
        figure.update_layout(xaxis_rangeslider_visible=True)
        figure.update_xaxes(title_font=dict(size=18, family='Arial',
                                            color='black'))
        figure.update_yaxes(title_font=dict(size=18, family='Arila',
                                            color='black'))
        st.plotly_chart(figure)
        st.markdown('''* La courbe montre une nette décroissance de la consommation \
                    énergétique à partir de 2013. 
 
* Une chute importante de la consommation est constatée pour 2014, qui peut \
    s’expliquer par les conditions climatiques.
        
                        
* La décroissance de la consommation énergetique de 2019 est liée à \
            la pandémie de Covid 19.''')

        st.markdown("##### **2. Etude de la consommation énergétique à \
                    l'echelle régional**")
        
        df_reg = pd.read_csv('consommation_reg.csv')
        
        fig = px.bar(df_reg, x='Region', y='Consommation (MW)',
                     color='Consommation (MW)',
                     text='Consommation (MW)',
                     width=800)
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(font_family="Arial",
                          font_color="Black", title_font_family="Arial",
                          title_font_color="Black",
                          title_font_size=20)
        fig.update_xaxes(title_font=dict(size=18, family='Arial',
                                         color='black'))
        fig.update_yaxes(title_font=dict(size=18, family='Arila',
                                         color='black'))
        st.plotly_chart(fig)
        st.markdown("* **Centre-Val de Loire**, **Bourgogne-Franche-Compté** \
                    et **la Bretagne** sont les régions qui ont consommées \
                        moins d’énérgie en France.")

        st.markdown("* **Ile de France** et **Auvergne-Rhône-Alpes** sont les \
                     régions les plus consommatrices d'énergie en France.")

    else:
        
        st.markdown("##### **1. Etude de la production énergétique en France**")
        df_prod = pd.read_csv('production_an.csv')
        dat = px.area(df_prod, x="Année", y="Production (MW)", color='Type', 
                      height=500, width=800)
        dat.update_layout(xaxis_rangeslider_visible=True)
        dat.update_yaxes(title_font=dict(size=18, family='Arila',
                                         color='black'))
        dat.update_layout(font_family="Arial",
                          font_color="Black", title_font_family="Arial",
                          title_font_color="Black",
                          title_font_size=20)
        st.plotly_chart(dat)
        st.markdown('''* L’énergie nucléaire reste la filière énergétique \
                    la plus produite en France. 
                        
* L’énergie solaire et les bioénergies sont les filières énergétiques\
            ayant été le moins produites pendant ces 8 années.''')
        
        
        st.markdown("##### **2. Etude de la production énergétique à \
                    l'échelle régional**")
                    
        df_reg = pd.read_csv('consommation_reg.csv')
        x = df_reg['Region']
        figs = go.Figure(layout=go.Layout(height=600, width=900))
        
        figs.add_trace(go.Bar(x=x, y=df_reg['Thermique (MW)'], 
                              name ='Thermique', marker_color='blue'))
        figs.add_trace(go.Bar(x=x, y=df_reg['Nucléaire (MW)'], 
                              name = 'Nucléaire', marker_color='magenta'))
        figs.add_trace(go.Bar(x=x, y=df_reg['Eolien (MW)'], 
                              name ='Eolien', marker_color='green'))
        figs.add_trace(go.Bar(x=x, y=df_reg['Solaire (MW)'],
                              name ='Solaire', marker_color='yellow'))
        figs.add_trace(go.Bar(x=x, y=df_reg['Hydraulique (MW)'],
                              name ='Hydraulique',marker_color='brown'))
        figs.add_trace(go.Bar(x=x, y=df_reg['Pompage (MW)'], 
                              name ='Pompage',marker_color='cyan'))
        figs.add_trace(go.Bar(x=x, y=df_reg['Ech. physiques (MW)'], 
                      name ='Ech. physiques', marker_color='orange'))
        figs.update_layout(barmode='relative', 
                           xaxis={'categoryorder':'total descending'})
        figs.update_yaxes(title_text='Production (MW)')
        figs.update_yaxes(title_font=dict(size=18, family='Arila',
                                         color='black'))
        figs.update_layout(font_family="Arial",
                          font_color="Black", title_font_family="Arial",
                          title_font_color="Black",
                          title_font_size=20)
        st.plotly_chart(figs)
        
        st.markdown(''' Le mix énergétique est plus diversifié dans les régions\
                    **d’Auvergne-Rhône-Alpes**, **Hauts-de-France**, \
                        **Grand-Est**, **Nouvelle-Aquitaine**, \
                            **Provence-Alpes-Côte d’Azur** et **Occitanie**.
        
l’**Auvergne-Rhône-Alpes**, **Grand-Est**, **Normandie** et \
    **Centre -Val de Loire** sont les régions les plus productrices d’énergie \
                (exportateurs d’énergie).''')


elif menu == 'Etude des corrélations':
    dg = pd.read_csv("ST_Conso_temp.csv")

    T = px.line(dg, x='TMoy (°C)', y='Consommation (MW)', 
                  color='Région', width=800, height=800)
    st.plotly_chart(T)
    
    st.markdown("#### Corrélation entre la température et la consommation d'énergie en France")
    dg_F = dg.groupby([ 'Date']).agg({'Consommation (MW)': 'sum','TMoy (°C)' : 'mean'})
    corr = dg_F.corr()
    sns.heatmap(corr,annot = True , cmap = "viridis", center = 0)
    st.pyplot()
    
    if st.checkbox("Afficher jeu de données"):
        st.dataframe(dg)

    dg_IDF = dg[dg.Région == 'Île-de-France']
    dg_CVL = dg[dg.Région == 'Centre-Val de Loire']
    dg_BFC = dg[dg.Région == 'Bourgogne-Franche-Comté']
    dg_NORM = dg[dg.Région == 'Normandie']
    dg_HDF = dg[dg.Région == 'Hauts-de-France']
    dg_BRE = dg[dg.Région == 'Bretagne']
    dg_NA = dg[dg.Région == 'Nouvelle-Aquitaine']
    dg_ARA = dg[dg.Région == 'Auvergne-Rhône-Alpes']
    dg_GE = dg[dg.Région == 'Grand Est']
    dg_OCC = dg[dg.Région == 'Occitanie']
    dg_PdL = dg[dg.Région == 'Pays de la Loire']
    
    Choix_région = st.selectbox("Corrélation entre la température et la consommation d'énergie par région",  options = ['Hauts-de-France','Île-de-France','Normandie','Nouvelle-Aquitaine', 'Occitanie','Pays de la Loire',"Provence-Alpes-Côte d'Azur",'Auvergne-Rhône-Alpes','Bourgogne-Franche-Comté','Bretagne','Centre-Val de Loire','Grand Est'])
    st.button("Corrélation")
    
    def correlation(Choix_région):
        if Choix_région == 'Île-de-France':
            df = dg_IDF
        elif Choix_région == 'Centre-Val de Loire':
            df = dg_CVL   
        elif Choix_région == 'Bourgogne-Franche-Comté':
            df = dg_BFC
        elif Choix_région == 'Normandie':
            df = dg_NORM   
        elif Choix_région == 'Hauts-de-France':
            df = dg_HDF  
        elif Choix_région == 'Bretagne':
            df =  dg_BRE  
        elif Choix_région == 'Nouvelle-Aquitaine':
            df = dg_NA
        elif Choix_région == 'Auvergne-Rhône-Alpes':
            df =  dg_ARA  
        elif Choix_région == 'Grand Est':
            df =  dg_GE
        elif Choix_région == 'Occitanie':
            df =  dg_OCC  
        elif Choix_région == 'Pays de la Loire':
            df = dg_PdL
        return(df)
        
    
    sns.heatmap(correlation(Choix_région).corr() ,annot = True , cmap = 'RdBu_r' , center = 0  )
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    st.title('Régression Linéaire') 
    dF = pd.read_csv("France_Conso_temp.csv")
    #st.dataframe(dF) 
    st.write('France : ')
    target= dF['Consommation (MW)']
    data = dF.drop('Consommation (MW)' , 1)
    
    
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=789)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    st.write('Coefficient de détermination du modèle :', lr.score(X_train, y_train))
    st.write('Coefficient de détermination obtenu par Cv :', cross_val_score(lr,X_train,y_train).mean())
    st.write("Le score du modèle sur l'ensemble de test est :", lr.score(X_test, y_test) )
    
    dR = pd.read_csv("Rég_Conso_temp.csv")
    if st.checkbox("Afficher jeu de données par Région"):
        st.dataframe(dR) 
    
    Choix_région_Reg = st.selectbox(" Modèl de Régression Linéaire entre la température et la consommation d'énergie par région",  options = ['Hauts-de-France','Île-de-France','Normandie','Nouvelle-Aquitaine', 'Occitanie','Pays de la Loire',"Provence-Alpes-Côte d'Azur",'Auvergne-Rhône-Alpes','Bourgogne-Franche-Comté','Bretagne','Centre-Val de Loire','Grand Est'])
    st.button("Régression Linéaire")
    
    dR_IDF = dR[dR.Région == 'Île-de-France']
    dR_CVL = dR[dR.Région == 'Centre-Val de Loire']
    dR_BFC = dR[dR.Région == 'Bourgogne-Franche-Comté']
    dR_NORM = dR[dR.Région == 'Normandie']
    dR_HDF = dR[dR.Région == 'Hauts-de-France']
    dR_BRE = dR[dR.Région == 'Bretagne']
    dR_NA = dR[dR.Région == 'Nouvelle-Aquitaine']
    dR_ARA = dR[dR.Région == 'Auvergne-Rhône-Alpes']
    dR_GE = dR[dR.Région == 'Grand Est']
    dR_OCC = dR[dR.Région == 'Occitanie']
    dR_PdL = dR[dR.Région == 'Pays de la Loire']
    
    
    def RegLin(Choix_région_Reg):
        if Choix_région_Reg == 'Île-de-France':
            dH = dR_IDF
        elif Choix_région_Reg == 'Centre-Val de Loire':
            dH = dR_CVL   
        elif Choix_région_Reg == 'Bourgogne-Franche-Comté':
            dH = dR_BFC
        elif Choix_région_Reg == 'Normandie':
            dH = dR_NORM   
        elif Choix_région_Reg == 'Hauts-de-France':
            dH = dR_HDF  
        elif Choix_région_Reg == 'Bretagne':
            dH =  dR_BRE  
        elif Choix_région_Reg == 'Nouvelle-Aquitaine':
            dH = dR_NA
        elif Choix_région_Reg == 'Auvergne-Rhône-Alpes':
            dH =  dR_ARA  
        elif Choix_région_Reg == 'Grand Est':
            dH =  dR_GE
        elif Choix_région_Reg == 'Occitanie':
            dH =  dR_OCC  
        elif Choix_région_Reg == 'Pays de la Loire':
            dH = dR_PdL 
        return(dH)
    st.dataframe(RegLin(Choix_région_Reg))
    
    targetR= RegLin(Choix_région_Reg)['Consommation (MW)']
    dataR = RegLin(Choix_région_Reg).drop(['Consommation (MW)', 'Région'] , 1)
    X_train, X_test, y_train, y_test = train_test_split(dataR, targetR, test_size=0.2, random_state=789)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    st.write('Coefficient de détermination du modèle :', lr.score(X_train, y_train))
    st.write('Coefficient de détermination obtenu par Cv :', cross_val_score(lr,X_train,y_train).mean())
    st.write("Le score du modèle sur l'ensemble de test est :", lr.score(X_test, y_test) ) 



elif menu == 'Etude de la consommation':
    consommation = st.radio("ETUDE DE LA CONSOMMATION",
                        ('Saisonnalité', 'Décomposition automatique', 
                         'Fonction autocorrélation', 'Autocorrélogramme',
                         'Modèle SARIMA', 'Evaluation du modèle'))
    if consommation == 'Saisonnalité':
        st.markdown("#### Etude de la consommation énergétique \
                    selon la saisonnalité")
    
        df_cons = pd.read_csv('consommation.csv')
        df_fig = px.line(df_cons, y = 'Consommation (MW)', x= 'Month', 
                     height=400, width=800)

        df_fig.update_xaxes(title_font=dict(size=18, family='Arial',
                                        color='black'))
        df_fig.update_yaxes(title_font=dict(size=18, family='Arila',
                                        color='black'))
        st.plotly_chart(df_fig)
        st.markdown(''' * La série temporelle présente une tendance stable \
                    de la consommation énergétique en fonction du temps.
* La consommation d’énergie varie fortement en fonction des saisons.''')
 
    elif consommation == 'Décomposition automatique':
        
        st.markdown("#### Décomposition automatique de la série temporelle")
        from statsmodels.tsa.seasonal import seasonal_decompose
        df_cons  = pd.read_csv('consommation.csv', header=0, parse_dates=[0], 
                               index_col=0, squeeze=True)
        
        if st.button('Seasonal_decompose'):
            con = seasonal_decompose(df_cons,model = 'multiplicatif')
            fig = con.plot()
            fig
        
            st.markdown(''' * Pas de  tendance globale pour la consommation \
                    d'énergie en France en fonction du temps.
* Une saisonnalité annuelle de période 12 (mois).

* Le résidu est un bruit blanc faible.''')

    elif consommation == 'Fonction autocorrélation':
        st.markdown('''#### Différenciation de la série temporelle
Une différentiation simple d'ordre 1 et une \
             différentation saisonnière sont réalisées pour rendre la \
                 série stationnaire.''')
        
        df_cons  = pd.read_csv('consommation.csv', header=0, parse_dates=[0], 
                               index_col=0, squeeze=True)
        if st.button("Différenciation de la série"):
            df_conslog = np.log(df_cons)
            conslog_1 = df_conslog.diff().dropna()
            conslog_2 = conslog_1.diff(periods = 12).dropna()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,6)) 
            conslog_2.plot(ax = ax1)
            pd.plotting.autocorrelation_plot(conslog_2, ax = ax2);
            fig
            st.markdown(''' * Les valeurs de consommation décroissent plus \
                    rapidement vers 0.
                        
* La p-value du test  augmenté de **Dickey-Fuller** = 6,49*e-6.''')

    elif consommation == 'Autocorrélogramme':
        st.markdown('''#### Détermination des ordres du modèle \
                    *SARIMA(p,d,q)(P,D,Q)n* par autocorrélogramme''')
        from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
        if st.button("Autocorrélogramme"):
            df_cons  = pd.read_csv('consommation.csv', header=0, parse_dates=[0], 
                               index_col=0, squeeze=True)
            df_conslog = np.log(df_cons)
            conslog_1 = df_conslog.diff().dropna()
            conslog_2 = conslog_1.diff(periods = 12).dropna()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
            plot_acf(conslog_2, lags = 36, ax=ax1)
            plot_pacf(conslog_2, lags = 36, ax=ax2)
            fig
        
            st.markdown(''' * Différentiation simple(d) d’ordre 1 et une \
                    différentiation saisonnière (D, n=12).  \
                        Donc **d** = 1, **D** = 1 et **n** = 12.
* L'autocorrélation simple et partielle tendent vers 0. Donc c'est un \
    processus **ARMA(p,q)**. On va commencer par estimer via un ***ARMA(2,2)***.

* Pour les pics saisonniers **(12, 24, 36)**, une coupure de l'ACF après \
    le pic 12 et une décroissance de la PACF. Donc modèle **MA(2)** avec \
        **P** = 0 et **Q** = 2. 
    
* Ainsi on va entrainer dans un premier un modèle ***SARIMA(2,1,2)(0,1,2)12***.''')
    
    elif consommation == 'Modèle SARIMA':
        st.markdown('''#### Entrainement du modèle ***SARIMA(0,1,1)(0,1,2)12***

Après plusieurs essais d’entrainement des modèles SARIMA, nous avons définis \
    un modèle SARIMA optimal : ***SARIMA(0,1,1)(0,1,2)12***.''')
    
        if st.button('Entrainement du modèle'):
             
             df_cons  = pd.read_csv('consommation.csv', header=0, parse_dates=[0], 
                           index_col=0, squeeze=True)
             df_conslog = np.log(df_cons)
             model = sm.tsa.SARIMAX(df_conslog[df_conslog.index <'2021-01-01']
                                    ,order=(0,1,1),seasonal_order=(0,1,2,12))
             sarima_op=model.fit()
             fig = sarima_op.summary()
             fig
    
             
             st.markdown('''Le résumé statistique du modèle montre :

* la p-value **(P > |z|)** des tous les coefficients est inférieure à **5%**. \
    Donc les termes du modèle SARIMA sont considérés comme significatifs.

* le **test de Ljung-Box** est un test de blancheur des résidus. C'est un test \
    statistique qui vise à rejeter ou non l'hypothèse H0 : \
        « Le résidu est un bruit blanc ». Ici on lit sur la ligne **Prob(Q)** \
            que la **p-value** de ce test est de **0.21**, donc on ne rejette \
                pas l'hypothèse.

* Le **test de Jarque-Bera** est un test de normalité. C'est un test statistique \
    qui vise à rejeter ou non l'hypothèse H0 : « Le résidu suit \
        une distribution normale ». Ici on lit sur la ligne **Prob (JB)** \
            que la **p-value** du test est de **0.14**. On ne rejette donc \
                pas l'hypothèse
* Le résidu de notre modèle est un bruit blanc et suit une \
    distribution normale.''')
    
    else :
        st.markdown('''#### Evaluation de la qualité du modèle \
                    ***SARIMA(0,1,1)(0,1,2)12***''')

        
        if st.button('Prédiction de la consommation'):
            st.markdown("***Prédiction de la consommation énergétique en France \
                        pour l’année 2021 (Année -1).***")
            
            df_cons  = pd.read_csv('consommation.csv', header=0, parse_dates=[0], 
                      index_col=0, squeeze=True)
            df_conslog = np.log(df_cons)
            model = sm.tsa.SARIMAX(df_conslog[df_conslog.index <'2021-01-01']
                               ,order=(0,1,1),seasonal_order=(0,1,2,12))
            sarima_op=model.fit()
            cons_pred = np.exp(sarima_op.predict(96, 106))
            df_reel  = pd.read_csv('cons_reel.csv', header=0, parse_dates=[0], 
                      index_col=0, squeeze=True)
        
            df_final = pd.concat([cons_pred, df_reel], axis=1)
            fig = px.line(df_final, markers=True, height=600, width=900)
            fig.update_layout(xaxis_rangeslider_visible=True)
            fig.update_layout(font_family="Arial",
                          font_color="Black", title_font_family="Arial",
                          title_font_color="Black",
                          title_font_size=20)
            fig.update_xaxes(title_text='Mois', 
                         title_font=dict(size=18, family='Arial',
                                        color='black'))
            fig.update_yaxes(title_text='Consommation (MW)' ,
                         title_font=dict(size=18, family='Arial',
                                        color='black')) 
            fig
            st.markdown(''' * les prédictions de la consommation réalisées par notre\
                    modèle pour l’année **2021** sont très proches des valeurs \
                        réelles.
 
 ***Pourcentage de l'erreur absolue moyenne du modèle (MAPE)***''')
            
            st.metric('MAPE', '2.30 %')
            
            st.markdown("* Le modèle est précis à environ 97,70 % pour prédire \
                        les 11 prochaines observations.")
   
elif menu == 'Prédiction de la consommation':
    
    st.markdown("#### Prédiction de la consommation énergétique")
    echelle = st.selectbox("Echelle à prédire", ('Nationale', 'Régionale'))
    if echelle == "Nationale" :
        
       
        if st.button('Prédiction'):
           
            st.markdown('''##### Prédiction de la consommation énergétique en \
                        France pour 2022 à l'aide du modèle SARIMA(0,1,1)(0,1,2)12''')


            df_cons  = pd.read_csv('consommation.csv', header=0, 
                                   parse_dates=[0],index_col=0, squeeze=True)
            df_conslog = np.log(df_cons)
            model = sm.tsa.SARIMAX(df_conslog, order=(0,1,1), 
                           seasonal_order=(0,1,2,12))
            sarima_F=model.fit()
            prediction = sarima_F.get_forecast(steps =13).summary_frame() 
            fig, ax = plt.subplots(figsize = (17,7))
            plt.plot(df_cons, linewidth=2)
            prediction = np.exp(prediction)
            prediction['mean'].plot(ax = ax, style = 'k--', linewidth=3) 
            ax.fill_between(prediction.index, prediction['mean_ci_lower'], 
                    prediction['mean_ci_upper'], color='k', alpha=0.1)
            
            plt.xlabel("Mois")
            plt.ylabel("Consommation (MW)")
            fig
            
    else:
        if st.button('Prédiction'):
            st.markdown(''' ##### Prédiction de la consommation énergétique en \
                        Ile de France pour 2022 à l'aide du modèle SARIMA(1,1,0)(0,1,2)12''')
   
        
            cons_IDF = pd.read_csv('cons_IDF.csv', header=0, 
                                   parse_dates=[0], index_col=0, squeeze=True)
            conslog_IDF = np.log(cons_IDF)
            model = sm.tsa.SARIMAX(conslog_IDF, order=(1,1,0), 
                                   seasonal_order=(0,1,2,12))
            sarima_IDF2=model.fit()
            prediction_IDF = sarima_IDF2.get_forecast(steps =13).summary_frame()  
            fig, ax = plt.subplots(figsize = (17,7))
            plt.plot(cons_IDF, 'g', linewidth=2)
            prediction_IDF = np.exp(prediction_IDF) 
            prediction_IDF['mean'].plot(ax = ax, style = 'r--', linewidth=3) 
            ax.fill_between(prediction_IDF.index, prediction_IDF['mean_ci_lower'],
                            prediction_IDF['mean_ci_upper'], 
                            color='b', alpha=0.1)
            plt.xlabel("Mois")
            plt.ylabel("Consommation (MW)")
            fig
    
