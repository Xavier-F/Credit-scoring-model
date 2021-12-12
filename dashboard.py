import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

import shap
import lightgbm as lgb
from scipy.special import expit as sigmoid

#Import du modèle
bst = lgb.Booster(model_file='mode.txt')

# Création d'une fonction qui va donner un label lisible à la classication
def class_label (number):
    if number==0:
        label = "Crédit accepté"
        return label
    elif number ==1 :
        label = "Crédit rejeté"
        return label

# Création d'une fonction qui va donner un label lisible au sexe
def gender_label (gender):
    if gender=="M":
        label = "HOMME"
        return label
    elif gender=="F" :
        label = "FEMME"
        return label

# Définition de la fonction objective : focale loss
def focal_loss_lgb_correct(y_true, y_pred,alpha, gamma):
    a,g = alpha, gamma
    def fl(x,t):
        p = 1/(1+np.exp(-x))
        return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )
    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    return grad, hess

# Défintion de la fonction d'évalulation LGBM associée à la focale loss
def focal_loss_lgb_eval_correct( y_true,y_pred, alpha, gamma):
    a,g = alpha, gamma
    p = 1/(1+np.exp(-y_pred))
    loss = -( a*y_true + (1-a)*(1-y_true) ) * (( 1 - ( y_true*p + (1-y_true)*(1-p)) )**g) * ( y_true*np.log(p)+(1-y_true)*np.log(1-p) )
    # (eval_name, eval_result, is_higher_better)
    return 'focal_loss', np.mean(loss), False




# Import du dataframe
df=pd.read_csv("dashboard_df.csv")

# Ajout du titre du tableau de bord
st.title('TABLEAU DE BORD - AIDE A LA DECISION CREDIT UTILISATEUR')
st.write("")
st.write("")

# Faire saisir à l'utilisateur l'ID du client que l'on souhaite étudier
input_ID= st.text_input(label="Identifiant client", value="0", max_chars=6)
st.write("")
# Filtrer les données personnelles de ce client
client_df = df.iloc[df.index==int(input_ID)]


# Annoncer le résultat de la demande de crédit
col1, col2 = st.columns(2)
col1.metric("Résultat", class_label(client_df["TARGET"].values[0]))
col2.write("L'algorithme d'aide à la décision a retourné ce résultat pour ce client ID. Si vous souhaitez en savoir plus, vous pouvez cliquez sur l'une des sections ci-dessous.")

# Ajout du premier widget
with st.expander("Données personelles"):
    
    # Affichage du genre du client
    st.write("Genre de l'utilisateur")    
    st.header(gender_label(client_df["GENDER"].values[0]))
    st.write("")
    st.write("")

    # Affichage du statut familial du client
    st.write("Statut familial de l'utilisateur")    
    st.header(client_df["FAMILY_STATUS"].values[0].upper())
    st.write("")
    st.write("")

    # Affichage du nombre d'enfants du client
    st.write("nombre d'enfants de l'utilisateur")    
    st.header(client_df["CNT_CHILDREN"].values[0])
    st.write("")
    st.write("")
 
   
    # Affichage de la distribution de l'âge et la valeur du client
    st.write("Age de l'utilisateur et comparaison avec la distribution")
       
    fig = plt.figure(figsize=(5, 5))
    plt.hist(x=df["AGE"], bins=10,color='#d4fffb')
    plt.axvline(x=client_df["AGE"].values[0], color='#3446eb')
    plt.text(50,13000 , 'Age utilisateur {}'.format(round(client_df["AGE"].values[0],0)), fontsize=10, color='#3446eb')
    plt.show()
    st.pyplot(fig)
    

       
    # Affichage de la durée depuis laquelle le client est employé
    st.write("Ancienneté dans l'emploi actuel et comparaison avec la distribution")
       
    fig = plt.figure(figsize=(5, 5))
    plt.hist(x=df["DAYS_WORK"], bins=10,color='#d4fffb')
    plt.axvline(x=client_df["DAYS_WORK"].values[0], color='#3446eb')
    plt.text(-15000, 40000, 'Ancienneté {}'.format(round(client_df["DAYS_WORK"].values[0],0)), fontsize=10, color='#3446eb')
    plt.show()
    st.pyplot(fig)


# Ajout du second widget
with st.expander("Prédiction individuelle"):
    df2 = pd.read_csv('encoded_train_X.csv')
    
    # Affichage de la prédiction individuelle avec les shap values
    st.write("Afficher de la prédiction individuelle du client, générée avec le modèle")
    
    from lightgbm.sklearn import LGBMClassifier
    
    encoded_train_X = pd.read_csv("encoded_train_X.csv")
    encoded_val_X = pd.read_csv("encoded_val_X.csv",  index_col=0)
    y_train = pd.read_csv("y_train.csv", index_col=0)
    y_val = pd.read_csv("y_val.csv", index_col=0)
    
    
    #Création du modèle final
    a, g =0.7,20
    focal_loss2 = lambda x,y: focal_loss_lgb_correct(x, y, alpha=a, gamma=g)
    focal_loss_eval2 = lambda x,y: focal_loss_lgb_eval_correct(x, y, alpha=a, gamma=g)
    
    
    # On instancie le modèle
    model_final2 =LGBMClassifier(objective=focal_loss2 ,
                           metric="",
                           learning_rate=0.1,
                           n_estimators=10,
                           n_jobs=-1
                           
                          )
    
    # On réentraine le modèle
    model_final2.fit( encoded_train_X,y_train, 
                eval_metric=focal_loss_eval2,
              eval_set =(encoded_val_X,y_val),
              early_stopping_rounds=10)
    
    #On modifie les dimensions du vecteur y
    y_train = y_train.squeeze()
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    data_for_prediction_A = encoded_train_X.iloc[encoded_train_X.index==int(input_ID)] 
    data_for_prediction_array_A = data_for_prediction_A.values.reshape(1, -1).astype('float')
    col_name = list(data_for_prediction_A.index)
    
    explainer = shap.TreeExplainer(model_final2)
    shap_values = explainer.shap_values(data_for_prediction_array_A)
    
    st.pyplot(shap.force_plot(explainer.expected_value, shap_values,matplotlib=True))
        