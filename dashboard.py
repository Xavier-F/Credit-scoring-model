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
    if number<0.5:
        label = "Crédit accepté"
        return label
    else :
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

# Défintion de la fonction qui va retourner la description détaillée d'une variable du jeu de données
def get_description(test):
    split_list=["+","-","/","*"]
    prefix_list=["APP_","PREV_BUR_MEAN_","PREV_BUR_","BUR_","PREVIOUS_APPL_","CREDIT_CARD_","POS_CASH_","INST_PAY_","_mean","_min","_y_max","_max","_count"]
    description_test=[]
    #On test si la chaine comporte un caractère opérateur
    if any(split in test for split in split_list):
        for split in split_list :
            if test.find(split)!=-1 :
                print(split)
                test=test.split(split)
                break

        #On parcours les éléments de la liste pour supprimer leur préfix
        for i in range(len(test)) :
            for prefix in prefix_list :
                if test[i].find(prefix)!=-1 :
                    test[i]=test[i].replace(prefix,"")
            description_test.append(description.loc[description ["Row"] == test[i]]["Description"].values[0])
        return description_test
    else :
        for prefix in prefix_list :
            if test.find(prefix)!=-1 :
                test=test.replace(prefix,"")
        description_test.append(description.loc[description ["Row"] == test]["Description"].values[0])
        return description_test


# Import du dataframe
df = pd.read_csv("dashboard_df.csv")
df2 = pd.read_csv("encoded_train_X.csv")

# Ajout du titre du tableau de bord
st.title('TABLEAU DE BORD - AIDE A LA DECISION CREDIT UTILISATEUR')
st.write("")
st.write("")

# Faire saisir à l'utilisateur l'ID du client que l'on souhaite étudier
input_ID= st.text_input(label="Identifiant client", value="0", max_chars=6)
st.write("")
# Filtrer les données personnelles de ce client
data_for_prediction_A = df2.iloc[df2.index==int(input_ID)]
data_for_prediction_array_A = data_for_prediction_A.values.reshape(1, -1).astype('float')
col_name = list(data_for_prediction_A.index)

#Calculer la prédiction individuelle en utilisant le modèle chargé
result_proba = sigmoid(bst.predict(data_for_prediction_array_A))[0]
result_label = class_label(sigmoid(bst.predict(data_for_prediction_array_A))[0])

if result_proba < 0.5 :
    # Annoncer le résultat de la demande de crédit
    col1, col2 = st.columns(2)
    col1.success(result_label)
    col2.write("L'algorithme d'aide à la décision a retourné ce résultat pour ce client ID.")

    # Annoncer la probabilité calculée
    col3, col4 = st.columns(2)
    col3.metric(label="", value="{:.0%}".format(result_proba))
    col4.write("L'algorithme d'aide à la décision a retourné cette probabilité de défaut pour ce client ID.")
else :
    col1, col2 = st.columns(2)
    col1.error(result_label)
    col2.write("L'algorithme d'aide à la décision a retourné ce résultat pour ce client ID. ")

    # Annoncer la probabilité calculée
    col3, col4 = st.columns(2)
    col3.metric(label="", value="{:.0%}".format(result_proba))
    col4.write("L'algorithme d'aide à la décision a retourné cette probabilité de défaut pour ce client ID.")



# # Ajout du premier widget
with st.expander("Visualisation des principales variables ayant généré cette probabilité"):

    # Création du nouveau modèle, car le chargement du modèle sauvegardé ne fonctionne pas avec les force_plot
    from lightgbm.sklearn import LGBMClassifier

    #Chargement des jeux de données complets
    encoded_train_X = pd.read_csv("encoded_train_X.csv")
    encoded_val_X = pd.read_csv("encoded_val_X.csv",  index_col=0)
    encoded_val_X=encoded_val_X.iloc[:,0:]
    y_train = pd.read_csv("y_train.csv", index_col=0)
    y_train=y_train.iloc[1:,:]
    y_val = pd.read_csv("y_val.csv", index_col=0)
    description = pd.read_csv("./Projet+Mise+en+prod+-+home-credit-default-risk/HomeCredit_columns_description.csv", encoding = "ISO-8859-1")

    #Création des fonctions obectives et évaluation
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
    # Calcul des shaps values
    line_A = int(input_ID)
    data_for_prediction_A = encoded_train_X.iloc[line_A]
    data_for_prediction_array_A = data_for_prediction_A.values.reshape(1, -1).astype('float')

    col_name = list(data_for_prediction_A.index)

    # On calcule la shap value pour cet enregistrement
    explainer = shap.TreeExplainer(model_final2)
    shap_values_A = explainer.shap_values(data_for_prediction_array_A)
    # SHAP Value visualization for team A
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values_A, data_for_prediction_array_A,feature_names=col_name,show=False,matplotlib=True)
    plt.savefig('grafico.png')
    #Affichage de l'image du force plot
    from PIL import Image
    image = Image.open('grafico.png')

    st.image(image, caption='Les variables rouges tirent la réponse vers l acceptation de crédit. Les variables bleues vers le rejet')

# Ajout du deuxième widget
with st.expander("Visualisation des 3 variables ayant tiré la prédiction vers le rejet"):
    min_index = shap_values_A.argsort()[0][0:3]

    for i in range(len(min_index)):
        j=i+1
        st.subheader("La top "+str(j)+ " variable est " +col_name[min_index[i]])
        indiv_value = data_for_prediction_A[col_name[min_index[i]]]
        all_values = list(encoded_train_X[col_name[min_index[i]]].values)

        #On va afficher en clair la description du champ pour l'utilisateur
        test=col_name[min_index[i]]
        description_list = get_description(test)
        for d in description_list:
            st.write("- La description de la variable est la suivante")
            st.write('"'+d+'"')



        st.write("- La moyenne de cette variable est :"+str(round(encoded_train_X[col_name[min_index[i]]].values.mean(),2)))
        st.write("- La médiane de cette variable est :"+str(round(np.median(encoded_train_X[col_name[min_index[i]]].values),2)))
        y_max = pd.Series(all_values).value_counts().values.max()
        fig=plt.figure(figsize=(10,5))
        plt.title("Comparaison de la valeur individuelle par rapport au reste de la distribution",fontsize=15)
        plt.hist(all_values, color='red')
        plt.axvline(indiv_value, color='black')
        plt.text(indiv_value+(0.1*indiv_value), y_max, 'Valeur individuelle {}'.format(round(indiv_value,0)), fontsize=12, color='black')
        plt.show()

        st.pyplot(fig)

# Ajout du troisième widget
with st.expander("Visualisation des 3 variables ayant tiré la prédiction vers l'acceptation'"):
    min_index = shap_values_A.argsort()[0][-3:]

    for i in range(len(min_index)):
        j=i+1
        st.subheader("La top "+str(j)+ " variable est " +col_name[min_index[i]])
        indiv_value = data_for_prediction_A[col_name[min_index[i]]]
        all_values = list(encoded_train_X[col_name[min_index[i]]].values)

        #On va afficher en clair la description du champ pour l'utilisateur
        test=col_name[min_index[i]]
        description_list = get_description(test)
        for d in description_list:
            st.write("- La description de la variable est la suivante")
            st.write('"'+d+'"')



        st.write("- La moyenne de cette variable est :"+str(round(encoded_train_X[col_name[min_index[i]]].values.mean(),2)))
        st.write("- La médiane de cette variable est :"+str(round(np.median(encoded_train_X[col_name[min_index[i]]].values),2)))
        y_max = pd.Series(all_values).value_counts().values.max()
        fig=plt.figure(figsize=(10,5))
        plt.title("Comparaison de la valeur individuelle par rapport au reste de la distribution",fontsize=15)
        plt.hist(all_values, color='green')
        plt.axvline(indiv_value, color='black')
        plt.text(indiv_value+(0.1*indiv_value), y_max, 'Valeur individuelle {}'.format(round(indiv_value,0)), fontsize=12, color='black')
        plt.show()

        st.pyplot(fig)


# Ajout du quatrième widget
with st.expander("Données personelles"):
    client_df = df.iloc[df.index==int(input_ID)]
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
