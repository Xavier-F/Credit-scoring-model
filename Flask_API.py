from flask import Flask
from flask import request
import lightgbm as lgb
import pandas as pd
from scipy.special import expit as sigmoid

app= Flask(__name__)

#Import du modèle
model = lgb.Booster(model_file='mode.txt')

#Import du jeu de données
X_train = pd.read_csv('encoded_train_X.csv') 

@app.route("/api", methods=["GET"])

def api_endpoint():
    user_id = request.args.get("id")
    print(user_id)
    
    X_user_infos = X_train[X_train.index == int(user_id)]
    pred = sigmoid(model.predict(X_user_infos))
    
    return str(pred)

app.run(debug=True)
