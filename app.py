import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, app, jsonify, url_for, render_template
import json
import pickle


app = Flask(__name__)

## Load classifier , preprocessors
classifier = pickle.load(open('Term_Deposit_Subscription_Predictor.pkl','rb'))
scaler = pickle.load(open('Standard_Scaler.pkl','rb'))
onehotencoder = pickle.load(open('OHE.pkl','rb'))
ordinalencoder = pickle.load(open('OE.pkl','rb'))
labelencoder = pickle.load(open('LE.pkl','rb'))



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [x for x in request.form.values()]
    ## Preprocessing the query
    query = np.array(input_data).reshape(1,-1)

    # Nominal features
    nominal_data = [query[0][1], query[0][2], query[0][4], query[0][6], query[0][7], 
                query[0][8], query[0][10], query[0][15]]
    nominal_data = np.array(nominal_data)
    nominal_data = nominal_data.reshape(1,-1)
    nominal_data_encoded1 = onehotencoder.transform(nominal_data)
                                        
    # Ordinal Features
    ordinal_data = np.array(query[0][3])
    ordinal_data = ordinal_data.reshape(1,-1)
    ordinal_data_encoded1 = ordinalencoder.transform(ordinal_data)


    # Numerical features
    numerical_data = [query[0][0],query[0][5],query[0][9],query[0][11],query[0][12],query[0][13],query[0][14]]
    numerical_data = np.array(numerical_data)

    # Making the query
    updated_query = np.append(numerical_data, nominal_data_encoded1)
    final_query = np.append(updated_query,ordinal_data_encoded1)
    final_query = final_query.reshape(1,-1)

    final_query_scaled = scaler.transform(final_query)

    result = classifier.predict(final_query_scaled)

    if result == 0.0:
        return render_template('home.html',prediction_text="The customer will not buy the term deposit")
    return render_template('home.html',prediction_text="The customer will will buy the term deposit")


if __name__ == "__main__":
    app.run(debug=True)