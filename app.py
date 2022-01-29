from logging import debug
from anyio import ExceptionGroup
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
from catboost import CatBoostRegressor


app = Flask(__name__)



@app.route("/")
#def classify(a, b, c, d, e , f): # Reshape the array
#    prediction = model.predict() # Retrieve from dictionary
 #   return prediction # Return the prediction

def home():
    return render_template('home.html')


@app.route("/predicts", methods=['GET', 'POST'])
def predicts():    #User input
    # Session_frequency = request.form.get('sessions', True )
    # Avg_RTT_UL = request.form.get('Avg_RTT_UL', True)
    # Avg_RTT_DL =request.form.get('Avg_RTT_DL', True)
    # Avg_BearerTP_UL =request.form.get('Avg_BearerTP_UL', True)
    # Avg_BearerTP_DL = request.form.get('Avg_BearerTP_DL', True)
    # TCP_Retrans_vol_UL =request.form.get('TCP_Retrans_vol_UL', True)
    # TCP_Retrans_vol_DL =request.form.get('TCP_Retrans_vol_DL', True)
    # Total_UL = request.form.get('Total_UL', True)
    # Total_DL = request.form.get('Total_DL', True)
    # Dur_ms = request.form.get('Durations', True)
    try:
        Session_frequency = request.args.get('sessions')
        Avg_RTT_UL = request.args.get('Avg_RTT_UL')
        Avg_RTT_DL =request.args.get('Avg_RTT_DL')
        Avg_BearerTP_UL =request.args.get('Avg_BearerTP_UL')
        Avg_BearerTP_DL = request.args.get('Avg_BearerTP_DL')
        TCP_Retrans_vol_UL =request.args.get('TCP_Retrans_vol_UL')
        TCP_Retrans_vol_DL =request.args.get('TCP_Retrans_vol_DL')
        Total_UL = request.args.get('Total_UL')
        Total_DL = request.args.get('Total_DL')
        Dur_ms = request.args.get('Durations')

        Avg_RTT = Avg_RTT_UL + Avg_RTT_DL
        Avg_TP = Avg_BearerTP_UL + Avg_BearerTP_DL
        Avg_TCP = TCP_Retrans_vol_UL + TCP_Retrans_vol_DL
        Total_Data =Total_UL + Total_DL
        
        #scaling
        scaler = StandardScaler()

        scaled = [Avg_RTT, Avg_TCP, Avg_TP, Session_frequency, Total_Data, Dur_ms]
        model = pickle.load(open('model.pkl', 'rb'))

        prediction = model.predict([scaled])    
        return render_template("output.html", predictions = prediction)
    except:
        return 'Error'
if __name__ == "__main__":
    app.run(debug=True)