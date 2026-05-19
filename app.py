from flask import Flask, jsonify, request
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Endpoint de test — verifică că serverul rulează
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "ok", "mesaj": "Serverul Flask functioneaza!"})

# Endpoint predictii — primeste date financiare si returneaza predictia
@app.route('/predict', methods=['POST'])
def predict():
    date = request.get_json()

    cheltuieli = date['cheltuieli_per_luna']   # lista de 6 valori
    luni_viitor = date['luni_viitor']          # 1, 3 sau 12

    # Pregatim datele pentru scikit-learn
    X = np.array(range(len(cheltuieli))).reshape(-1, 1)
    y = np.array(cheltuieli)

    # Antrenam modelul
    model = LinearRegression()
    model.fit(X, y)

    # Calculam predictia pentru luna dorita
    luna_predictie = len(cheltuieli) - 1 + luni_viitor
    predictie = model.predict([[luna_predictie]])[0]

    # Calculam trendul fata de luna curenta
    cheltuieli_curente = cheltuieli[-1]
    diferenta = predictie - cheltuieli_curente
    procent = (diferenta / cheltuieli_curente) * 100 if cheltuieli_curente > 0 else 0

    return jsonify({
        "predictie_total": round(float(predictie), 2),
        "cheltuieli_curente": cheltuieli_curente,
        "diferenta": round(float(diferenta), 2),
        "procent_trend": round(float(procent), 2)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)