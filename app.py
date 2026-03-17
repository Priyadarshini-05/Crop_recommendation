from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form

    data = [
        float(form_data["N"]),
        float(form_data["P"]),
        float(form_data["K"]),
        float(form_data["temperature"]),
        float(form_data["humidity"]),
        float(form_data["ph"]),
        float(form_data["rainfall"])
    ]

    sample = pd.DataFrame([data], columns=[
        "N","P","K","temperature","humidity","ph","rainfall"
    ])

    prediction = model.predict(sample)[0]

    # 👇 Send values back to HTML
    return render_template("index.html", result=prediction, values=form_data)

if __name__ == "__main__":
    app.run(debug=True)