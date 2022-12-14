from flask import Flask, render_template
from flask_googlemaps import GoogleMaps
from flask_googlemaps import Map

app = Flask(__name__)

app.config['GOOGLEMAPS_KEY'] = "8JZ7i18MjFuM35dJHq70n3Hx4"

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/createOrder")
def createOrder():
    return render_template("createOrder.html")


@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/tarif")
def tarif():
    return render_template("tarif.html")


@app.route("/registration")
def registration():
    return render_template("registration.html")


if __name__ == "__main__":
    app.run(debug=True)
