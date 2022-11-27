from flask import Flask, render_template
from flask_googlemaps import GoogleMaps
from flask_googlemaps import Map
import jinja2

app = Flask(__name__)

@app.route("/")
def main():
    return render_template("paramBussiness.html")


if __name__ == "__main__":
    app.run(debug=True)
