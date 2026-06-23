import nltk

nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

from flask import Flask, render_template, request
from routes.api import api_bp, load_model_globals
from database.db import init_db

app = Flask(__name__)

app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 3600

app.register_blueprint(api_bp)


@app.after_request
def add_cache_headers(response):
    if request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "public, max-age=3600"
    return response


@app.route("/")
def index():
    return render_template("index.html")


@app.errorhandler(404)
def not_found(e):
    return {"error": "Not found"}, 404


@app.errorhandler(500)
def server_error(e):
    return {"error": "Internal server error"}, 500


def init_app():
    init_db()
    load_model_globals()


if __name__ == "__main__":
    init_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
