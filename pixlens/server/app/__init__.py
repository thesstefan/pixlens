import flask

from pixlens.server.app import ml_models
from pixlens.server.app.routes import bp


def create_app() -> flask.Flask:
    app = flask.Flask(__name__)
    app.register_blueprint(bp)

    # Load moedl and cache it
    ml_models.get_detect_and_segment_model()

    return app
