import flask

from pixlens.server.app import ml_models
from pixlens.server.app.routes import bp
from pixlens.server.config import Config


def create_app() -> flask.Flask:
    app = flask.Flask(__name__)
    app.register_blueprint(bp)
    app.config.from_object(Config)

    with app.app_context():
        ml_models.load_model()

    return app
