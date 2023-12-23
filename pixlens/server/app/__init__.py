import flask

from pixlens.server.app import ml_models
from pixlens.server.app.routes import bp
from pixlens.server.config import Config


def create_app() -> flask.Flask:
    app = flask.Flask(__name__)
    app.register_blueprint(bp)
    app.config.from_object(Config)

    config = Config()

    if config.DETECT_SEGMENT_MODEL:
        ml_models.get_detect_segment_model(
            config.DETECT_SEGMENT_MODEL,
            config.DEVICE,
        )

    if config.EDIT_MODEL:
        ml_models.get_edit_model(config.EDIT_MODEL, config.DEVICE)

    return app
