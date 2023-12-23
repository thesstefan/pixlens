import io
import pathlib

import flask
import numpy as np
from PIL import Image
from werkzeug.datastructures import FileStorage

from pixlens.server.app import ml_models
from pixlens.server.config import Config
from pixlens.utils.utils import get_cache_dir
from pixlens.visualization import annotation

bp = flask.Blueprint(
    "pixlens",
    __name__,
    url_prefix="/pixlens",
)

config = Config()


def save_tmp_image(  # type: ignore[no-any-unimported]
    file: FileStorage,
) -> tuple[pathlib.Path, Image.Image]:
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # TODO: This is incredibly awkward - change models
    #       to take an input image instead of a path
    image_path = get_cache_dir() / "tmp.png"
    image.save(image_path)

    return image_path, image


def create_image_response(image: Image.Image) -> flask.Response:
    image_bytes = io.BytesIO()
    image.save(image_bytes, "png", quality=100)
    image_bytes.seek(0)

    return flask.send_file(image_bytes, mimetype="image/png")


@bp.route("/edit", methods=["POST"])
def edit() -> flask.Response:
    image_path, image = save_tmp_image(flask.request.files["image"])
    prompt = flask.request.form["prompt"]

    model = ml_models.get_edit_model(
        config.EDIT_MODEL,
        config.DEVICE,
    )
    edit_output = model.edit(prompt, str(image_path))

    image_path.unlink()

    return create_image_response(edit_output.image)


@bp.route("/detect_and_segment", methods=["POST"])
def detect_and_segment() -> flask.Response:
    image_path, image = save_tmp_image(flask.request.files["image"])
    prompt = flask.request.form["prompt"]

    model = ml_models.get_detect_segment_model(
        config.DETECT_SEGMENT_MODEL,
        config.DEVICE,
    )
    (
        segmentation_output,
        detection_output,
    ) = model.detect_and_segment(prompt, str(image_path))

    image_path.unlink()
    image_source = np.asarray(image)

    annotated_image = annotation.annotate_detection_output(
        image_source,
        detection_output,
    )
    masked_annotated_image = annotation.annotate_mask(
        segmentation_output.masks,
        annotated_image,
    )

    return create_image_response(masked_annotated_image)
