import io

import flask
import numpy as np
from PIL import Image

from pixlens.server.app import ml_models
from pixlens.utils import get_cache_dir
from pixlens.visualization import annotation

bp = flask.Blueprint(
    "pixlens",
    __name__,
    url_prefix="/pixlens",
)


@bp.route("/detect_and_segment", methods=["POST"])
def detect_and_segment() -> flask.Response:
    image_file = flask.request.files["image"]
    prompt = flask.request.form["prompt"]

    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # TODO(thesstefan): This is incredibly awkward - change models
    #                   to take an input image instead of a path
    image_path = get_cache_dir() / "tmp.png"
    image.save(image_path)

    model = ml_models.get_detect_and_segment_model()
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

    mask_annotated_bytes = io.BytesIO()
    masked_annotated_image.save(mask_annotated_bytes, "png", quality=100)
    mask_annotated_bytes.seek(0)

    return flask.send_file(mask_annotated_bytes, mimetype="image/png")
