import io

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

from cell_classifier import config
from cell_classifier.preprocessing import ImageTooLargeError, load_image, preprocess, to_model_input
from conftest import make_image_bytes


@pytest.mark.parametrize("mode,fmt", [("RGB", "PNG"), ("RGBA", "PNG"), ("L", "PNG"), ("RGB", "JPEG"), ("RGB", "TIFF")])
def test_output_contract(mode, fmt):
    x = preprocess(make_image_bytes(mode=mode, fmt=fmt))
    assert x.shape == (1, *config.IMG_SIZE, 3)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0


def test_rgba_alpha_is_dropped_not_blended():
    # Keras load_img does img.convert("RGB"), which discards alpha. We must match.
    raw = make_image_bytes(mode="RGBA", color=(200, 100, 50, 0))
    x = preprocess(raw)
    np.testing.assert_allclose(x[0, 0, 0], np.array([200, 100, 50]) / 255.0, atol=1e-6)


def test_matches_keras_training_loader(tmp_path):
    """The serving pipeline must be bit-identical to what flow_from_directory fed the model."""
    keras_utils = pytest.importorskip("tensorflow").keras.utils
    p = tmp_path / "img.png"
    p.write_bytes(make_image_bytes(size=(997, 613), mode="RGBA"))
    expected = np.asarray(keras_utils.load_img(p, target_size=config.IMG_SIZE, interpolation="nearest"),
                          dtype=np.float32) / 255.0
    np.testing.assert_array_equal(preprocess(p)[0], expected)


def test_pixel_limit_checked_before_decode():
    with pytest.raises(ImageTooLargeError):
        load_image(make_image_bytes(size=(100, 100)), max_pixels=100 * 100 - 1)


def test_garbage_bytes_rejected():
    with pytest.raises(UnidentifiedImageError):
        preprocess(b"definitely not an image")


def test_exif_orientation_applied():
    img = Image.new("RGB", (40, 20), "red")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° CW
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    assert load_image(buf.getvalue()).size == (20, 40)


def test_to_model_input_uses_nearest():
    # A 2-colour checkerboard stays 2-colour under NEAREST (bicubic would create greys).
    board = np.indices((448, 448)).sum(axis=0) % 2 * 255
    img = Image.fromarray(np.stack([board] * 3, -1).astype(np.uint8))
    assert set(np.unique(to_model_input(img))) <= {0.0, 1.0}
