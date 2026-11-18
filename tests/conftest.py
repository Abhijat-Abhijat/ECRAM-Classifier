import io
import os

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def make_image_bytes(size=(320, 240), mode="RGB", fmt="PNG", color=None) -> bytes:
    rng = np.random.default_rng(0)
    channels = {"RGB": 3, "RGBA": 4, "L": 1}[mode]
    arr = rng.integers(0, 256, (size[1], size[0], channels), dtype=np.uint8)
    if color is not None:
        arr[:] = color
    img = Image.fromarray(arr.squeeze() if channels == 1 else arr, mode)
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


@pytest.fixture
def png_bytes():
    return make_image_bytes()
