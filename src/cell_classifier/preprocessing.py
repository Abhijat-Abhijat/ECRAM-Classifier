"""
Image loading and preprocessing.

This module is the single source of truth for turning an image into model input.
Training (scripts/prepare_data.py), evaluation and the API all call it, so there
is no train/serve skew.

Why NEAREST resampling?
    The model was trained with Keras `flow_from_directory`, which loads images via
    `keras.utils.load_img(..., interpolation="nearest")`. Serving with a different
    filter (e.g. Pillow's default BICUBIC) changes what the model sees and measurably
    changes predictions - see reports/evaluation.md.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
from PIL import Image, ImageOps

from . import config

ImageSource = Union[str, Path, bytes, BinaryIO]


class ImageTooLargeError(ValueError):
    """Raised when an image exceeds config.MAX_IMAGE_PIXELS."""


def load_image(source: ImageSource, max_pixels: int | None = None) -> Image.Image:
    """Open an image from a path, bytes or file object and return it as RGB.

    Raises:
        ImageTooLargeError: if width*height exceeds ``max_pixels``.
        PIL.UnidentifiedImageError / OSError: if the data is not a readable image.
    """
    max_pixels = config.MAX_IMAGE_PIXELS if max_pixels is None else max_pixels
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)

    # We do our own pixel-count check below, so disable Pillow's warning/limit
    # (it would otherwise reject the legitimate gigapixel slides).
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(source)

    # Header is read lazily, so this check happens BEFORE the pixels are decoded.
    w, h = img.size
    if w * h > max_pixels:
        raise ImageTooLargeError(
            f"Image is {w}x{h} ({w * h / 1e6:.0f} MP); limit is {max_pixels / 1e6:.0f} MP."
        )

    # Multi-page TIFF: use the first page (full resolution), as Keras does.
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)

    # Phone photos carry an EXIF rotation flag - honour it.
    img = ImageOps.exif_transpose(img)

    # Avoid a full-size copy for already-RGB gigapixel images.
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def to_model_input(img: Image.Image) -> np.ndarray:
    """Resize (nearest) + scale to [0, 1]. Returns float32 array of shape (1, 224, 224, 3)."""
    resized = img.resize(config.IMG_SIZE, resample=Image.Resampling.NEAREST)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]


def preprocess(source: ImageSource, max_pixels: int | None = None) -> np.ndarray:
    """load_image + to_model_input in one call."""
    with load_image(source, max_pixels=max_pixels) as img:
        return to_model_input(img)
