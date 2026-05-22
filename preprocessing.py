from functools import lru_cache

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

from config import IMAGE_SIZE

MIN_IMAGE_DIM = 10

def validate_image_dimensions(image: np.ndarray) -> None:
    h, w = image.shape[:2]
    if h < MIN_IMAGE_DIM or w < MIN_IMAGE_DIM:
        raise ValueError(
            f"Image too small ({w}x{h} px). "
            f"Minimum is {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM} px."
        )


def preprocess_image_array(image: np.ndarray) -> np.ndarray:
    validate_image_dimensions(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image


@lru_cache(maxsize=32)
def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes into a BGR numpy array.

    Raises
    ------
    ValueError
        When the bytes cannot be decoded into a valid image.
    """
    file_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            "The uploaded file appears to be corrupted or is not a valid image."
        )
    return image


@lru_cache(maxsize=32)
def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode *and* preprocess raw image bytes in one shot."""
    return preprocess_image_array(decode_image_bytes(image_bytes))