"""
PixelTruth — unified inference pipeline.

This module is the **single source of truth** for image preprocessing and
deepfake prediction.  Both the Streamlit dashboard (``app.py``) and the CLI
import from here, ensuring identical behaviour regardless of the entry-point.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from config import (
    IMAGE_SIZE,
    LOG_FORMAT,
    SUPPORTED_EXTENSIONS,
)
from preprocessing import (
    preprocess_image_array,
    preprocess_image_bytes,
    decode_image_bytes,
)
from exceptions import PreprocessingError, ModelExecutionError
from model_utils import ensure_model_file, get_model_path, get_model_url, get_model_sha256

import logging

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading (lazy singleton)
# ---------------------------------------------------------------------------

_model = None


def load_deepfake_model(model_path: str | None = None):
    """Load (or return cached) deepfake-detection model.

    Parameters
    ----------
    model_path:
        Override the model file location.  When *None*, the path is resolved
        via ``model_utils.get_model_path()`` / ``PIXELTRUTH_MODEL_PATH``.
    """
    global _model
    if _model is not None:
        return _model

    resolved_path = model_path or get_model_path()
    model_file = ensure_model_file(
        model_path=resolved_path,
        model_url=get_model_url(),
        model_sha256=get_model_sha256(),
        download_if_missing=True,
    )
    _model = load_model(model_file)
    return _model


# ---------------------------------------------------------------------------
# Unified preprocessing — accepts file paths, numpy arrays, or raw bytes
# ---------------------------------------------------------------------------


def preprocess_image(image_input) -> np.ndarray:
    """Preprocess an image for model inference.

    Accepts multiple input types so that every caller (CLI, Streamlit, tests)
    can use a single function:

    * **str / Path** — filesystem path; the file is read and decoded.
    * **bytes** — raw image bytes (e.g. from ``UploadedFile.read()``).
    * **np.ndarray** — a BGR image already loaded into memory.

    Parameters
    ----------
    image_input:
        The image to preprocess. See above for accepted types.

    Returns
    -------
    np.ndarray
        Shape ``(1, H, W, 3)`` with values in ``[0, 1]``, channels in RGB
        order — ready to be passed directly to ``model.predict()``.

    Raises
    ------
    FileNotFoundError
        When a path string is provided but the file does not exist.
    ValueError
        When a path has an unsupported extension, or bytes cannot be decoded.
    PreprocessingError
        When preprocessing fails for any other reason.
    TypeError
        When *image_input* is not a supported type.
    """
    # --- str / Path  →  read bytes, then preprocess ---
    if isinstance(image_input, (str, Path)):
        image_path = str(image_input)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        ext = os.path.splitext(image_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        try:
            with open(image_path, "rb") as fh:
                image_bytes = fh.read()
            return preprocess_image_bytes(image_bytes)
        except Exception as e:
            logger.error(f"Image preprocessing failed for {image_path}: {e}", exc_info=True)
            raise PreprocessingError(f"Failed to preprocess image: {str(e)}") from e

    # --- bytes  →  decode + preprocess ---
    if isinstance(image_input, bytes):
        try:
            return preprocess_image_bytes(image_input)
        except Exception as e:
            logger.error(f"Byte-based preprocessing failed: {e}", exc_info=True)
            raise PreprocessingError(f"Failed to preprocess image bytes: {str(e)}") from e

    # --- numpy array  →  preprocess directly ---
    if isinstance(image_input, np.ndarray):
        try:
            return preprocess_image_array(image_input)
        except Exception as e:
            logger.error(f"Array-based preprocessing failed: {e}", exc_info=True)
            raise PreprocessingError(f"Failed to preprocess image array: {str(e)}") from e

    raise TypeError(
        f"Unsupported image input type: {type(image_input).__name__}. "
        "Expected a file path (str/Path), raw bytes, or a numpy ndarray."
    )


# ---------------------------------------------------------------------------
# Unified prediction
# ---------------------------------------------------------------------------


def predict_image(image_input, *, model_path: str | None = None) -> dict:
    """Run deepfake detection on a single image.

    Parameters
    ----------
    image_input:
        Any input accepted by :func:`preprocess_image` — file path, bytes,
        or numpy array.
    model_path:
        Optional override for the model file location.

    Returns
    -------
    dict
        ``{"label": "Real"|"Fake", "confidence": float, "raw": list[float],
          "processed_image": np.ndarray}``

        * ``confidence`` is a **float in [0, 1]** (NOT a percentage).
        * ``processed_image`` is the preprocessed tensor used for inference.
        * For CLI callers the dict also includes ``"image": str`` when a path
          was provided.

    Raises
    ------
    FileNotFoundError / ValueError / PreprocessingError / ModelExecutionError
        See :func:`preprocess_image` and model loading docs.
    """
    processed = preprocess_image(image_input)

    try:
        model = load_deepfake_model(model_path)
        prediction = model.predict(processed, verbose=0)
    except (PreprocessingError, FileNotFoundError, ValueError, TypeError):
        raise
    except Exception as e:
        logger.error(f"Model prediction failed: {e}", exc_info=True)
        raise ModelExecutionError(f"Model prediction failed: {str(e)}") from e

    class_index = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction))
    # Dataset mapping: class 0 = Real, class 1 = Fake
    label = "Fake" if class_index == 1 else "Real"

    result: dict = {
        "label": label,
        "confidence": confidence,
        "raw": prediction[0].tolist(),
        "processed_image": processed,
    }

    # Include path metadata when the input was a file path
    if isinstance(image_input, (str, Path)):
        result["image"] = str(image_input)

    return result


# ---------------------------------------------------------------------------
# Convenience wrappers (backward-compat for app.py)
# ---------------------------------------------------------------------------


def predict_image_tuple(image_input):
    """Thin wrapper returning ``(label, confidence, processed_image)``.

    Used by ``app.py`` which was originally built around a tuple return value.
    If no model is loaded, returns ``(None, None, None)``.
    """
    try:
        result = predict_image(image_input)
    except Exception:
        # When model loading fails altogether, mirror the old None-tuple.
        return None, None, None
    return result["label"], result["confidence"], result["processed_image"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description=(
            "PixelTruth — deepfake image detector.\n"
            "Classifies one or more images as Real or Fake."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python predict.py photo.jpg\n"
            "  python predict.py img1.jpg img2.png --json\n"
            "  python predict.py --model /weights/model.h5 photo.jpg\n\n"
            "Environment variables:\n"
            "  PIXELTRUTH_MODEL_PATH   path to model file\n"
            "  PIXELTRUTH_MODEL_URL    URL to download model if missing\n"
            "  PIXELTRUTH_MODEL_SHA256 expected SHA-256 of the model file"
        ),
    )
    parser.add_argument(
        "images",
        metavar="IMAGE",
        nargs="+",
        help="path(s) to image file(s) to classify",
    )
    parser.add_argument(
        "--model",
        metavar="PATH",
        default=None,
        help=(
            "path to the .h5 model file "
            "(default: $PIXELTRUTH_MODEL_PATH or 'deepfake_detection_model.h5')"
        ),
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="print results as JSON (useful for scripting)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress informational messages; only print results",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns 0 on success, 1 if any image fails."""
    parser = build_parser()
    args = parser.parse_args(argv)

    results = []
    exit_code = 0

    for image_path in args.images:
        try:
            result = predict_image(image_path, model_path=args.model)
            # Remove the numpy array before serialising for the CLI
            cli_result = {k: v for k, v in result.items() if k != "processed_image"}
            # Convert confidence to percentage for CLI output
            cli_result["confidence"] = round(cli_result["confidence"] * 100, 1)
            results.append(cli_result)
        except (FileNotFoundError, ValueError, PreprocessingError, ModelExecutionError) as exc:
            # Non-fatal: report the error and continue with remaining images.
            error_result = {
                "image": image_path,
                "error": str(exc),
            }
            results.append(error_result)
            exit_code = 1
            if not args.quiet:
                print(f"[ERROR] {exc}", file=sys.stderr)

    if args.output_json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))
    else:
        for result in results:
            if "error" in result:
                continue  # already printed to stderr above
            if not args.quiet:
                print(f"\nImage      : {result['image']}")
                print(f"Raw output : {result['raw']}")
            print(f"Prediction : {result['label']}")
            print(f"Confidence : {result['confidence']:.1f}%")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())