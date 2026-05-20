import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from preprocessing import preprocess_image_bytes
import logging
from exceptions import PreprocessingError, ModelExecutionError

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)
from model_utils import ensure_model_file, get_model_path, get_model_url, get_model_sha256
from exceptions import PreprocessingError, ModelExecutionError

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = get_model_path()
MODEL_URL = get_model_url()
MODEL_SHA256 = get_model_sha256()
_model = None


def load_deepfake_model():
    global _model
    if _model is not None:
        return _model

    model_file_path = ensure_model_file(
        model_path=MODEL_PATH,
        model_url=MODEL_URL,
        model_sha256=MODEL_SHA256,
        download_if_missing=True,
    )
    _model = load_model(model_file_path)
    return _model

def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image nahi mili: {image_path}")

    with open(image_path, "rb") as file_handle:
        image_bytes = file_handle.read()

    return preprocess_image_bytes(image_bytes)

def predict_image(image_path):
    try:
        model = load_deepfake_model()
        def preprocess_image(image_path):
            try:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image nahi mili: {image_path}")

                # Read as bytes and use shared cached preprocessing for uploads
                with open(image_path, "rb") as file_handle:
                    image_bytes = file_handle.read()

                return preprocess_image_bytes(image_bytes)
            except Exception as e:
                logger.error(f"Image preprocessing failed for {image_path}: {e}", exc_info=True)
                raise PreprocessingError(f"Failed to preprocess image: {str(e)}") from e


        def predict_image(image_path):
            try:
                model = load_deepfake_model()
                image = preprocess_image(image_path)

                try:
                    prediction = model.predict(image, verbose=0)
                    print(f"Raw prediction: {prediction}")
                    class_label = np.argmax(prediction, axis=1)[0]
                    confidence = float(np.max(prediction)) * 100
                    # Class 1 = Fake, Class 0 = Real (dataset mapping)
                    label = "Fake" if class_label == 1 else "Real"
                    print(f"Prediction  : {label}")
                    print(f"Confidence  : {confidence:.1f}%")
                    return label
                except Exception as e:
                    logger.error(f"Model prediction failed: {e}", exc_info=True)
                    raise ModelExecutionError(f"Model prediction failed: {str(e)}") from e

            except PreprocessingError as e:
                print(f"Error: Could not process image {image_path}. Check logs for details.")
                sys.exit(1)
            except ModelExecutionError as e:
                print(f"Error: Model execution failed. Check logs for details.")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Unexpected error in prediction pipeline: {e}", exc_info=True)
                print(f"Error: An unexpected error occurred. Check logs for details.")
                sys.exit(1)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in prediction pipeline: {e}", exc_info=True)
        print(f"Error: An unexpected error occurred. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python predict.py <image_path>")

    predict_image(sys.argv[1])
