import tensorflow as tf
import numpy as np
import cv2

from config import IMAGE_SIZE


def find_last_conv_layer(model) -> str:
    """Walk a Keras model's layers to find the name of the last Conv2D layer.

    Works with both Sequential and Functional API models as well as
    models wrapped inside another model (e.g. ``model.layers[0]``).

    Parameters
    ----------
    model:
        A ``tf.keras.Model`` (or sub-model / backbone).

    Returns
    -------
    str
        The ``layer.name`` of the last convolutional layer found.

    Raises
    ------
    ValueError
        If no Conv2D layer is found in the model.
    """
    last_conv_name: str | None = None

    for layer in model.layers:
        # Check the layer itself
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name
        # For nested models (Sequential / Functional wrapped inside another),
        # recurse into their layers as well.
        elif hasattr(layer, "layers"):
            for sub_layer in layer.layers:
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    last_conv_name = sub_layer.name

    if last_conv_name is None:
        raise ValueError(
            "No Conv2D layer found in the model. "
            "Grad-CAM requires at least one convolutional layer."
        )

    return last_conv_name


def make_gradcam_heatmap(img_array, model, last_conv_layer):
    """Generate a Grad-CAM heatmap for the predicted class.

    Parameters
    ----------
    img_array:
        Preprocessed image tensor of shape ``(1, H, W, 3)``.
    model:
        A ``tf.keras.Model`` used for inference.
    last_conv_layer:
        Name of the convolutional layer whose activations are used.

    Returns
    -------
    np.ndarray
        2-D heatmap with values in ``[0, 1]``.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [
            model.get_layer(last_conv_layer).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay a Grad-CAM *heatmap* onto the original *image*.

    Parameters
    ----------
    image:
        Original BGR image (numpy array).
    heatmap:
        2-D array from :func:`make_gradcam_heatmap`.
    alpha:
        Blending weight for the heatmap overlay.

    Returns
    -------
    np.ndarray
        BGR image with the heatmap blended on top.
    """
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(
        image,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return superimposed_img