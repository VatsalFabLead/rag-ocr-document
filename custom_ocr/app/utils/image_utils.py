import base64
from pathlib import Path
from typing import Tuple, Optional, Union
import cv2
import numpy as np
from PIL import Image

def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Safely loads an image from a path (supports Windows unicode paths via cv2.imdecode).
    Returns BGR numpy array.
    """
    image_path = str(image_path)
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    with open(image_path, "rb") as f:
        bytes_data = bytearray(f.read())
        numpy_array = np.asarray(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image from {image_path}")
        return img

def save_image(image: np.ndarray, output_path: Union[str, Path]) -> str:
    """
    Safely writes an image to disk (supports Windows paths via cv2.imencode).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower() or ".png"
    success, encoded_img = cv2.imencode(ext, image)
    if not success:
        raise ValueError(f"Failed to encode image to {ext}")
    with open(output_path, "wb") as f:
        f.write(encoded_img.tobytes())
    return str(output_path)

def image_to_base64(image: np.ndarray, ext: str = ".png") -> str:
    """Encodes a numpy image to a base64 data URI string."""
    success, buffer = cv2.imencode(ext, image)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64_str}"

def draw_bounding_boxes(
    image: np.ndarray,
    boxes_with_labels: list,
    color: Tuple[int, int, int] = (0, 180, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Draws bounding boxes and labels onto a copy of the image.
    boxes_with_labels: list of dicts with keys: 'x', 'y', 'width', 'height', and optional 'text', 'confidence'.
    """
    canvas = image.copy()
    for item in boxes_with_labels:
        x, y, w, h = item['x'], item['y'], item['width'], item['height']
        text = item.get('text', '')
        conf = item.get('confidence')

        # Box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)

        # Label background
        if text:
            label = f"{text[:25]}"
            if conf is not None:
                label += f" ({int(conf * 100)}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
            cv2.rectangle(canvas, (x, max(0, y - th - 6)), (x + tw + 4, max(th + 6, y)), color, -1)
            cv2.putText(canvas, label, (x + 2, max(th + 2, y - 4)), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas
