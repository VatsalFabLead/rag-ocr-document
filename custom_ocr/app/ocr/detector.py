import cv2
import numpy as np
from typing import List, Tuple
from app.models.ocr_result import BoundingBox

class CustomDetector:
    """
    Custom Computer Vision Text Detection Engine.
    Uses morphological operations, Sobel gradients, and contour analysis
    to detect text regions, words, and lines from document images.
    """

    def __init__(self, min_area: int = 40, max_area_ratio: float = 0.95):
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio

    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Detects text bounding boxes in an image.
        Returns a list of BoundingBox objects in natural reading order.
        """
        img_h, img_w = image.shape[:2]

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. Morphological Gradient / Sobel to emphasize text boundaries
        grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
        gradient = cv2.subtract(grad_x, grad_y)
        gradient = cv2.convertScaleAbs(gradient)

        # 2. Blur to remove high frequency noise
        blurred = cv2.GaussianBlur(gradient, (5, 5), 0)

        # 3. Otsu Thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Morphological Close with a horizontal rectangular kernel
        # Keep vertical kernel height small (2-3px) so adjacent lines never merge
        kernel_w = max(9, min(20, int(img_w * 0.012)))
        kernel_h = 2
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 5. Find contours of candidate text blocks
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_boxes: List[Tuple[int, int, int, int]] = []
        max_area = (img_w * img_h) * self.max_area_ratio

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            # Filter out tiny noise and full-image enclosing contours
            if area < self.min_area or area > max_area:
                continue

            # Minimum height and width for readable text
            if h < 8 or w < 8:
                continue

            # Add padding
            pad_x = int(w * 0.02)
            pad_y = int(h * 0.04)
            x_pad = max(0, x - pad_x)
            y_pad = max(0, y - pad_y)
            w_pad = min(img_w - x_pad, w + 2 * pad_x)
            h_pad = min(img_h - y_pad, h + 2 * pad_y)

            # Split multi-line merged blocks via horizontal projection profile
            if h_pad > 40:
                patch = gray[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                splits = self._split_multiline_box(patch, (x_pad, y_pad, w_pad, h_pad))
                raw_boxes.extend(splits)
            else:
                raw_boxes.append((x_pad, y_pad, w_pad, h_pad))

        # 6. Sort boxes in natural reading order: Top-to-bottom, then Left-to-right
        sorted_boxes = self._sort_reading_order(raw_boxes, line_threshold=15)

        # Convert to BoundingBox objects with normalized coordinates
        bbox_objects = [
            BoundingBox.from_coords(x, y, w, h, img_w, img_h)
            for (x, y, w, h) in sorted_boxes
        ]

        return bbox_objects

    def _split_multiline_box(
        self, gray_patch: np.ndarray, box: Tuple[int, int, int, int], min_line_h: int = 10
    ) -> List[Tuple[int, int, int, int]]:
        """Segments a tall multi-line bounding box into individual single-line boxes."""
        x, y, w, h = box
        if gray_patch is None or gray_patch.size == 0 or h <= 35:
            return [box]

        try:
            _, bin_p = cv2.threshold(gray_patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            proj = np.sum(bin_p, axis=1)
            max_p = np.max(proj)
            if max_p == 0:
                return [box]

            threshold = max_p * 0.12
            splits = []
            start_y = 0
            in_text = False

            for r in range(h):
                if proj[r] > threshold and not in_text:
                    in_text = True
                    start_y = r
                elif proj[r] <= threshold and in_text:
                    in_text = False
                    if r - start_y >= min_line_h:
                        splits.append((x, y + start_y, w, r - start_y))

            if in_text and (h - start_y) >= min_line_h:
                splits.append((x, y + start_y, w, h - start_y))

            return splits if splits else [box]
        except Exception:
            return [box]

    def _sort_reading_order(
        self, boxes: List[Tuple[int, int, int, int]], line_threshold: int = 15
    ) -> List[Tuple[int, int, int, int]]:
        """Sorts bounding boxes into top-to-bottom, left-to-right reading order."""
        if not boxes:
            return []

        # Sort primarily by Y coordinate
        boxes_by_y = sorted(boxes, key=lambda b: b[1])

        lines: List[List[Tuple[int, int, int, int]]] = []
        current_line: List[Tuple[int, int, int, int]] = [boxes_by_y[0]]

        for box in boxes_by_y[1:]:
            prev_box = current_line[-1]
            # Check if current box is on roughly the same vertical line
            if abs(box[1] - prev_box[1]) <= line_threshold or abs((box[1] + box[3]) - (prev_box[1] + prev_box[3])) <= line_threshold:
                current_line.append(box)
            else:
                lines.append(sorted(current_line, key=lambda b: b[0]))
                current_line = [box]

        if current_line:
            lines.append(sorted(current_line, key=lambda b: b[0]))

        # Flatten sorted lines
        sorted_result = [box for line in lines for box in line]
        return sorted_result
