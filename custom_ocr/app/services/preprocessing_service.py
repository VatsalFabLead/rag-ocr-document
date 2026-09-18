import math
from pathlib import Path
from typing import Dict, Tuple, Optional
import cv2
import numpy as np
from app.api.schemas.ocr_schema import PreprocessingOptions, PreprocessingStepImage
from app.utils.image_utils import save_image

class PreprocessingService:
    """
    Step 2 — Image Preprocessing Pipeline:
    Original Image
          ↓
       Resize
          ↓
      Grayscale
          ↓
    Noise Removal
          ↓
    Contrast Enhancement
          ↓
       Deskew
          ↓
     Thresholding
          ↓
     Clean Image
    """

    def preprocess(
        self,
        image: np.ndarray,
        output_dir: Path,
        page_num: int = 1,
        options: Optional[PreprocessingOptions] = None
    ) -> Tuple[np.ndarray, float, list]:
        """
        Executes the full preprocessing pipeline.
        Returns: (clean_image, skew_angle, steps_metadata)
        """
        if options is None:
            options = PreprocessingOptions()

        steps_metadata = []
        current = image.copy()
        skew_angle = 0.0

        # Step 0: Original Image
        step0_path = output_dir / f"page_{page_num}_0_original.png"
        save_image(current, step0_path)
        steps_metadata.append(PreprocessingStepImage(
            step_key="original",
            title="Original Image",
            description="The original uploaded document image without alterations.",
            image_url=f"/outputs/{step0_path.name}"
        ))

        # Step 1: Resize (Normalize DPI / Resolution)
        if options.enable_resize:
            current = self.resize_image(current, max_dimension=2400)
            step1_path = output_dir / f"page_{page_num}_1_resized.png"
            save_image(current, step1_path)
            steps_metadata.append(PreprocessingStepImage(
                step_key="resize",
                title="1. Resizing & DPI Normalization",
                description="Rescaled to standard OCR resolution while preserving aspect ratio.",
                image_url=f"/outputs/{step1_path.name}"
            ))

        # Step 2: Grayscale
        if options.enable_grayscale and len(current.shape) == 3:
            current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            step2_path = output_dir / f"page_{page_num}_2_grayscale.png"
            save_image(current, step2_path)
            steps_metadata.append(PreprocessingStepImage(
                step_key="grayscale",
                title="2. Grayscale Conversion",
                description="Color channels collapsed to single intensity channel to eliminate color noise.",
                image_url=f"/outputs/{step2_path.name}"
            ))

        # Step 3: Noise Removal (Bilateral filter to smooth noise while keeping text edges sharp)
        if options.enable_denoise:
            current = self.remove_noise(current)
            step3_path = output_dir / f"page_{page_num}_3_denoised.png"
            save_image(current, step3_path)
            steps_metadata.append(PreprocessingStepImage(
                step_key="denoise",
                title="3. Noise Removal",
                description="Applied bilateral smoothing to eliminate scanner grain and background speckles.",
                image_url=f"/outputs/{step3_path.name}"
            ))

        # Step 4: Contrast Enhancement (CLAHE)
        if options.enable_contrast_enhancement:
            current = self.enhance_contrast(current)
            step4_path = output_dir / f"page_{page_num}_4_clahe.png"
            save_image(current, step4_path)
            steps_metadata.append(PreprocessingStepImage(
                step_key="contrast",
                title="4. Contrast Enhancement (CLAHE)",
                description="Contrast-Limited Adaptive Histogram Equalization to normalize uneven lighting/shadows.",
                image_url=f"/outputs/{step4_path.name}"
            ))

        # Step 5: Deskew
        if options.enable_deskew:
            current, skew_angle = self.deskew(current)
            step5_path = output_dir / f"page_{page_num}_5_deskewed.png"
            save_image(current, step5_path)
            steps_metadata.append(PreprocessingStepImage(
                step_key="deskew",
                title=f"5. Deskewing ({round(skew_angle, 2)}°)",
                description=f"Detected skew angle of {round(skew_angle, 2)}° and rotated document to true horizontal.",
                image_url=f"/outputs/{step5_path.name}"
            ))

        # Step 6: Thresholding (Binarization)
        if options.enable_thresholding:
            current = self.threshold(current, method=options.threshold_method)
            step6_path = output_dir / f"page_{page_num}_6_thresholded.png"
            save_image(current, step6_path)
            steps_metadata.append(PreprocessingStepImage(
                step_key="thresholding",
                title="6. Adaptive Thresholding",
                description="High-contrast black-on-white binarization separating glyph foreground from background.",
                image_url=f"/outputs/{step6_path.name}"
            ))

        # Step 7: Clean Image (Morphological opening to eliminate isolated pixel dust)
        current = self.clean_binary(current)
        step7_path = output_dir / f"page_{page_num}_7_clean.png"
        save_image(current, step7_path)
        steps_metadata.append(PreprocessingStepImage(
            step_key="clean",
            title="7. Clean Final Image",
            description="Morphological artifact removal leaving pristine text glyphs ready for OCR engine.",
            image_url=f"/outputs/{step7_path.name}"
        ))

        return current, skew_angle, steps_metadata

    def resize_image(self, image: np.ndarray, max_dimension: int = 2400) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) <= max_dimension:
            return image
        scale = max_dimension / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def remove_noise(self, gray: np.ndarray) -> np.ndarray:
        # Bilateral filter smooths flat regions while preserving sharp edges
        return cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    def enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detects the skew angle of text lines and rotates the image back to 0°."""
        # Detect edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        # Use probabilistic Hough lines to find dominant line angles
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=20)

        angles = []
        if lines is not None:
            for line in lines:
                coords = line.flatten()
                if len(coords) >= 4:
                    x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    # Only consider slight tilt angles typical of scanned documents (-45 to 45 deg)
                    if abs(angle) < 45:
                        angles.append(angle)

        if not angles:
            # Fallback: Minimum area bounding box around text threshold
            coords = np.column_stack(np.where(image > 0))
            if coords.size > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                if abs(angle) < 45:
                    angles.append(angle)

        skew_angle = float(np.median(angles)) if angles else 0.0

        if abs(skew_angle) > 0.3:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
            deskewed = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return deskewed, skew_angle

        return image, 0.0

    def threshold(self, gray: np.ndarray, method: str = "adaptive") -> np.ndarray:
        """Binarizes the image using adaptive Gaussian or Otsu thresholding."""
        if method == "otsu":
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return bin_img
        else:
            # Adaptive Gaussian
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11
            )

    def clean_binary(self, binary: np.ndarray) -> np.ndarray:
        """Removes salt-and-pepper noise via small morphological opening."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return opened
