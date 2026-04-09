"""
Classificateur par analyse de pics du profil perpendiculaire.

Classes prédites :
  0 → 1 pic (balancier immobile côté gauche ou droit)
  1 → 2 pics (faible → fort)
  2 → 2 pics (fort → faible)
  3 → 3 pics (le "saut" — passage par l'équilibre)
 -1 → non classifiable
"""

import cv2
import numpy as np
from scipy.signal import find_peaks


class SignalClassifier:
    """
    Classification par analyse de pics du profil perpendiculaire.
    Classes :
      0 → 1 pic (balancier immobile côté gauche ou droit)
      1 → 2 pics (faible → fort)
      2 → 2 pics (fort → faible)
      3 → 3 pics (le "saut" — passage par l'équilibre)
     -1 → non classifiable
    """

    def __init__(self, height_threshold: int = 5, subtract_threshold: int = 15):
        self.height_threshold = height_threshold
        self.subtract_threshold = subtract_threshold
        self.last_processed = None  # Dernière image traitée (subtract_median)

    def crop_to_square(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        if h == w:
            return img
        if h > w:
            start = (h - w) // 2
            return img[start:start+w, :]
        else:
            start = (w - h) // 2
            return img[:, start:start+h]

    def subtract_median(self, images: list) -> np.ndarray:
        assert len(images) == 4, "Il faut exactement 4 images"
        median_image = np.median(np.stack(images[:3]), axis=0)
        result = images[3].astype(np.float32) - median_image
        result = np.clip(result, 0, 255).astype(np.uint8)
        result = self.crop_to_square(result)
        result[result < self.subtract_threshold] = 0
        return result

    def extract_perpendicular_vector(self, img, alpha=0.99, nb_pixels=15):
        _, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        ys, xs = np.where(img_thresh > 0)
        if len(xs) == 0:
            return np.array([]), False

        coords = np.column_stack((xs, ys))
        h, w = img.shape
        center = np.array([w // 2, h // 2])
        dists = np.linalg.norm(coords - center, axis=1)
        tip = coords[np.argmax(dists)]

        vec = tip - center
        norm = np.linalg.norm(vec)
        if norm < 1:
            return np.array([]), False

        new_base = center + alpha * vec
        d_perp = np.array([-vec[1], vec[0]], dtype=float)
        d_perp = d_perp / np.linalg.norm(d_perp)

        line_points = []
        for i in range(-nb_pixels, nb_pixels + 1):
            pt = new_base + i * d_perp
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                line_points.append(img[y, x])
            else:
                return np.array([]), False

        if np.all(np.array(line_points) == 0):
            return np.array([]), False

        return np.array(line_points), True

    def classify_signal(self, sequence: np.ndarray) -> int:
        peaks, properties = find_peaks(sequence, height=self.height_threshold, distance=3)
        heights = properties.get("peak_heights", [])

        if len(peaks) == 1:
            return 0
        elif len(peaks) == 2:
            return 1 if heights[0] < heights[1] else 2
        elif len(peaks) == 3:
            return 3
        else:
            return -1

    def predict(self, images: list) -> int:
        if len(images) != 4:
            raise ValueError("Il faut exactement 4 images.")
        processed = self.subtract_median(images)
        self.last_processed = processed
        profile, valid = self.extract_perpendicular_vector(processed)
        if not valid:
            return -1
        return self.classify_signal(profile)
