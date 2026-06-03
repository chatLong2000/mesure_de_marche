"""Caméra factice générant des images synthétiques de balancier (mode test)."""

import math
import time

import numpy as np
import cv2


class DummyCamera:
    """Caméra factice générant des images synthétiques de balancier."""

    def __init__(self):
        self.connected = False
        self.hw_trigger = True  # Pas de filtrage dark-frame en mode dummy
        self.width = 800
        self.height = 600
        self._frame_count = 0
        self._angle = 0.0
        self._angle_step = 0.05  # Simule un léger écart de fréquence

    def connect(self, device_index=0):
        self.connected = True
        print("[CAM DUMMY] Caméra factice connectée.")
        return True

    def start_acquisition(self):
        pass

    def stop_acquisition(self):
        pass

    def capture_frame(self, timeout_us=5_000_000):
        """Génère une image synthétique simulant le balancier."""
        self._frame_count += 1
        self._angle += self._angle_step

        img = np.zeros((self.height, self.width), dtype=np.uint8)
        cx, cy = self.width // 2, self.height // 2

        # Simuler l'aiguille du balancier à différents angles
        angle_rad = self._angle
        length = min(cx, cy) - 50
        ex = int(cx + length * math.cos(angle_rad))
        ey = int(cy + length * math.sin(angle_rad))

        cv2.line(img, (cx, cy), (ex, ey), 200, 3)
        cv2.circle(img, (cx, cy), 10, 150, -1)

        # Ajouter du bruit
        noise = np.random.randint(0, 15, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)

        return img

    def capture_sequence(self, count, interval_ms=0):
        images = []
        for i in range(count):
            frame = self.capture_frame()
            if frame is not None:
                images.append(frame)
            if interval_ms > 0 and i < count - 1:
                time.sleep(interval_ms / 1000.0)
        return images

    def set_exposure(self, exposure_us):
        pass

    def disconnect(self):
        self.connected = False
