"""Pilotage de la caméra SVS EXO273CGE via Aravis (GObject Introspection)."""

import os
import time
import math
from typing import Optional

import cv2
import numpy as np

os.environ.setdefault("GI_TYPELIB_PATH", "/opt/homebrew/lib/girepository-1.0")

try:
    import gi
    gi.require_version("Aravis", "0.8")
    from gi.repository import Aravis
    ARAVIS_AVAILABLE = True
except (ImportError, ValueError):
    Aravis = None
    ARAVIS_AVAILABLE = False


class AravisCamera:
    """Pilotage de la caméra SVS EXO273CGE via Aravis GI."""

    DARK_FRAME_THRESHOLD = 30  # Seuil sur le percentile 99 pour détecter les frames illuminées

    def __init__(self, exposure_us: int = 10000, gain_db: float = 0.0):
        if not ARAVIS_AVAILABLE:
            raise RuntimeError("Aravis non disponible — installez gobject-introspection + Aravis.")
        self.camera = None
        self.stream = None
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.connected = False
        self.hw_trigger = False  # True si le trigger matériel est actif
        self.width = 0
        self.height = 0

    def connect(self, device_index: int = 0) -> bool:
        """Découvre et connecte la première caméra GigE."""
        Aravis.update_device_list()
        n = Aravis.get_n_devices()
        if n == 0:
            raise RuntimeError("Aucune caméra détectée sur le réseau.")

        dev_id = Aravis.get_device_id(device_index)
        self.camera = Aravis.Camera.new(dev_id)

        model = self.camera.get_model_name()
        serial_num = self.camera.get_device_serial_number()
        print(f"[CAM] Connecté: {model} (S/N {serial_num})")

        # Vérifier le contrôle exclusif (GigE)
        dev = self.camera.get_device()
        if isinstance(dev, Aravis.GvDevice) and not dev.is_controller():
            raise RuntimeError("Impossible d'obtenir le contrôle exclusif de la caméra.")

        # Configuration
        dev = self.camera.get_device()

        # Exposure
        try:
            dev.set_string_feature_value("ExposureAuto", "Off")
        except Exception as e:
            print(f"[CAM WARN] ExposureAuto: {e}")
        # try:
        #     self.camera.set_exposure_time(self.exposure_us)
        # except Exception as e:
        #     print(f"[CAM WARN] ExposureTime: {e}")
        try:
            dev.set_string_feature_value("ExposureMode", "TriggerWidth")
            # dev.set_string_feature_value("ExposureMode", "Timed")
        except Exception as e:
            print(f"[CAM WARN] ExposureMode: {e}")

        try:
            self.camera.set_gain(self.gain_db)
        except Exception as e:
            print(f"[CAM WARN] Gain: {e}")

        # Activer le trigger matériel (signal trig.expo du Pico)
        dev = self.camera.get_device()
        trigger_ok = False
        # Essayer plusieurs sources de trigger selon le modèle de caméra
        for src in ["Line1", "Line0", "Line2"]:
            try:
                dev.set_string_feature_value("TriggerMode", "On")
                dev.set_string_feature_value("TriggerSource", src)
                try:
                    dev.set_string_feature_value("TriggerActivation", "RisingEdge")
                except Exception:
                    pass  # Certaines caméras n'exposent pas TriggerActivation
                self.hw_trigger = True
                trigger_ok = True
                print(f"[CAM] Trigger matériel activé ({src})")
                break
            except Exception:
                # Remettre TriggerMode Off si la source n'est pas valide
                try:
                    dev.set_string_feature_value("TriggerMode", "Off")
                except Exception:
                    pass
                continue
        if not trigger_ok:
            # Lister les sources disponibles pour le debug
            try:
                node = dev.get_feature("TriggerSource")
                entries = node.get_childs() if node else []
                names = [e.get_name() for e in entries] if entries else []
                print(f"[CAM WARN] Trigger matériel non disponible. Sources connues: {names}")
            except Exception:
                print("[CAM WARN] Trigger matériel non disponible (aucune source trouvée)")
            print("[CAM] Mode free-run — filtrage des frames sombres activé")

        roi = self.camera.get_region()
        self.width = roi.width
        self.height = roi.height
        print(f"[CAM] Résolution: {self.width}x{self.height}")

        # Créer le stream et les buffers
        self.stream = self.camera.create_stream(None, None)
        payload = self.camera.get_payload()
        for _ in range(10):
            self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))

        self.connected = True
        return True

    def start_acquisition(self):
        self.camera.start_acquisition()

    def stop_acquisition(self):
        self.camera.stop_acquisition()

    def capture_frame(self, timeout_us: int = 5_000_000) -> Optional[np.ndarray]:
        """Capture une seule frame. Retourne l'image en niveaux de gris."""
        buf = self.stream.timeout_pop_buffer(timeout_us)
        if buf is None:
            print("[CAM DIAG] timeout_pop_buffer → None (aucun trigger reçu ?)")
            return None

        if buf.get_status() != Aravis.BufferStatus.SUCCESS:
            print(f"[CAM DIAG] Buffer status NOK: {buf.get_status()}")
            self.stream.push_buffer(buf)
            return None

        img = self._buffer_to_numpy(buf)

        # Diagnostic sur les premières frames : format + stats pixels
        if not hasattr(self, "_diag_count"):
            self._diag_count = 0
        if self._diag_count < 5:
            try:
                pf = buf.get_image_pixel_format()
                print(f"[CAM DIAG] frame#{self._diag_count} "
                      f"pixel_format=0x{pf:08x} shape={img.shape} dtype={img.dtype} "
                      f"min={img.min()} max={img.max()} mean={img.mean():.2f} "
                      f"p99={np.percentile(img, 99):.1f}")
            except Exception as e:
                print(f"[CAM DIAG] diag error: {e}")
            self._diag_count += 1

        self.stream.push_buffer(buf)
        return img

    def capture_sequence(self, count: int, interval_ms: float = 0) -> list:
        """Capture une séquence de `count` images en niveaux de gris."""
        images = []
        for i in range(count):
            frame = self.capture_frame()
            if frame is not None:
                # Convertir en niveaux de gris si couleur
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                images.append(frame)
            if interval_ms > 0 and i < count - 1:
                time.sleep(interval_ms / 1000.0)
        return images

    def set_exposure(self, exposure_us: int):
        self.exposure_us = exposure_us
        if self.camera:
            self.camera.set_exposure_time(exposure_us)

    def disconnect(self):
        if self.camera:
            dev = self.camera.get_device()
            if isinstance(dev, Aravis.GvDevice):
                dev.leave_control()
        self.connected = False

    def _buffer_to_numpy(self, buf) -> np.ndarray:
        """Convertit un buffer Aravis en array NumPy (BGR ou mono)."""
        pixel_format = buf.get_image_pixel_format()
        h = buf.get_image_height()
        w = buf.get_image_width()
        data = buf.get_data()
        raw = np.frombuffer(data, dtype=np.uint8)

        bayer_map = {
            Aravis.PIXEL_FORMAT_BAYER_RG_8: cv2.COLOR_BayerRG2BGR,
            Aravis.PIXEL_FORMAT_BAYER_GR_8: cv2.COLOR_BayerGR2BGR,
            Aravis.PIXEL_FORMAT_BAYER_GB_8: cv2.COLOR_BayerGB2BGR,
            Aravis.PIXEL_FORMAT_BAYER_BG_8: cv2.COLOR_BayerBG2BGR,
        }

        if pixel_format in bayer_map:
            raw = raw.reshape((h, w))
            return cv2.cvtColor(raw, bayer_map[pixel_format])
        elif pixel_format == Aravis.PIXEL_FORMAT_MONO_8:
            return raw.reshape((h, w))
        else:
            return raw.reshape((h, w))
