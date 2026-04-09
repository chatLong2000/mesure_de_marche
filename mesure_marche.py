#!/usr/bin/env python3
"""
Mesure de marche automatisée — Version 2026
============================================
Script principal intégrant :
  1. Synchronisation automatique du saut (cas n°4)
  2. Calcul automatisé de la marche (f_réelle → s/j)
  3. Caractérisation des performances (comparaison étalon Witschi)

Architecture modulaire :
  - predictor/  : algorithmes de classification (interchangeables)
  - auto/       : automatisation (synchro, calcul marche, validation)

Utilise :
  - Aravis (GObject Introspection) pour le pilotage caméra SVS EXO273CGE
  - Protocole série UART pour le contrôle du flasher stroboscopique

Usage:
    conda activate ContrHorlo
    export GI_TYPELIB_PATH="/opt/homebrew/lib/girepository-1.0"

    # Avec matériel réel :
    python mesure_marche.py /dev/ttyUSB0

    # Mode test (sans matériel) :
    python mesure_marche.py test
"""

import sys
import os
import time
import math
import csv
import argparse
from typing import Optional

import cv2
import numpy as np

# Modules internes
from predictor import SignalClassifier
from auto import (
    AutoSynchronizer, RateCalculator, PerformanceValidator,
    MeasureResult, WitschiComparison, FREQ_NOMINALES, SECONDS_PER_DAY,
)

# ---------------------------------------------------------------------------
# Aravis import (GigE Vision camera)
# ---------------------------------------------------------------------------
os.environ.setdefault("GI_TYPELIB_PATH", "/opt/homebrew/lib/girepository-1.0")

try:
    import gi
    gi.require_version("Aravis", "0.8")
    from gi.repository import Aravis
    ARAVIS_AVAILABLE = True
except (ImportError, ValueError):
    ARAVIS_AVAILABLE = False
    print("[WARN] Aravis non disponible — mode dummy caméra activé.")

# ---------------------------------------------------------------------------
# Serial import
# ---------------------------------------------------------------------------
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARN] pyserial non disponible — mode dummy série activé.")

# ===========================================================================
#  FLASHER — Contrôle série (repris de controler/main.py)
# ===========================================================================

class Flasher:
    """Contrôle du flasher stroboscopique via UART."""

    COMMAND_TIMEOUT = 0.15  # secondes entre chaque commande

    def __init__(self, ser):
        self.ser = ser
        self.trig_enabled = False
        self.current_trig_off = 92000    # µs par défaut
        self.current_flash_on = 1000     # µs
        self.current_flash_off = 15000   # µs
        self.current_trig_expo = 19      # µs
        self.current_trig_shift = 1000   # µs

    def _send_cmd(self, cmd: str) -> None:
        """Envoie une commande au flasher (protocole texte `cmd:val;`)."""
        try:
            full_cmd = cmd if cmd.endswith(';') else cmd + ';'
            self.ser.write(full_cmd.encode())
            time.sleep(self.COMMAND_TIMEOUT)
        except Exception as e:
            print(f"[FLASH ERR] Envoi commande '{cmd}': {e}")

    # -- Commandes de base --
    def on(self):
        self._send_cmd("trig.en:1")
        self.trig_enabled = True

    def off(self):
        self._send_cmd("trig.en:0")
        self.trig_enabled = False

    def set_trig_off(self, val_us: int):
        """Période entre triggers (µs) → contrôle la fréquence du flash."""
        self._send_cmd(f"trig.off:{val_us}")
        self.current_trig_off = val_us

    def set_trig_expo(self, val_us: int):
        self._send_cmd(f"trig.expo:{val_us}")
        self.current_trig_expo = val_us

    def set_trig_shift(self, val_us: int):
        self._send_cmd(f"trig.shift:{val_us}")
        self.current_trig_shift = val_us

    def set_flash_on(self, val_us: int):
        self._send_cmd(f"flash.on:{val_us}")
        self.current_flash_on = val_us

    def set_flash_off(self, val_us: int):
        self._send_cmd(f"flash.off:{val_us}")
        self.current_flash_off = val_us

    @property
    def flash_frequency_hz(self) -> float:
        """Fréquence du flash = 1 / T_trig_off (convertie de µs en Hz)."""
        if self.current_trig_off <= 0:
            return 0.0
        return 1e6 / self.current_trig_off

    def apply_defaults(self):
        """Envoie tous les paramètres par défaut au flasher."""
        self.set_trig_expo(self.current_trig_expo)
        self.set_trig_off(self.current_trig_off)
        self.set_flash_on(self.current_flash_on)
        self.set_flash_off(self.current_flash_off)
        self.set_trig_shift(self.current_trig_shift)


class DummySerial:
    """Port série factice pour les tests sans matériel."""

    def __init__(self):
        self.is_open = True

    def write(self, data: bytes) -> None:
        decoded = data.decode().strip()
        if decoded:
            print(f"  [DUMMY TX] {decoded}")

    def readline(self) -> bytes:
        time.sleep(0.5)
        return b"OK\n"

    def close(self):
        self.is_open = False


# ===========================================================================
#  CAMÉRA — Pilotage Aravis
# ===========================================================================

class AravisCamera:
    """Pilotage de la caméra SVS EXO273CGE via Aravis GI."""

    DARK_FRAME_THRESHOLD = 30  # Seuil sur le percentile 99 pour détecter les frames illuminées

    def __init__(self, exposure_us: int = 10000, gain_db: float = 0.0):
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
        try:
            self.camera.set_exposure_time(self.exposure_us)
        except Exception as e:
            print(f"[CAM WARN] ExposureTime: {e}")

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
            return None

        if buf.get_status() != Aravis.BufferStatus.SUCCESS:
            self.stream.push_buffer(buf)
            return None

        img = self._buffer_to_numpy(buf)
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



# ===========================================================================
#  POINT D'ENTRÉE
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Mesure de marche horlogère — Stroboscopie optique 2026"
    )
    parser.add_argument("port", help="Port série (ex: /dev/ttyUSB0, COM3, ou 'test')")
    parser.add_argument("--calibre", default="28800",
                        choices=list(FREQ_NOMINALES.keys()),
                        help="Fréquence nominale du calibre en A/h (défaut: 28800)")
    parser.add_argument("--trig-off", type=int, default=250000,
                        help="T_trig_off initial en µs (défaut: 250000 → 4 Hz)")
    parser.add_argument("--flash-on", type=int, default=1000,
                        help="Durée flash ON en µs (défaut: 1000)")
    parser.add_argument("--exposure", type=int, default=10000,
                        help="Temps d'exposition caméra en µs (défaut: 10000)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Durée de mesure en secondes (défaut: 10)")
    parser.add_argument("--validate", type=int, default=0, metavar="N",
                        help="Mode caractérisation : effectuer N mesures comparatives")
    parser.add_argument("--skip-sync", action="store_true",
                        help="Passer la phase de synchronisation automatique")
    parser.add_argument("--show-preview", action="store_true",
                        help="Afficher les images capturées en temps réel (fenêtre OpenCV)")
    parser.add_argument("--peak-height", type=int, default=5,
                        help="Seuil de hauteur des pics pour la classification (défaut: 5)")
    parser.add_argument("--subtract-threshold", type=int, default=15,
                        help="Seuil de bruit après soustraction de médiane (défaut: 15)")
    parser.add_argument("--output", default="mesure_results.csv",
                        help="Fichier CSV de sortie (défaut: mesure_results.csv)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  MESURE DE MARCHE HORLOGÈRE — v2026")
    print("  Stroboscopie optique automatisée")
    print("=" * 60)

    # -- 1. Connexion série (flasher) --
    if args.port.lower() == "test":
        ser = DummySerial()
        print("[INFO] Mode test — série factice")
    else:
        if not SERIAL_AVAILABLE:
            print("[ERR] pyserial requis. pip install pyserial")
            sys.exit(1)
        ser = serial.Serial(args.port, 115200, timeout=1)
        print(f"[INFO] Série connectée : {args.port} @ 115200")

    flasher = Flasher(ser)

    # -- 2. Connexion caméra --
    if ARAVIS_AVAILABLE:
        camera = AravisCamera(exposure_us=args.exposure)
        try:
            camera.connect()
        except Exception as e:
            print(f"[WARN] Caméra Aravis : {e}")
            print("[INFO] Basculement sur caméra factice.")
            camera = DummyCamera()
            camera.connect()
    else:
        camera = DummyCamera()
        camera.connect()

    # -- 3. Configurer le flasher --
    f_nominale = FREQ_NOMINALES[args.calibre]
    print(f"\n[INFO] Calibre : {args.calibre} A/h → f_nominale = {f_nominale} Hz")

    flasher.set_trig_off(args.trig_off)
    flasher.set_flash_on(args.flash_on)
    flasher.apply_defaults()
    flasher.on()
    print(f"[INFO] Flash activé — f_flash = {flasher.flash_frequency_hz:.4f} Hz")

    classifier = SignalClassifier(height_threshold=args.peak_height,
                                  subtract_threshold=args.subtract_threshold)

    try:
        # -- 4. Synchronisation automatique --
        if not args.skip_sync:
            syncer = AutoSynchronizer(flasher, camera, classifier,
                                     show_preview=args.show_preview)
            locked = syncer.run()
            if not locked:
                print("\n[WARN] Synchronisation non verrouillée. "
                      "La mesure continuera avec les paramètres actuels.")
        else:
            print("\n[INFO] Synchronisation ignorée (--skip-sync)")

        # -- 5. Mesure de marche --
        rate_calc = RateCalculator(flasher, camera, classifier,
                                   f_nominale_hz=f_nominale,
                                   show_preview=args.show_preview)

        if args.validate > 0:
            # Mode caractérisation
            validator = PerformanceValidator(rate_calc)
            validator.run_validation(n_measures=args.validate,
                                     measure_duration_s=args.duration)
            validator.export_csv(args.output)
        else:
            # Mesure simple
            result = rate_calc.measure(duration_s=args.duration)

            # Sauvegarder
            with open(args.output, 'a', newline='') as f:
                writer = csv.writer(f)
                if os.path.getsize(args.output) == 0:
                    writer.writerow(["timestamp", "f_flash", "f_app", "f_reelle",
                                     "f_nominale", "ecart_hz", "marche_s_j",
                                     "classe_dom", "nb_images", "trig_off",
                                     "confidence"])
                writer.writerow([result.timestamp, result.f_flash_hz,
                                 result.f_apparente_hz, result.f_reelle_hz,
                                 result.f_nominale_hz, result.ecart_hz,
                                 result.marche_s_par_jour,
                                 result.classe_dominante, result.nb_images,
                                 result.trig_off_us, result.confidence])
            print(f"\n[INFO] Résultat sauvegardé → {args.output}")

    except KeyboardInterrupt:
        print("\n\n[INFO] Interruption utilisateur.")

    finally:
        # Nettoyage
        flasher.off()
        camera.stop_acquisition() if hasattr(camera, 'stop_acquisition') else None
        camera.disconnect()
        ser.close()
        print("[INFO] Terminé.")


if __name__ == "__main__":
    main()
