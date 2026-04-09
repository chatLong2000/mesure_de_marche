"""
Calcul automatisé de la marche (écart de fréquence en secondes/jour).

Mesure la fréquence apparente du balancier par stroboscopie
et en déduit la marche horlogère.
"""

import os
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from .models import MeasureResult, SECONDS_PER_DAY


class RateCalculator:
    """
    Calcul de la marche (écart de fréquence en secondes/jour).

    Principe :
    ──────────
    En stroboscopie, la fréquence apparente du balancier est :
        f_app = |f_balancier - f_flash|

    Donc :
        f_balancier = f_flash ± f_app

    La marche en s/j :
        marche = (f_reelle - f_nominale) / f_nominale × 86400

    Pour mesurer f_app, on compte le nombre de cycles apparents
    (transitions 0→1→3→2→0 = 1 cycle complet) pendant un temps donné.
    """

    DARK_FRAME_THRESHOLD = 30

    def __init__(self, flasher, camera, classifier,
                 f_nominale_hz: float = 4.0, show_preview: bool = False):
        self.flasher = flasher
        self.camera = camera
        self.classifier = classifier
        self.f_nominale = f_nominale_hz
        self.show_preview = show_preview

    def measure(self, duration_s: float = 10.0, verbose: bool = True) -> MeasureResult:
        """
        Effectue une mesure de marche sur `duration_s` secondes.

        Retourne un MeasureResult avec la marche en s/j.
        """
        print("\n" + "=" * 60)
        print("  MESURE DE MARCHE")
        print("=" * 60)
        print(f"  f_nominale : {self.f_nominale:.1f} Hz ({self.f_nominale * 7200:.0f} A/h)")
        print(f"  f_flash    : {self.flasher.flash_frequency_hz:.6f} Hz")
        print(f"  Durée      : {duration_s:.1f} s")
        print()

        self.camera.start_acquisition()

        classifications = []
        timestamps = []
        image_buffer = []
        t_start = time.time()

        try:
            while (time.time() - t_start) < duration_s:
                frame = self.camera.capture_frame()
                if frame is None:
                    continue

                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Filtrer les frames sombres (non illuminées par le flash)
                if not self.camera.hw_trigger:
                    brightness = np.percentile(frame, 99)
                    if brightness < self.DARK_FRAME_THRESHOLD:
                        continue

                # Buffer glissant (raw, sans CLAHE)
                image_buffer.append(frame)
                if len(image_buffer) > 4:
                    image_buffer.pop(0)

                if len(image_buffer) < 4:
                    continue

                classe = self.classifier.predict(list(image_buffer))
                classifications.append(classe)
                timestamps.append(time.time())

                # Sauvegarder les images de classe 3 (saut)
                if classe == 3:
                    self._save_class3_frame(frame, len(classifications), prefix="mesure_class3")
                    if self.classifier.last_processed is not None:
                        self._save_class3_frame(self.classifier.last_processed, len(classifications), prefix="mesure_processed")

                # -- Affichage live --
                if self.show_preview:
                    self._show_frame(frame, classe, classifications, t_start)

                if verbose and len(classifications) % 10 == 0:
                    elapsed = time.time() - t_start
                    print(f"  [{elapsed:.1f}s] {len(classifications)} classifications, "
                          f"dernière classe = {classe}")

        finally:
            self.camera.stop_acquisition()
            if self.show_preview:
                cv2.destroyWindow("Mesure — Live")

        # -- Analyser les résultats --
        if len(classifications) < 10:
            print("[WARN] Pas assez de données pour une mesure fiable.")
            return self._empty_result()

        f_app = self._measure_apparent_frequency(classifications, timestamps)
        f_flash = self.flasher.flash_frequency_hz

        # Déterminer le signe : f_réelle = f_flash + f_app ou f_flash - f_app
        direction = self._detect_rotation_direction(classifications)
        f_reelle = f_flash + direction * f_app

        ecart = f_reelle - self.f_nominale
        marche = (ecart / self.f_nominale) * SECONDS_PER_DAY

        # Classe dominante
        valid_classes = [c for c in classifications if c >= 0]
        classe_dom = max(set(valid_classes), key=valid_classes.count) if valid_classes else -1

        # Indice de confiance basé sur le taux de classifications valides
        confidence = len(valid_classes) / len(classifications) if classifications else 0

        result = MeasureResult(
            timestamp=datetime.now().isoformat(),
            f_flash_hz=f_flash,
            f_apparente_hz=f_app,
            f_reelle_hz=f_reelle,
            f_nominale_hz=self.f_nominale,
            ecart_hz=ecart,
            marche_s_par_jour=marche,
            classe_dominante=classe_dom,
            nb_images=len(classifications),
            trig_off_us=self.flasher.current_trig_off,
            confidence=confidence,
        )

        self._print_result(result)
        return result

    def _measure_apparent_frequency(self, classes: list, ts: list) -> float:
        """
        Mesure la fréquence apparente en comptant les cycles complets.
        """
        jump_times = [ts[i] for i, c in enumerate(classes) if c == 3]
        for i in range(1, len(classes)):
            if ((classes[i-1] == 1 and classes[i] == 2)
                    or (classes[i-1] == 2 and classes[i] == 1)):
                jump_times.append(ts[i])
        jump_times.sort()

        if len(jump_times) < 2:
            return self._estimate_freq_from_transitions(classes, ts)

        periods = [jump_times[i+1] - jump_times[i]
                    for i in range(len(jump_times) - 1)
                    if (jump_times[i+1] - jump_times[i]) > 0.1]

        if not periods:
            return 0.0

        mean_period = np.mean(periods)
        return 1.0 / mean_period if mean_period > 0 else 0.0

    def _estimate_freq_from_transitions(self, classes: list, ts: list) -> float:
        """Estimation de secours par comptage de transitions entre classes."""
        transitions = 0
        for i in range(1, len(classes)):
            if classes[i] != classes[i-1] and classes[i] >= 0 and classes[i-1] >= 0:
                transitions += 1

        if len(ts) < 2:
            return 0.0

        total_time = ts[-1] - ts[0]
        if total_time <= 0:
            return 0.0

        cycles = transitions / 4.0
        return cycles / total_time

    def _detect_rotation_direction(self, classes: list) -> int:
        """
        Détecte le sens de rotation apparent.
        +1 si f_réelle > f_flash (séquence 0→1→3→2)
        -1 si f_réelle < f_flash (séquence 0→2→3→1)
        """
        forward = 0
        backward = 0

        for i in range(1, len(classes)):
            prev, curr = classes[i-1], classes[i]
            if (prev == 1 and curr == 3) or (prev == 3 and curr == 2):
                forward += 1
            elif (prev == 2 and curr == 3) or (prev == 3 and curr == 1):
                backward += 1

        return 1 if forward >= backward else -1

    def _print_result(self, r: MeasureResult):
        print(f"\n  ────────── RÉSULTAT ──────────")
        print(f"  f_flash        : {r.f_flash_hz:.6f} Hz")
        print(f"  f_apparente    : {r.f_apparente_hz:.6f} Hz")
        print(f"  f_réelle       : {r.f_reelle_hz:.6f} Hz")
        print(f"  f_nominale     : {r.f_nominale_hz:.6f} Hz")
        print(f"  Écart          : {r.ecart_hz:+.6f} Hz")
        print(f"  ╔══════════════════════════════╗")
        print(f"  ║  MARCHE : {r.marche_s_par_jour:+.2f} s/jour       ║")
        print(f"  ╚══════════════════════════════╝")
        print(f"  Confiance      : {r.confidence:.0%}")
        print(f"  Images         : {r.nb_images}")
        print(f"  ─────────────────────────────")

    def _save_class3_frame(self, frame: np.ndarray, index: int,
                           prefix: str = "mesure_class3"):
        """Enregistre une image sur disque."""
        out_dir = "captures/class3_mesure"
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        fname = os.path.join(out_dir, f"{prefix}_{index:04d}_{ts}.png")
        cv2.imwrite(fname, frame)
        print(f"    [SAVE] {fname}")

    def _show_frame(self, frame: np.ndarray, classe: int,
                    classifications: list, t_start: float):
        """Affiche la frame courante avec les infos de mesure."""
        if frame.ndim == 2:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(frame)
            display = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            display = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        h, w = display.shape[:2]

        cv2.line(display, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
        cv2.line(display, (0, h // 2), (w, h // 2), (0, 255, 0), 1)

        elapsed = time.time() - t_start
        classe_names = {0: "1 pic", 1: "2p (f>F)", 2: "2p (F>f)", 3: "SAUT", -1: "?"}
        color = (0, 0, 255) if classe == 3 else (255, 255, 255)

        cv2.putText(display, f"Classe: {classe} ({classe_names.get(classe, '?')})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display, f"t={elapsed:.1f}s  n={len(classifications)}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, f"f_flash: {self.flasher.flash_frequency_hz:.4f} Hz",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Mini-histogramme des classes
        bar_h = 20
        bar_y = h - bar_h - 5
        recent = classifications[-min(len(classifications), w // 4):]
        class_colors = {
            0: (180, 180, 180), 1: (255, 180, 0), 2: (0, 180, 255),
            3: (0, 0, 255), -1: (50, 50, 50)
        }
        bar_w = max(1, (w - 20) // max(len(recent), 1))
        for i, c in enumerate(recent):
            x = 10 + i * bar_w
            cv2.rectangle(display, (x, bar_y), (x + bar_w - 1, bar_y + bar_h),
                          class_colors.get(c, (50, 50, 50)), -1)

        if w > 960:
            scale = 960 / w
            display = cv2.resize(display, None, fx=scale, fy=scale)

        cv2.imshow("Mesure — Live", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt("Arrêt demandé via fenêtre preview")

    def _empty_result(self) -> MeasureResult:
        return MeasureResult(
            timestamp=datetime.now().isoformat(),
            f_flash_hz=self.flasher.flash_frequency_hz,
            f_apparente_hz=0.0,
            f_reelle_hz=0.0,
            f_nominale_hz=self.f_nominale,
            ecart_hz=0.0,
            marche_s_par_jour=0.0,
            classe_dominante=-1,
            nb_images=0,
            trig_off_us=self.flasher.current_trig_off,
            confidence=0.0,
        )
