"""
Synchronisation automatique du saut (cas n°4).

Algorithme PLL numérique simplifié pour verrouiller le flash
stroboscopique sur la fréquence du balancier.
"""

import os
import time
from datetime import datetime

import cv2
import numpy as np

from async_saver import save_image_async


class AutoSynchronizer:
    """
    Algorithme de synchronisation automatique du saut (cas n°4).

    Principe stroboscopique :
    ─────────────────────────
    Le flash éclaire à une fréquence f_flash ≈ f_balancier.
    Le léger écart Δf = f_balancier - f_flash fait que le balancier
    semble se déplacer lentement dans les images.

    Le "saut" (classe 3 = 3 pics) correspond au passage du balancier
    par sa position d'équilibre. Pour le "verrouiller" au centre de
    l'acquisition, on ajuste T_trig_off par petits pas.

    Algorithme :
    ────────────
    1. Capturer des séquences glissantes de 4 images
    2. Classifier chaque séquence (0, 1, 2, 3)
    3. Suivre la position du saut dans le cycle apparent :
       - Si le saut (classe 3) arrive trop tôt → augmenter T_trig_off
       - Si le saut arrive trop tard → diminuer T_trig_off
    4. Converger vers un verrouillage (PLL numérique simplifié)
    """

    # Paramètres de l'algorithme
    COARSE_STEP_US = 100          # Pas large pour la recherche (µs)
    FINE_STEP_US = 10             # Pas fin pour le verrouillage (µs)
    WINDOW_SIZE = 20              # Nb d'images dans la fenêtre d'analyse
    TARGET_JUMP_POSITION = 0.5    # Position cible du saut dans le cycle (0.5 = milieu)
    TOLERANCE = 0.15              # Tolérance sur la position (±15%)
    MAX_ITERATIONS = 1000         # Itérations max avant abandon
    CONVERGENCE_COUNT = 5         # Nb de mesures stables pour déclarer la convergence
    MAX_SWEEP_US = 5000           # Distance max de balayage avant inversion (µs)
    MIN_SAUT_COUNT = 2            # Nb minimum de sauts pour autoriser le verrouillage
    DARK_FRAME_THRESHOLD = 30     # Seuil p99 pour détecter les frames illuminées

    def __init__(self, flasher, camera, classifier,
                 show_preview: bool = False):
        self.flasher = flasher
        self.camera = camera
        self.classifier = classifier
        self.locked = False
        self.history = []  # Historique des classes observées
        self.show_preview = show_preview
        self.phase = "SEARCH"         # SEARCH → balayage, LOCK → convergence fine
        self.sweep_direction = 1      # +1 ou -1
        self.sweep_origin = 0         # trig_off de départ du balayage
        self.saut_count = 0           # Nb total de sauts détectés

    def run(self) -> bool:
        """
        Boucle de synchronisation. Retourne True si verrouillé.

        Le cycle apparent du balancier en stroboscopie montre la séquence :
          ... → 0 → 1 → 3 → 2 → 0 → 1 → 3 → 2 → ...
        (immobile → approche → SAUT → éloignement → immobile → ...)

        On veut que la classe 3 (saut) soit centrée dans notre fenêtre
        d'observation, ce qui correspond à f_flash bien calé sur f_balancier.
        """
        print("\n" + "=" * 60)
        print("  SYNCHRONISATION AUTOMATIQUE DU SAUT")
        print("=" * 60)
        self.sweep_origin = self.flasher.current_trig_off
        print(f"  T_trig_off initial : {self.flasher.current_trig_off} µs")
        print(f"  f_flash            : {self.flasher.flash_frequency_hz:.4f} Hz")
        print(f"  Pas (recherche)    : {self.COARSE_STEP_US} µs")
        print()

        self.camera.start_acquisition()
        stable_count = 0
        lock_miss_count = 0           # Compteur d'itérations sans classe 3 en LOCK
        LOCK_MISS_LIMIT = 2 * self.WINDOW_SIZE  # Retour SEARCH si trop de miss
        image_buffer = []  # Buffer glissant de 4 images
        dark_skip_count = 0  # Compteur de frames sombres filtrées
        if not self.camera.hw_trigger:
            print(f"  [INFO] Filtrage dark-frame actif (seuil p99 >= {self.DARK_FRAME_THRESHOLD})")

        try:
            iteration = 0
            while iteration < self.MAX_ITERATIONS:
                # -- Capturer une nouvelle image --
                frame = self.camera.capture_frame()
                if frame is None:
                    continue

                # Convertir en niveaux de gris si nécessaire
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Filtrer les frames sombres (non illuminées par le flash)
                if not self.camera.hw_trigger:
                    brightness = np.percentile(frame, 99)
                    if brightness < self.DARK_FRAME_THRESHOLD:
                        dark_skip_count += 1
                        if dark_skip_count % 100 == 0:
                            print(f"  [dark-filter] {dark_skip_count} frames sombres ignorées (dernière p99={brightness:.0f})")
                        continue

                # Buffer glissant : garder les 4 dernières images (raw, sans CLAHE)
                image_buffer.append(frame)
                iteration += 1
                if len(image_buffer) > 4:
                    image_buffer.pop(0)

                # -- Affichage live --
                if self.show_preview:
                    self._show_frame(frame, len(image_buffer))

                # On a besoin de 4 images minimum pour classifier
                if len(image_buffer) < 4:
                    continue

                # -- Classifier --
                t_cls = time.perf_counter()
                classe = self.classifier.predict(list(image_buffer))
                dt_cls = (time.perf_counter() - t_cls) * 1000
                print(f"    [CHRONO] predict: {dt_cls:.1f} ms")
                self.history.append(classe)

                # Détection de saut (classe 3 ou transition 1↔2)
                if classe == 3:
                    self.saut_count += 1
                    lock_miss_count = 0  # Reset du compteur de miss
                    self._save_class3_frame(frame, iteration, prefix="sync_class3")
                    if self.classifier.last_processed is not None:
                        self._save_class3_frame(self.classifier.last_processed, iteration, prefix="sync_processed")
                    if self.phase == "SEARCH":
                        self.phase = "LOCK"
                        self.sweep_origin = self.flasher.current_trig_off
                        print(f"  >>> Saut (classe 3) détecté → phase LOCK")
                elif (len(self.history) >= 2
                      and ((self.history[-2] == 1 and classe == 2)
                           or (self.history[-2] == 2 and classe == 1))):
                    self.saut_count += 1
                    lock_miss_count = 0
                    if self.phase == "SEARCH":
                        self.phase = "LOCK"
                        self.sweep_origin = self.flasher.current_trig_off
                        print(f"  >>> Transition saut détectée → phase LOCK")
                else:
                    if self.phase == "LOCK":
                        lock_miss_count += 1
                        if lock_miss_count >= LOCK_MISS_LIMIT:
                            self.phase = "SEARCH"
                            self.sweep_origin = self.flasher.current_trig_off
                            stable_count = 0
                            lock_miss_count = 0
                            print(f"  >>> Saut perdu → retour SEARCH")

                # Garder l'historique limité
                if len(self.history) > self.WINDOW_SIZE:
                    self.history.pop(0)

                # -- Analyser la position du saut dans l'historique --
                if len(self.history) >= self.WINDOW_SIZE // 2:
                    jump_position = self._estimate_jump_position()
                    adjustment = self._compute_adjustment(jump_position)

                    phase_tag = self.phase
                    status = "LOCKED" if abs(adjustment) == 0 else f"adj={adjustment:+d} µs"
                    print(f"  [{iteration:3d}] classe={classe:2d}  "
                          f"pos_saut={jump_position:.2f}  "
                          f"trig_off={self.flasher.current_trig_off} µs  "
                          f"{status}  [{phase_tag}]")

                    if abs(adjustment) == 0 and self.saut_count >= self.MIN_SAUT_COUNT:
                        stable_count += 1
                        if stable_count >= self.CONVERGENCE_COUNT:
                            self.locked = True
                            print(f"\n  ✓ VERROUILLÉ après {iteration + 1} itérations")
                            print(f"    T_trig_off final : {self.flasher.current_trig_off} µs")
                            print(f"    f_flash          : {self.flasher.flash_frequency_hz:.6f} Hz")
                            print(f"    Sauts détectés   : {self.saut_count}")
                            break
                    elif abs(adjustment) == 0:
                        # Position OK mais pas assez de sauts → petit pas fin
                        adjustment = self.sweep_direction * self.FINE_STEP_US

                    if adjustment != 0:
                        stable_count = max(0, stable_count - 1)
                        new_trig_off = self.flasher.current_trig_off + adjustment
                        new_trig_off = max(1000, min(1_000_000, new_trig_off))
                        self.flasher.set_trig_off(new_trig_off)

                        # Inversion de direction si on dépasse MAX_SWEEP_US
                        sweep_dist = abs(self.flasher.current_trig_off - self.sweep_origin)
                        if sweep_dist >= self.MAX_SWEEP_US:
                            self.sweep_direction *= -1
                            print(f"  >>> Inversion de balayage "
                                  f"(direction={'+' if self.sweep_direction > 0 else '-'})")

        finally:
            self.camera.stop_acquisition()

        if self.show_preview:
            cv2.destroyWindow("Synchro — Live")

        if not self.locked:
            print(f"\n  ✗ Non verrouillé après {self.MAX_ITERATIONS} itérations")

        return self.locked

    def _show_frame(self, frame: np.ndarray, buf_len: int):
        """Affiche la frame courante avec des annotations de synchro."""
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

        # Crosshair central
        cv2.line(display, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
        cv2.line(display, (0, h // 2), (w, h // 2), (0, 255, 0), 1)

        # Infos textuelles
        classe = self.history[-1] if self.history else -1
        classe_names = {0: "1 pic", 1: "2p (f>F)", 2: "2p (F>f)", 3: "SAUT", -1: "?"}
        color = (0, 0, 255) if classe == 3 else (255, 255, 255)
        cv2.putText(display, f"Classe: {classe} ({classe_names.get(classe, '?')})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display, f"trig_off: {self.flasher.current_trig_off} us",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, f"f_flash: {self.flasher.flash_frequency_hz:.4f} Hz",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        status = "LOCKED" if self.locked else "searching..."
        cv2.putText(display, status, (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if self.locked else (0, 165, 255), 1)

        if w > 960:
            scale = 960 / w
            display = cv2.resize(display, None, fx=scale, fy=scale)

        cv2.imshow("Synchro — Live", display)
        cv2.waitKey(1)

    def _save_class3_frame(self, frame: np.ndarray, iteration: int,
                           prefix: str = "sync_class3"):
        """Enregistre une image sur disque (asynchrone)."""
        out_dir = "captures/class3_sync"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        fname = os.path.join(out_dir, f"{prefix}_{iteration:04d}_{ts}.png")
        save_image_async(fname, frame)
        print(f"    [SAVE] {fname}")

    def _estimate_jump_position(self) -> float:
        """
        Estime la position normalisée du saut (classe 3) dans l'historique.
        Retourne une valeur entre 0.0 et 1.0 (0.5 = milieu = cible).
        """
        window = self.history[-self.WINDOW_SIZE:]
        jump_indices = [i for i, c in enumerate(window) if c == 3]

        if not jump_indices:
            for i in range(1, len(window)):
                if ((window[i - 1] == 1 and window[i] == 2)
                        or (window[i - 1] == 2 and window[i] == 1)):
                    jump_indices.append(i)

        if not jump_indices:
            return -1.0

        avg_pos = np.mean(jump_indices) / len(window)
        return avg_pos

    def _compute_adjustment(self, jump_position: float) -> int:
        """
        Calcule l'ajustement de T_trig_off en µs.
        Positif → ralentir le flash (augmenter la période)
        Négatif → accélérer le flash (diminuer la période)
        """
        if jump_position < 0:
            step = self.COARSE_STEP_US if self.phase == "SEARCH" else self.FINE_STEP_US
            return self.sweep_direction * step

        error = jump_position - self.TARGET_JUMP_POSITION

        if abs(error) <= self.TOLERANCE:
            return 0

        step = self.FINE_STEP_US if self.phase == "LOCK" else self.COARSE_STEP_US
        direction = 1 if error > 0 else -1
        magnitude = max(1, min(step, int(abs(error) * step * 2)))
        return direction * magnitude
