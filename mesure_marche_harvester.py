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
  - Harvester (caméra SVS EXO273CGE)
  - Protocole série UART pour le contrôle du flasher stroboscopique
"""

import sys
import os
import math
import csv
import argparse
import cv2
import numpy as np
import serial
import time

from harvesters.core import Harvester

# Modules internes
from src.signal_classifier import SignalClassifier
from src.flasher import Flasher
from src.camera import HarvesterCamera
from src.synchronizer import AutoSynchronizer
from src.rate_calculator import RateCalculator
from src.validator import PerformanceValidator
from src.models import FREQ_NOMINALES

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
    parser.add_argument("--trig-off", type=int, default=680008,
                        help="T_trig_off initial en µs (défaut: 250000 → 4 Hz)")
    parser.add_argument("--flash-on", type=int, default=1000,
                        help="Durée flash ON en µs (défaut: 1000)")
    parser.add_argument("--exposure", type=int, default=1000,
                        help="Temps d'exposition caméra en µs (défaut: 10000)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Durée de mesure en secondes (défaut: 10)")
    parser.add_argument("--validate", type=int, default=0, metavar="N",
                        help="Mode caractérisation : effectuer N mesures comparatives")
    parser.add_argument("--skip-sync", action="store_true",
                        help="Passer la phase de synchronisation automatique")
    parser.add_argument("--show-preview", action="store_true",
                        help="Afficher les images capturées en temps réel (fenêtre OpenCV)")
    parser.add_argument("--output", default="mesure_results.csv",
                        help="Fichier CSV de sortie (défaut: mesure_results.csv)")
    return parser.parse_args()


def main():
    args = parse_args()

    # -- 1. Connexion série (flasher) --
    ser = serial.Serial(args.port, 115200, timeout=1)
    print(f"[INFO] Série connectée : {args.port} @ 115200")

    flasher = Flasher(ser)

    # -- 2. Connexion caméra --
    CTI_FILE = "/home/rlaborde/master_hes/pi/VimbaX_2025-3/cti/VimbaGigETL.cti"
    camera = HarvesterCamera(CTI_FILE)
    camera.connect()

    # -- 3. Configurer le flasher --
    f_nominale = FREQ_NOMINALES[args.calibre]
    print(f"\n[INFO] Calibre : {args.calibre} A/h → f_nominale = {f_nominale} Hz")

    flasher.trig_off(680008)
    time.sleep(1)
    flasher.trig_expo(19)
    time.sleep(1)
    flasher.trig_shift(1000)
    time.sleep(1)
    flasher.flash_on(1000)
    time.sleep(1)
    flasher.flash_off(85417)
    time.sleep(1)

    flasher.print_config()

    flasher.on()
    time.sleep(1)
    # print(f"[INFO] Flash activé")
    # input("Press Enter to continue...")
    # flasher.off()
    # # camera.stop_acquisition() if hasattr(camera, 'stop_acquisition') else None
    # # camera.disconnect()
    # ser.close()
    # print("[INFO] Terminé.")
    # exit()

    classifier = SignalClassifier() # using default height_threshold = 10

    try:
        # -- 4. Synchronisation automatique --
        syncer = AutoSynchronizer(flasher, camera, classifier, show_preview=False)
        
        locked = syncer.run()

        if not locked:
            print("\n[WARN] Synchronisation non verrouillée. "
                    "La mesure continuera avec les paramètres actuels.")
            exit()

        # -- 5. Mesure de marche --
        rate_calc = RateCalculator(flasher, camera, classifier, f_nominale_hz=f_nominale, show_preview=False)

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
        flasher.off()
        camera.stop_acquisition() if hasattr(camera, 'stop_acquisition') else None
        camera.disconnect()
        ser.close()
        print("[INFO] Terminé.")

if __name__ == "__main__":
    main()
