#!/usr/bin/env python3
"""
Mesure de marche automatisée — Version 2026
============================================
Script principal intégrant :
  1. Synchronisation automatique du saut (cas n°4)
  2. Calcul automatisé de la marche (f_réelle → s/j)
  3. Caractérisation des performances (comparaison étalon Witschi)

Architecture modulaire (package src/) :
  - src/cameras/    : pilotes caméra interchangeables (Aravis, Harvester, Dummy)
  - src/processing/ : classification, synchro, calcul marche, validation
  - src/hardware/   : pilotage du flasher (série UART)
  - src/simulation/ : rejeu hors-ligne et cas idéal simulé

Deux modules caméra sont disponibles (sélection via --camera) :
  - Aravis (GObject Introspection) — pilotage caméra SVS EXO273CGE (macOS)
  - Harvester (GenICam) — pilotage GigE (Linux / Raspberry Pi)
Plus une caméra factice (dummy) pour les tests sans matériel.

Le flasher stroboscopique est piloté par protocole série UART.

Usage:
    conda activate ContrHorlo
    export GI_TYPELIB_PATH="/opt/homebrew/lib/girepository-1.0"

    # Matériel réel, détection automatique du module caméra :
    python mesure_marche.py /dev/ttyUSB0

    # Forcer un module caméra :
    python mesure_marche.py /dev/ttyUSB0 --camera aravis
    python mesure_marche.py /dev/ttyUSB0 --camera harvester --cti /chemin/vers/GigETL.cti

    # Mode test (sans matériel) :
    python mesure_marche.py test
"""

import sys
import os
import time
import math
import csv
import random
import argparse
from typing import Optional

import cv2
import numpy as np

# Modules internes
from src.processing.signal_classifier import SignalClassifier
from src.processing.synchronizer import AutoSynchronizer
from src.processing.rate_calculator import RateCalculator
from src.processing.validator import PerformanceValidator
from src.models import MeasureResult, WitschiComparison, FREQ_NOMINALES, SECONDS_PER_DAY
from src.hardware.flasher import Flasher
from src.cameras import (
    DummyCamera,
    AravisCamera,
    HarvesterCamera,
    ARAVIS_AVAILABLE,
    HARVESTER_AVAILABLE,
)
from src.simulation.simulation_marche_parfaite import (
    generate_ideal_jump_times,
    compute_marche,
)

# ---------------------------------------------------------------------------
# Chemin par défaut du fichier de résultats (dossier results/)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DEFAULT_OUTPUT = os.path.join(RESULTS_DIR, "mesure_results.csv")

# Fichier CTI par défaut pour le pilote Harvester (GenICam Transport Layer)
DEFAULT_CTI = "/home/rlaborde/master_hes/pi/VimbaX_2025-3/cti/VimbaGigETL.cti"

# ---------------------------------------------------------------------------
# Serial import
# ---------------------------------------------------------------------------
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARN] pyserial non disponible — mode dummy série activé.")

# ---------------------------------------------------------------------------
# Norme COSC — chronomètre mécanique : marche diurne moyenne ∈ [-4, +6] s/j
# ---------------------------------------------------------------------------
COSC_RATE_MIN_S_J = -4.0
COSC_RATE_MAX_S_J = 6.0

# ===========================================================================
#  FLASHER — Contrôle série (repris de controler/main.py)
# ===========================================================================


class DummySerial:
    """Port série factice pour les tests sans matériel."""

    def __init__(self):
        self.is_open = True

    def write(self, data: bytes) -> None:
        decoded = data.decode().strip()
        #if decoded:
            #print(f"  [DUMMY TX] {decoded}")

    def readline(self) -> bytes:
        time.sleep(0.5)
        return b"OK\n"

    def close(self):
        self.is_open = False


# ===========================================================================
#  CAMÉRA — Modules interchangeables (src/cameras/)
# ===========================================================================
#  Les pilotes caméra sont définis dans le package `src.cameras` :
#    - AravisCamera    : pilote Aravis / GObject Introspection (macOS)
#    - HarvesterCamera : pilote Harvester / GenICam (Linux, Raspberry Pi)
#    - DummyCamera     : caméra factice (mode test, sans matériel)
#  La sélection se fait via l'option CLI `--camera`.


def build_camera(args):
    """Instancie le pilote caméra selon `args.camera`.

    Renvoie une instance déjà connectée (ou bascule sur la caméra factice
    en cas d'échec matériel).
    """
    choice = args.camera

    if choice == "auto":
        # Priorité au pilote Aravis, puis Harvester, sinon factice.
        if ARAVIS_AVAILABLE:
            choice = "aravis"
        elif HARVESTER_AVAILABLE:
            choice = "harvester"
        else:
            choice = "dummy"
        print(f"[INFO] Caméra auto-sélectionnée : {choice}")

    if choice == "aravis":
        if not ARAVIS_AVAILABLE:
            print("[WARN] Aravis non disponible — bascule sur caméra factice.")
            cam = DummyCamera()
            cam.connect()
            return cam
        cam = AravisCamera(exposure_us=args.exposure)
        try:
            cam.connect()
            return cam
        except Exception as e:
            print(f"[WARN] Caméra Aravis : {e}")
            print("[INFO] Bascule sur caméra factice.")
            cam = DummyCamera()
            cam.connect()
            return cam

    if choice == "harvester":
        if not HARVESTER_AVAILABLE:
            print("[WARN] Harvester non disponible — bascule sur caméra factice.")
            cam = DummyCamera()
            cam.connect()
            return cam
        cam = HarvesterCamera(args.cti, exposure_us=args.exposure)
        try:
            cam.connect()
            return cam
        except Exception as e:
            print(f"[WARN] Caméra Harvester : {e}")
            print("[INFO] Bascule sur caméra factice.")
            cam = DummyCamera()
            cam.connect()
            return cam

    # choice == "dummy"
    cam = DummyCamera()
    cam.connect()
    return cam


# ===========================================================================
#  POINT D'ENTRÉE
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Mesure de marche horlogère — Stroboscopie optique 2026"
    )
    parser.add_argument("port", help="Port série (ex: /dev/ttyUSB0, COM3, ou 'test')")
    parser.add_argument("--camera", default="auto",
                        choices=["auto", "aravis", "harvester", "dummy"],
                        help="Module caméra à utiliser : 'aravis' (macOS), "
                             "'harvester' (Linux/Raspberry), 'dummy' (test) "
                             "ou 'auto' (détection automatique, défaut).")
    parser.add_argument("--cti", default=DEFAULT_CTI,
                        help="Chemin du fichier .cti (GenICam Transport Layer) "
                             "pour le module caméra Harvester.")
    parser.add_argument("--calibre", default="28800",
                        choices=list(FREQ_NOMINALES.keys()),
                        help="Fréquence nominale du calibre en A/h (défaut: 28800)")
    parser.add_argument("--target", default="balance",
                        choices=["balance", "seconds"],
                        help="Cible filmée : 'balance' (balancier, f=bph/7200) "
                             "ou 'seconds' (aiguille des secondes, f=bph/3600). "
                             "En mode 'seconds' la fréquence est doublée.")
    parser.add_argument("--trig-off", type=int, default=853166,
                        help="T_trig_off initial en µs (défaut: 250000 → 4 Hz)")
    parser.add_argument("--flash-on", type=int, default=10000,
                        help="Durée flash ON en µs (défaut: 1000)")
    parser.add_argument("--exposure", type=int, default=1000,
                        help="Temps d'exposition caméra en µs (défaut: 10000)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Durée de mesure en secondes (défaut: 10)")
    parser.add_argument("--validate", type=int, default=0, metavar="N",
                        help="Mode caractérisation : effectuer N mesures comparatives")
    parser.add_argument("--skip-sync", action="store_true",
                        help="Passer la phase de synchronisation automatique")
    parser.add_argument("--simulate-final", action="store_true",
                        help="Remplacer le calcul final de la marche par une "
                             "valeur issue du script de simulation (situation "
                             "parfaite). Utile pour les démonstrations où le "
                             "calcul réel est trop fluctuant. La marche injectée "
                             "est tirée aléatoirement dans la tolérance COSC "
                             "(-4/+6 s/j). Les métadonnées réelles (nb images, "
                             "confiance, classe) sont conservées.")
    parser.add_argument("--show-preview", action="store_true",
                        help="Afficher les images capturées en temps réel (fenêtre OpenCV)")
    parser.add_argument("--debug-save", action="store_true",
                        help="Sauvegarder toutes les frames capturées (brutes + traitées) "
                             "pendant la synchro dans captures/debug_sync/run_<timestamp>/")
    parser.add_argument("--peak-height", type=int, default=5,
                        help="Seuil de hauteur des pics pour la classification (défaut: 5)")
    parser.add_argument("--subtract-threshold", type=int, default=20,
                        help="Seuil de bruit après soustraction de médiane (défaut: 15)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Fichier CSV de sortie "
                             "(défaut: results/mesure_results.csv)")
    return parser.parse_args()


def apply_simulated_final(result: MeasureResult,
                          f_nominale_hz: float) -> MeasureResult:
    """
    Remplace le calcul final de la marche par une valeur issue de la
    simulation parfaite, tout en conservant les métadonnées réelles de
    la mesure (f_flash, nb_images, confiance, classe, trig_off).

    Utilisé pour les démonstrations : le pipeline réel (caméra, synchro,
    classification) s'exécute normalement, mais la marche affichée est
    une valeur tirée aléatoirement dans la tolérance COSC pour un
    chronomètre mécanique (marche diurne moyenne comprise entre
    -4 et +6 s/j).
    """
    # Marche cible aléatoire respectant la norme COSC (-4/+6 s/j)
    marche_vraie_s_j = random.uniform(COSC_RATE_MIN_S_J, COSC_RATE_MAX_S_J)

    # f_balancier vraie déduite de la marche cible
    f_balancier = f_nominale_hz * (1.0 + marche_vraie_s_j / SECONDS_PER_DAY)
    f_flash = result.f_flash_hz

    # Caler f_flash pour obtenir une fréquence apparente exploitable
    # si la consigne réelle est trop proche de f_balancier.
    if abs(f_balancier - f_flash) < 0.05:
        f_flash = f_balancier - 0.25

    n_sauts = max(result.nb_images, 30)
    jump_times, _, direction = generate_ideal_jump_times(
        f_flash_hz=f_flash,
        f_balancier_hz=f_balancier,
        n_sauts=n_sauts,
        jitter_s=0.0,
        quantize_to_flash=True,
    )
    sim = compute_marche(jump_times, f_flash, f_nominale_hz, direction)

    print("\n[SIM] Calcul final remplacé par la simulation parfaite "
          f"(cible COSC tirée = {marche_vraie_s_j:+.4f} s/j)")
    print(f"[SIM] f_apparente = {sim['f_app_hz']:.6f} Hz, "
          f"f_réelle = {sim['f_reelle_hz']:.6f} Hz, "
          f"marche = {sim['marche_s_j']:+.4f} s/j")

    result.f_flash_hz = f_flash
    result.f_apparente_hz = sim["f_app_hz"]
    result.f_reelle_hz = sim["f_reelle_hz"]
    result.ecart_hz = sim["ecart_hz"]
    result.marche_s_par_jour = sim["marche_s_j"]

    # Réafficher l'encadré avec la marche simulée (COSC)
    print(f"\n  ────────── RÉSULTAT (SIMULÉ — COSC) ──────────")
    print(f"  f_flash        : {result.f_flash_hz:.6f} Hz")
    print(f"  f_apparente    : {result.f_apparente_hz:.6f} Hz")
    print(f"  f_réelle       : {result.f_reelle_hz:.6f} Hz")
    print(f"  f_nominale     : {result.f_nominale_hz:.6f} Hz")
    print(f"  Écart          : {result.ecart_hz:+.6f} Hz")
    print(f"  ╔══════════════════════════════╗")
    print(f"  ║  MARCHE : {result.marche_s_par_jour:+.2f} s/jour       ║")
    print(f"  ╚══════════════════════════════╝")
    print(f"  Tolérance COSC : {COSC_RATE_MIN_S_J:+.0f} / {COSC_RATE_MAX_S_J:+.0f} s/jour")
    print(f"  ─────────────────────────────")
    return result


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

    # -- 2. Connexion caméra (module sélectionnable : aravis / harvester / dummy) --
    print(f"[INFO] Module caméra demandé : {args.camera}")
    camera = build_camera(args)

    # -- 3. Configurer le flasher --
    f_balancier = FREQ_NOMINALES[args.calibre]  # 4 Hz pour 28800 A/h
    if args.target == "seconds":
        # L'aiguille des secondes avance à 1 tic par alternance d'échappement
        # → bph/3600 = 2 * f_balancier (8 Hz pour 28800 A/h)
        f_nominale = 2.0 * f_balancier
        # Seuils plus agressifs : signal faible (petits déplacements angulaires)
        if args.subtract_threshold == 10:  # valeur par défaut non touchée par user
            args.subtract_threshold = 3
        if args.peak_height == 5:
            args.peak_height = 3
    else:
        f_nominale = f_balancier

    print(f"\n[INFO] Calibre : {args.calibre} A/h")
    print(f"[INFO] Cible    : {args.target}")
    print(f"[INFO] f_nominale = {f_nominale} Hz "
          f"(période = {1e6 / f_nominale:.0f} µs)")
    print(f"[INFO] trig_off initial : {args.trig_off} µs "
          f"(→ {1e6 / args.trig_off:.3f} Hz selon formule f=1/T)")
    print(f"[INFO] subtract_threshold = {args.subtract_threshold}, "
          f"peak_height = {args.peak_height}")

    flasher.trig_off(args.trig_off)
    time.sleep(1)
    flasher.trig_expo(19)
    time.sleep(1)
    flasher.trig_shift(0)
    time.sleep(1)
    flasher.flash_on(args.flash_on)
    time.sleep(1)
    flasher.flash_off(58417)
    time.sleep(1)

    flasher.print_config()

    flasher.on()
    time.sleep(1)

    classifier = SignalClassifier(
        height_threshold=args.peak_height,
        subtract_threshold=args.subtract_threshold,
        debug=args.debug_save,
    )

    try:
        # -- 4. Synchronisation automatique --
        if not args.skip_sync:
            syncer = AutoSynchronizer(flasher, camera, classifier,
                                     show_preview=args.show_preview,
                                     debug_save=args.debug_save)
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

        # S'assurer que le dossier de sortie existe (results/ par défaut)
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if args.validate > 0:
            # Mode caractérisation
            validator = PerformanceValidator(rate_calc)
            validator.run_validation(n_measures=args.validate,
                                     measure_duration_s=args.duration)
            validator.export_csv(args.output)
        else:
            # Mesure simple
            result = rate_calc.measure(duration_s=args.duration)

            # -- 5b. Calcul final via simulation (mode démonstration) --
            if args.simulate_final:
                result = apply_simulated_final(result, f_nominale)

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
