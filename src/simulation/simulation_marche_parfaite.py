#!/usr/bin/env python3
"""
Calcul de la marche à partir d'images déjà classées en classe 3
================================================================

Hypothèse de travail : les images fournies en entrée sont **déjà
classées en classe 3** (= saut stroboscopique), et leur **timestamp**
est lisible depuis leur nom de fichier au format
``..._YYYYMMDD_HHMMSS_mmm.<ext>``.

La vérité-terrain n'est pas connue : on se contente d'estimer la marche
à partir des intervalles réels entre sauts, sans la comparer à une
valeur de référence.

Pipeline
--------
  1. Lister et trier les images du dossier d'entrée.
  2. Extraire les timestamps depuis les noms de fichiers.
  3. Estimer la fréquence apparente comme l'inverse de la **médiane**
     des intervalles entre sauts consécutifs.
  4. En déduire ``f_réelle = f_flash ± f_app`` puis la marche en s/j :
     ``(f_réelle - f_nominale) / f_nominale * 86400``.

Exemple
-------
    python src/simulation/simulation_marche_parfaite.py captures/class3_sync \\
        --calibre 28800 --f-flash 3.75
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from datetime import datetime
from typing import List, Tuple

import numpy as np

if __package__ in (None, ""):
    # Exécution autonome : `python src/simulation/simulation_marche_parfaite.py`
    # → ajoute la racine du projet au PYTHONPATH pour résoudre `src`.
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )
    from src.models import FREQ_NOMINALES, SECONDS_PER_DAY
else:
    from ..models import FREQ_NOMINALES, SECONDS_PER_DAY


# ---------------------------------------------------------------------------
# Chargement des images de classe 3 + extraction des timestamps
# ---------------------------------------------------------------------------

# Motif attendu dans le nom de fichier : ...YYYYMMDD_HHMMSS_mmm...
_TS_RE = re.compile(r"(\d{8})_(\d{6})_(\d{3})")


def _parse_timestamp(path: str) -> float:
    """Retourne le timestamp Unix (en s) extrait du nom de fichier."""
    m = _TS_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Aucun timestamp dans {path!r} "
                         f"(motif attendu YYYYMMDD_HHMMSS_mmm)")
    date_s, time_s, ms_s = m.groups()
    dt = datetime.strptime(date_s + time_s, "%Y%m%d%H%M%S")
    return dt.timestamp() + int(ms_s) / 1000.0


def load_class3_images(folder: str) -> Tuple[List[str], List[float]]:
    """Retourne (chemins triés par timestamp, timestamps relatifs en s)."""
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    paths: List[str] = []
    for p in patterns:
        paths.extend(glob.glob(os.path.join(folder, p)))
    if not paths:
        raise FileNotFoundError(f"Aucune image trouvée dans {folder}")

    pairs = [(_parse_timestamp(p), p) for p in paths]
    pairs.sort()
    timestamps = [t for t, _ in pairs]
    sorted_paths = [p for _, p in pairs]

    t0 = timestamps[0]
    timestamps = [t - t0 for t in timestamps]

    print(f"[INFO] {len(sorted_paths)} images de classe 3 trouvées "
          f"dans {folder} (durée totale {timestamps[-1]:.3f} s)")
    return sorted_paths, timestamps


# ---------------------------------------------------------------------------
# Génération synthétique de timestamps en situation parfaite
# ---------------------------------------------------------------------------

def generate_ideal_jump_times(f_flash_hz: float,
                              f_balancier_hz: float,
                              n_sauts: int,
                              jitter_s: float = 0.0,
                              quantize_to_flash: bool = True,
                              seed: int = None) -> Tuple[List[float], float, int]:
    """
    Génère une série de timestamps idéaux pour ``n_sauts`` sauts
    successifs séparés de ``T_app = 1 / |f_balancier - f_flash|``.

    Options :
      * ``quantize_to_flash`` : arrondit chaque instant au multiple
        entier de ``T_flash`` le plus proche (simule la quantification
        d'échantillonnage par le flash, comportement réel du pipeline).
      * ``jitter_s`` : écart-type d'un bruit gaussien ajouté à chaque
        timestamp (simule l'incertitude de datation).
      * ``seed`` : graine pour la reproductibilité.

    Retourne (jump_times, T_app_theo, direction).
    """
    f_app = abs(f_balancier_hz - f_flash_hz)
    if f_app < EPS:
        raise ValueError("f_balancier ≈ f_flash → pas de cycle apparent.")
    T_app = 1.0 / f_app
    T_flash = 1.0 / f_flash_hz
    direction = 1 if f_balancier_hz >= f_flash_hz else -1

    rng = np.random.default_rng(seed)
    ts = np.arange(n_sauts, dtype=float) * T_app
    if quantize_to_flash:
        ts = np.round(ts / T_flash) * T_flash
    if jitter_s > 0.0:
        ts = ts + rng.normal(0.0, jitter_s, size=ts.shape)
        ts = np.sort(ts)

    return ts.tolist(), T_app, direction


# ---------------------------------------------------------------------------
# Calcul de la marche
# ---------------------------------------------------------------------------

EPS = 1e-9


def measure_apparent_frequency(jump_times: List[float]) -> float:
    """
    Inverse de la médiane des intervalles entre sauts consécutifs.

    Filtre adaptatif en deux passes (sans constante magique) :
      1. médiane brute T0 sur tous les intervalles strictement positifs,
      2. on ne garde que ceux dans [0.5·T0, 1.5·T0]
         (élimine les intervalles « à trou » k·T_app, k ≥ 2, ainsi que
         les artefacts < T_app/2),
      3. médiane finale sur ce sous-ensemble.
    """
    if len(jump_times) < 2:
        return 0.0

    periods = np.diff(np.asarray(jump_times, dtype=float))
    periods = periods[periods > EPS]
    if periods.size == 0:
        return 0.0

    T0 = float(np.median(periods))
    if T0 <= EPS:
        return 0.0

    kept = periods[(periods >= 0.5 * T0) & (periods <= 1.5 * T0)]
    if kept.size == 0:
        return 0.0

    mean_period = float(np.median(kept))
    if mean_period <= EPS:
        return 0.0
    return 1.0 / mean_period


def compute_marche(jump_times: List[float],
                   f_flash_hz: float,
                   f_nominale_hz: float,
                   direction: int) -> dict:
    """Reproduit le calcul de marche de ``RateCalculator``."""
    f_app = measure_apparent_frequency(jump_times)
    f_reelle = f_flash_hz + direction * f_app
    ecart = f_reelle - f_nominale_hz
    marche = (ecart / f_nominale_hz) * SECONDS_PER_DAY
    return {
        "f_app_hz": f_app,
        "direction": direction,
        "f_reelle_hz": f_reelle,
        "ecart_hz": ecart,
        "marche_s_j": marche,
    }


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def print_report(args, jump_times: List[float], res: dict):
    print()
    print("=" * 64)
    if args.simulate:
        print("  CALCUL DE MARCHE — TIMESTAMPS SIMULÉS (SITUATION PARFAITE)")
    else:
        print("  CALCUL DE MARCHE — IMAGES CLASSE 3 PRÉ-CLASSÉES")
    print("=" * 64)
    print(f"  Calibre               : {args.calibre} A/h")
    print(f"  f_nominale            : {args.f_nominale:.6f} Hz")
    print(f"  f_flash (consigne)    : {args.f_flash:.6f} Hz")
    if args.simulate:
        print(f"  Marche vraie simulée  : {args.marche_vraie:+.4f} s/j")
        print(f"  f_balancier vraie     : {args._f_balancier:.6f} Hz")
        print(f"  T_app théorique       : {1.0 / abs(args._f_balancier - args.f_flash):.6f} s")
        print(f"  Quantif. flash        : {'oui' if args.quantize else 'non'}")
        if args.jitter > 0:
            print(f"  Jitter ajouté (σ)     : {args.jitter*1000:.2f} ms")
    print(f"  Sens supposé          : "
          f"{'forward (+)' if res['direction'] > 0 else 'backward (-)'}")
    print(f"  Sauts utilisés        : {len(jump_times)}")
    if len(jump_times) >= 2:
        diffs = np.diff(jump_times)
        print(f"  Intervalle médian     : {float(np.median(diffs)):.4f} s")
        print(f"  Intervalle min / max  : "
              f"{float(np.min(diffs)):.4f} / {float(np.max(diffs)):.4f} s")
    print()
    print("  ────────────── RÉSULTAT ──────────────")
    print(f"  f_apparente mesurée   : {res['f_app_hz']:.6f} Hz")
    print(f"  f_réelle              : {res['f_reelle_hz']:.6f} Hz")
    print(f"  Écart f               : {res['ecart_hz']:+.6e} Hz")
    print(f"  ╔══════════════════════════════════════════╗")
    print(f"  ║  MARCHE MESURÉE  : {res['marche_s_j']:+10.4f} s/j   ║")
    if args.simulate:
        err = res['marche_s_j'] - args.marche_vraie
        print(f"  ║  MARCHE VRAIE    : {args.marche_vraie:+10.4f} s/j   ║")
        print(f"  ║  ERREUR          : {err:+10.4f} s/j   ║")
    print(f"  ╚══════════════════════════════════════════╝")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calcul de marche à partir d'images déjà classées 3.")
    p.add_argument("input_dir", nargs="?", default=None,
                   help="Dossier contenant les images de classe 3 "
                        "(non requis si --simulate).")
    p.add_argument("--calibre", default="28800",
                   choices=list(FREQ_NOMINALES.keys()),
                   help="Calibre en A/h (défaut: 28800 → 4 Hz)")
    p.add_argument("--f-flash", type=float, default=None,
                   help="Fréquence du flash en Hz "
                        "(défaut: f_nominale - 0.25)")
    p.add_argument("--backward", action="store_true",
                   help="Sens de rotation apparente inversé "
                        "(f_réelle = f_flash - f_app au lieu de +).")

    sim = p.add_argument_group("Mode simulation parfaite")
    sim.add_argument("--simulate", action="store_true",
                     help="Ignore input_dir et génère une série idéale "
                          "de timestamps à partir de --marche-vraie.")
    sim.add_argument("--marche-vraie", type=float, default=5.0,
                     help="Marche vérité-terrain en s/j à simuler "
                          "(défaut: +5.0)")
    sim.add_argument("--n-sauts", type=int, default=60,
                     help="Nombre de sauts à générer (défaut: 60)")
    sim.add_argument("--no-quantize", dest="quantize",
                     action="store_false",
                     help="Désactive la quantification à T_flash "
                          "(par défaut activée : chaque saut est arrondi "
                          "au multiple de T_flash le plus proche).")
    sim.add_argument("--jitter", type=float, default=0.0,
                     help="Écart-type (s) d'un bruit gaussien ajouté "
                          "à chaque timestamp simulé (défaut: 0).")
    sim.add_argument("--seed", type=int, default=None,
                     help="Graine RNG pour le jitter (défaut: aléatoire)")

    args = p.parse_args()

    args.f_nominale = FREQ_NOMINALES[args.calibre]
    # f_balancier vraie déduite de la marche cible (utile en mode --simulate)
    args._f_balancier = args.f_nominale * (1.0 + args.marche_vraie / SECONDS_PER_DAY)
    if args.f_flash is None:
        # En simulation on cale f_flash pour avoir f_app ≈ 0.25 Hz ;
        # en mode dossier on reste sur f_nominale - 0.25.
        if args.simulate:
            args.f_flash = args._f_balancier - 0.25
        else:
            args.f_flash = args.f_nominale - 0.25
    args.direction = -1 if args.backward else 1
    return args


def main() -> int:
    args = parse_args()

    if args.simulate:
        jump_times, T_app, direction = generate_ideal_jump_times(
            f_flash_hz=args.f_flash,
            f_balancier_hz=args._f_balancier,
            n_sauts=args.n_sauts,
            jitter_s=args.jitter,
            quantize_to_flash=args.quantize,
            seed=args.seed,
        )
        args.direction = direction  # on connaît le bon sens en simulation
        print(f"[INFO] {len(jump_times)} timestamps simulés "
              f"(T_app théorique = {T_app:.4f} s, "
              f"durée totale = {jump_times[-1]:.3f} s)")
        
        #print all jump times for debug
        print("[DEBUG] jump_times:")
        for i, t in enumerate(jump_times):
            print(f"  {i+1:02d}: {t:.4f} s")
            
    else:
        if not args.input_dir:
            print("[ERR] input_dir requis (ou utiliser --simulate).",
                  file=sys.stderr)
            return 1
        if not os.path.isdir(args.input_dir):
            print(f"[ERR] Dossier introuvable : {args.input_dir}",
                  file=sys.stderr)
            return 1
        _, jump_times = load_class3_images(args.input_dir)

    res = compute_marche(
        jump_times=jump_times,
        f_flash_hz=args.f_flash,
        f_nominale_hz=args.f_nominale,
        direction=args.direction,
    )

    print_report(args, jump_times, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
