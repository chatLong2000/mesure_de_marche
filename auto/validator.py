"""
Caractérisation des performances par comparaison avec un étalon Witschi.
"""

import csv
from datetime import datetime

import numpy as np

from .models import MeasureResult, WitschiComparison
from .rate_calculator import RateCalculator


class PerformanceValidator:
    """
    Outil de caractérisation des performances par comparaison
    avec un étalon acoustique (type Witschi).

    Workflow :
    ──────────
    1. Effectuer N mesures optiques consécutives
    2. L'utilisateur saisit la valeur Witschi correspondante
    3. Calculer l'écart, la moyenne, l'écart-type
    4. Exporter les résultats en CSV pour analyse statistique
    """

    def __init__(self, rate_calculator: RateCalculator):
        self.calc = rate_calculator
        self.comparisons: list[WitschiComparison] = []
        self.optical_measurements: list[MeasureResult] = []

    def run_validation(self, n_measures: int = 10, measure_duration_s: float = 10.0):
        """Effectue n_measures mesures et compare avec la saisie Witschi."""
        print("\n" + "=" * 60)
        print("  CARACTÉRISATION — Comparaison avec étalon Witschi")
        print("=" * 60)
        print(f"  Nombre de mesures : {n_measures}")
        print(f"  Durée par mesure  : {measure_duration_s} s")
        print()

        for i in range(n_measures):
            print(f"\n{'─' * 40}")
            print(f"  Mesure {i + 1}/{n_measures}")
            print(f"{'─' * 40}")

            # Mesure optique
            result = self.calc.measure(duration_s=measure_duration_s, verbose=False)
            self.optical_measurements.append(result)

            marche_optique = result.marche_s_par_jour
            print(f"  Marche optique : {marche_optique:+.2f} s/j")

            # Saisie Witschi
            print()
            witschi_str = input(f"  → Valeur Witschi (s/j) [Entrée pour passer] : ").strip()

            if witschi_str:
                try:
                    marche_witschi = float(witschi_str)
                    ecart = marche_optique - marche_witschi
                    ecart_rel = (ecart / marche_witschi * 100) if marche_witschi != 0 else float('inf')

                    comp = WitschiComparison(
                        timestamp=datetime.now().isoformat(),
                        marche_optique_s_j=marche_optique,
                        marche_witschi_s_j=marche_witschi,
                        ecart_s_j=ecart,
                        ecart_relatif_pct=ecart_rel,
                    )
                    self.comparisons.append(comp)
                    print(f"  Écart : {ecart:+.2f} s/j ({ecart_rel:+.1f}%)")
                except ValueError:
                    print("  [WARN] Valeur invalide, mesure Witschi ignorée.")

        self._print_summary()

    def _print_summary(self):
        """Affiche le résumé statistique de la validation."""
        print("\n" + "=" * 60)
        print("  RÉSUMÉ CARACTÉRISATION")
        print("=" * 60)

        if self.optical_measurements:
            marches = [m.marche_s_par_jour for m in self.optical_measurements]
            print(f"\n  Mesures optiques ({len(marches)} mesures) :")
            print(f"    Moyenne  : {np.mean(marches):+.2f} s/j")
            print(f"    Écart-type: {np.std(marches):.2f} s/j")
            print(f"    Min      : {min(marches):+.2f} s/j")
            print(f"    Max      : {max(marches):+.2f} s/j")

        if self.comparisons:
            ecarts = [c.ecart_s_j for c in self.comparisons]
            print(f"\n  Comparaisons Witschi ({len(ecarts)} paires) :")
            print(f"    Écart moyen          : {np.mean(ecarts):+.2f} s/j")
            print(f"    Écart-type des écarts: {np.std(ecarts):.2f} s/j")
            print(f"    Écart max            : {max(abs(e) for e in ecarts):.2f} s/j")

            # Incertitude élargie (k=2, ~95% de confiance)
            u = np.std(ecarts) * 2
            print(f"    Incertitude (k=2)    : ±{u:.2f} s/j")

    def export_csv(self, filepath: str = "validation_results.csv"):
        """Exporte les résultats en CSV."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Mesures optiques
            writer.writerow(["=== MESURES OPTIQUES ==="])
            writer.writerow(["timestamp", "f_flash_hz", "f_app_hz", "f_reelle_hz",
                             "f_nominale_hz", "ecart_hz", "marche_s_j",
                             "classe_dom", "nb_images", "trig_off_us", "confidence"])
            for m in self.optical_measurements:
                writer.writerow([m.timestamp, m.f_flash_hz, m.f_apparente_hz,
                                 m.f_reelle_hz, m.f_nominale_hz, m.ecart_hz,
                                 m.marche_s_par_jour, m.classe_dominante,
                                 m.nb_images, m.trig_off_us, m.confidence])

            writer.writerow([])
            writer.writerow(["=== COMPARAISONS WITSCHI ==="])
            writer.writerow(["timestamp", "marche_optique_s_j", "marche_witschi_s_j",
                             "ecart_s_j", "ecart_relatif_pct"])
            for c in self.comparisons:
                writer.writerow([c.timestamp, c.marche_optique_s_j,
                                 c.marche_witschi_s_j, c.ecart_s_j,
                                 c.ecart_relatif_pct])

        print(f"\n  Résultats exportés → {filepath}")
