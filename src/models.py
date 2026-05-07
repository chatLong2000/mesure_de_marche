"""
Data classes et constantes partagées par les modules d'automatisation.
"""

from dataclasses import dataclass


# ===========================================================================
#  CONSTANTES HORLOGÈRES
# ===========================================================================

FREQ_NOMINALES = {
    "18000": 2.5,     # 18 000 A/h  →  2.5 Hz
    "21600": 3.0,     # 21 600 A/h  →  3.0 Hz
    "25200": 3.5,     # 25 200 A/h  →  3.5 Hz
    "28800": 4.0,     # 28 800 A/h  →  4.0 Hz
    "36000": 5.0,     # 36 000 A/h  →  5.0 Hz
}

SECONDS_PER_DAY = 86400


# ===========================================================================
#  DATA CLASSES
# ===========================================================================

@dataclass
class MeasureResult:
    """Résultat d'une mesure de marche."""
    timestamp: str
    f_flash_hz: float           # Fréquence du flash stroboscopique
    f_apparente_hz: float       # Fréquence apparente mesurée
    f_reelle_hz: float          # Fréquence réelle du balancier
    f_nominale_hz: float        # Fréquence nominale du calibre
    ecart_hz: float             # Écart f_réelle - f_nominale
    marche_s_par_jour: float    # Marche en secondes/jour
    classe_dominante: int       # Classe dominante dans la séquence
    nb_images: int              # Nombre d'images analysées
    trig_off_us: int            # Valeur trig_off utilisée
    confidence: float           # Indice de confiance (0-1)


@dataclass
class WitschiComparison:
    """Résultat de comparaison avec un étalon Witschi."""
    timestamp: str
    marche_optique_s_j: float
    marche_witschi_s_j: float
    ecart_s_j: float
    ecart_relatif_pct: float
