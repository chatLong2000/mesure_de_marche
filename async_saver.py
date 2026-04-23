"""
Sauvegarde asynchrone d'images.

Utilise un ThreadPoolExecutor pour découpler l'écriture disque
de la boucle d'acquisition caméra / flash, évitant tout
ralentissement du pipeline temps-réel.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

# Pool partagé — 2 threads suffisent pour absorber les pics d'écriture
_executor = ThreadPoolExecutor(max_workers=2)


def save_image_async(path: str, image: np.ndarray) -> None:
    """Soumet l'écriture d'une image au pool de threads."""
    # Copier l'image pour éviter toute corruption si le buffer est réutilisé
    img_copy = image.copy()
    _executor.submit(_write_image, path, img_copy)


def _write_image(path: str, image: np.ndarray) -> None:
    """Écrit l'image sur disque (exécuté dans un thread worker)."""
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        cv2.imwrite(path, image)
    except Exception as e:
        print(f"[ASYNC SAVE ERR] {path}: {e}")


def shutdown_saver() -> None:
    """Attend la fin de toutes les écritures en cours."""
    _executor.shutdown(wait=True)
