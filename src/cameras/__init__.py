"""Modules caméra interchangeables pour la mesure de marche.

Deux pilotes matériels sont disponibles :
  - :class:`AravisCamera`    — via Aravis / GObject Introspection (macOS)
  - :class:`HarvesterCamera` — via Harvester / GenICam (Linux, Raspberry Pi)

Plus une caméra factice :class:`DummyCamera` pour les tests sans matériel.

Les imports des pilotes matériels sont protégés : si la dépendance
(Aravis ou harvesters) est absente, le symbole correspondant vaut ``None``
et le drapeau ``*_AVAILABLE`` est ``False``.
"""

from .dummy_camera import DummyCamera

try:
    from .aravis_camera import AravisCamera, ARAVIS_AVAILABLE
except Exception:  # pragma: no cover - dépendance optionnelle
    AravisCamera = None
    ARAVIS_AVAILABLE = False

try:
    from .harvester_camera import HarvesterCamera
    HARVESTER_AVAILABLE = True
except Exception:  # pragma: no cover - dépendance optionnelle
    HarvesterCamera = None
    HARVESTER_AVAILABLE = False

__all__ = [
    "DummyCamera",
    "AravisCamera",
    "HarvesterCamera",
    "ARAVIS_AVAILABLE",
    "HARVESTER_AVAILABLE",
]
