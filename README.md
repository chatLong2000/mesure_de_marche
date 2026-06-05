# Mesure de marche horlogère

Mesure automatisée de la **marche** d'un mouvement mécanique (écart de cadence
en secondes par jour) par **stroboscopie optique**. Une caméra filme le
balancier éclairé par un flash dont la fréquence est pilotée logiciellement ;
l'analyse du mouvement apparent permet de remonter à la fréquence réelle du
balancier, puis à la marche.

Le projet fournit deux interfaces :

- **un script en ligne de commande** (`mesure_marche.py`) pour lancer une mesure ;
- **une application web Flask** (`app.py`) pour piloter les mesures et visualiser
  l'historique des résultats.

---

## Principe

En stroboscopie, la fréquence apparente du balancier observée à l'image vaut :

$$f_{app} = \lvert f_{balancier} - f_{flash} \rvert$$

En connaissant la fréquence du flash (`f_flash`, imposée par le flasher) et en
mesurant la fréquence apparente, on en déduit la fréquence réelle du balancier,
puis la marche par rapport à la fréquence nominale du calibre :

$$\text{marche}\;[s/j] = \frac{f_{réelle} - f_{nominale}}{f_{nominale}} \times 86400$$

La tolérance de référence COSC pour un chronomètre mécanique est de **−4 / +6 s/jour**.

---

## Architecture

```
mesure_marche.py        Script principal (CLI) : synchro, mesure, calcul, validation
app.py                  Application web Flask (pilotage + visualisation)
templates/index.html    Interface web
results/                Résultats de mesure (CSV)
captures/               Frames sauvegardées (debug, synchro)

src/
├── models.py           Data classes et constantes horlogères
├── cameras/            Pilotes caméra interchangeables
│   ├── aravis_camera.py        SVS EXO273CGE via Aravis / GObject (macOS)
│   ├── harvester_camera.py     GigE via GenICam (Linux / Raspberry Pi)
│   └── dummy_camera.py         Caméra factice (tests sans matériel)
├── hardware/
│   └── flasher.py      Pilotage du flasher stroboscopique (série UART)
├── processing/
│   ├── signal_classifier.py    Classification du signal du balancier
│   ├── synchronizer.py         Synchronisation automatique du saut
│   ├── rate_calculator.py      Calcul de la marche (f_réelle → s/j)
│   └── validator.py            Comparaison à un étalon Witschi
└── simulation/
    └── simulation_marche_parfaite.py   Cas idéal simulé / rejeu hors-ligne
```

---

## Matériel

- **Caméra** : SVS EXO273CGE (GigE Vision) ou toute caméra GenICam compatible.
- **Flasher stroboscopique** piloté par liaison série UART.

Le projet fonctionne aussi **sans matériel** grâce à la caméra factice (`dummy`)
et au port série factice (`test`), utiles pour le développement et les démos.

---

## Installation

L'environnement de référence est un environnement conda nommé `ContrHorlo`.

```bash
conda create -n ContrHorlo python=3.11
conda activate ContrHorlo
pip install -r requirements.txt
```

### Pilotes caméra (optionnels selon la plateforme)

- **macOS (Aravis)** : installer Aravis via Homebrew puis exposer le typelib GI :

  ```bash
  brew install aravis
  export GI_TYPELIB_PATH="/opt/homebrew/lib/girepository-1.0"
  ```

- **Linux / Raspberry Pi (Harvester / GenICam)** : installer `harvesters` et
  fournir un fichier `.cti` (GenICam Transport Layer) via l'option `--cti`.

---

## Utilisation - ligne de commande

```bash
conda activate ContrHorlo
export GI_TYPELIB_PATH="/opt/homebrew/lib/girepository-1.0"   # macOS / Aravis

# Mesure avec matériel réel (détection automatique du module caméra)
python mesure_marche.py /dev/ttyUSB0

# Forcer un module caméra
python mesure_marche.py /dev/ttyUSB0 --camera aravis
python mesure_marche.py /dev/ttyUSB0 --camera harvester --cti /chemin/GigETL.cti

# Mode test, sans aucun matériel
python mesure_marche.py test
```

### Principales options

| Option | Défaut | Description |
| --- | --- | --- |
| `port` | - | Port série du flasher (`/dev/ttyUSB0`, `COM3`, ou `test`) |
| `--camera` | `auto` | `aravis`, `harvester`, `dummy` ou `auto` |
| `--cti` | - | Fichier `.cti` pour le module Harvester |
| `--calibre` | `28800` | Fréquence nominale en A/h (18000, 21600, 25200, 28800, 36000) |
| `--target` | `balance` | Cible filmée : `balance` (balancier) ou `seconds` (aiguille) |
| `--trig-off` | `250000` | Période entre triggers du flash en µs |
| `--duration` | `10` | Durée de mesure en secondes |
| `--validate N` | `0` | Mode caractérisation : N mesures comparatives |
| `--skip-sync` | - | Passer la phase de synchronisation automatique |
| `--show-preview` | - | Afficher les frames en temps réel (OpenCV) |
| `--debug-save` | - | Sauvegarder les frames de synchro dans `captures/debug_sync/` |
| `--output` | `results/mesure_results.csv` | Fichier CSV de sortie |

Les résultats sont ajoutés au fichier CSV (`results/mesure_results.csv` par défaut).

---

## Utilisation - application web

```bash
conda activate ContrHorlo
python app.py
```

Ouvrir ensuite [http://127.0.0.1:5000](http://127.0.0.1:5000).

L'interface permet de :

- lancer / arrêter une mesure (le script `mesure_marche.py` est exécuté en
  sous-processus dans l'environnement conda `ContrHorlo`) ;
- suivre les logs en temps réel ;
- consulter, filtrer et supprimer l'historique des résultats.

### API HTTP

| Méthode | Route | Rôle |
| --- | --- | --- |
| `GET` | `/` | Page principale |
| `GET` | `/api/results` | Liste des résultats (JSON) |
| `POST` | `/api/start` | Lancer une mesure |
| `POST` | `/api/stop` | Arrêter la mesure en cours |
| `GET` | `/api/status` | État et logs de la mesure |
| `DELETE` | `/api/delete/<idx>` | Supprimer un résultat |

---

## Format des résultats

Chaque mesure est enregistrée dans le CSV avec les colonnes suivantes :

`timestamp, f_flash, f_app, f_reelle, f_nominale, ecart_hz, marche_s_j,
classe_dom, nb_images, trig_off, confidence`

---

## Documentation

- `docs/`- cahier des charges et rapport technique.
- `presentation/` - powerpoint.
