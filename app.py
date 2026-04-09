#!/usr/bin/env python3
"""
Application web Flask pour la mesure de marche horlogère.
Affiche les résultats, lance des mesures et visualise les données.
"""

import csv
import os
import subprocess
import threading
from datetime import datetime
from dataclasses import dataclass

from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__)

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "mesure_results.csv")
SCRIPT_PATH = os.path.join(BASE_DIR, "mesure_marche.py")

# Fréquences nominales
FREQ_NOMINALES = {
    "18000": 2.5,
    "21600": 3.0,
    "25200": 3.5,
    "28800": 4.0,
    "36000": 5.0,
}

# État global de la mesure en cours
measurement_state = {
    "running": False,
    "log": [],
    "process": None,
}
state_lock = threading.Lock()


def read_csv_results():
    """Lit les résultats depuis le fichier CSV."""
    results = []
    if not os.path.exists(CSV_FILE):
        return results
    with open(CSV_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                results.append({
                    "timestamp": row.get("timestamp", ""),
                    "f_flash": float(row.get("f_flash", 0)),
                    "f_app": float(row.get("f_app", 0)),
                    "f_reelle": float(row.get("f_reelle", 0)),
                    "f_nominale": float(row.get("f_nominale", 0)),
                    "ecart_hz": float(row.get("ecart_hz", 0)),
                    "marche_s_j": float(row.get("marche_s_j", 0)),
                    "classe_dom": int(row.get("classe_dom", -1)),
                    "nb_images": int(row.get("nb_images", 0)),
                    "trig_off": int(row.get("trig_off", 0)),
                    "confidence": float(row.get("confidence", 0)),
                })
            except (ValueError, TypeError):
                continue
    return results


@app.route("/")
def index():
    results = read_csv_results()
    return render_template("index.html", results=results, calibres=FREQ_NOMINALES)


@app.route("/api/results")
def api_results():
    """Renvoie les résultats en JSON."""
    return jsonify(read_csv_results())


@app.route("/api/start", methods=["POST"])
def api_start_measure():
    """Lance une mesure via le script mesure_marche.py."""
    with state_lock:
        if measurement_state["running"]:
            return jsonify({"error": "Une mesure est déjà en cours."}), 409

    data = request.get_json(silent=True) or {}
    port = data.get("port", "test")
    calibre = data.get("calibre", "28800")
    duration = data.get("duration", 10)
    trig_off = data.get("trig_off", 250000)
    skip_sync = data.get("skip_sync", False)

    # Validation
    if calibre not in FREQ_NOMINALES:
        return jsonify({"error": f"Calibre invalide: {calibre}"}), 400
    try:
        duration = float(duration)
        trig_off = int(trig_off)
    except (ValueError, TypeError):
        return jsonify({"error": "Paramètres numériques invalides"}), 400
    if not (0.1 <= duration <= 300):
        return jsonify({"error": "Durée doit être entre 0.1 et 300 s"}), 400
    if not (1000 <= trig_off <= 1000000):
        return jsonify({"error": "trig_off doit être entre 1000 et 1000000 µs"}), 400

    # Construire la commande (exécutée dans l'env conda ContrHorlo)
    conda_env = data.get("conda_env", "ContrHorlo")
    cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_env,
        "python", SCRIPT_PATH,
        port,
        "--calibre", calibre,
        "--duration", str(duration),
        "--trig-off", str(trig_off),
    ]
    if skip_sync:
        cmd.append("--skip-sync")

    def run_measurement():
        with state_lock:
            measurement_state["running"] = True
            measurement_state["log"] = []

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=BASE_DIR,
            )
            with state_lock:
                measurement_state["process"] = proc

            for line in proc.stdout:
                with state_lock:
                    measurement_state["log"].append(line.rstrip())

            proc.wait()
        except Exception as e:
            with state_lock:
                measurement_state["log"].append(f"[ERREUR] {e}")
        finally:
            with state_lock:
                measurement_state["running"] = False
                measurement_state["process"] = None

    thread = threading.Thread(target=run_measurement, daemon=True)
    thread.start()

    return jsonify({"status": "started", "command": " ".join(cmd)})


@app.route("/api/stop", methods=["POST"])
def api_stop_measure():
    """Arrête la mesure en cours."""
    with state_lock:
        proc = measurement_state.get("process")
        if proc and proc.poll() is None:
            proc.terminate()
            return jsonify({"status": "stopped"})
        return jsonify({"error": "Aucune mesure en cours."}), 404


@app.route("/api/status")
def api_status():
    """Renvoie l'état de la mesure en cours."""
    with state_lock:
        return jsonify({
            "running": measurement_state["running"],
            "log": measurement_state["log"],
        })


@app.route("/api/delete/<int:idx>", methods=["DELETE"])
def api_delete_result(idx):
    """Supprime une ligne du CSV par son index."""
    results = read_csv_results()
    if idx < 0 or idx >= len(results):
        return jsonify({"error": "Index invalide"}), 404

    results.pop(idx)

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "f_flash", "f_app", "f_reelle",
                         "f_nominale", "ecart_hz", "marche_s_j",
                         "classe_dom", "nb_images", "trig_off", "confidence"])
        for r in results:
            writer.writerow([
                r["timestamp"], r["f_flash"], r["f_app"], r["f_reelle"],
                r["f_nominale"], r["ecart_hz"], r["marche_s_j"],
                r["classe_dom"], r["nb_images"], r["trig_off"], r["confidence"],
            ])
    return jsonify({"status": "deleted"})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
