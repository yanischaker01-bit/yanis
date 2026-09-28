import json
import logging
import os
import signal
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from meteo_test import LGVSeaMonitor


class TimeoutException(Exception):
    pass


@contextmanager
def timeout_handler(seconds: int, task_name: str = "task"):
    def signal_handler(signum, frame):
        raise TimeoutException(f"{task_name} dépassé le timeout de {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    except TimeoutException:
        logging.warning("Timeout: %s", task_name)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


ALIASES = {
    "commune": "commune_name",
    "nom_commune": "commune_name",
    "libelle_commune": "commune_name",
    "lat": "latitude",
    "y": "latitude",
    "latitude_wgs84": "latitude",
    "lon": "longitude",
    "lng": "longitude",
    "long": "longitude",
    "x": "longitude",
    "longitude_wgs84": "longitude",
    "pk": "pk_km",
    "pk_gps": "pk_km",
    "pk_theorique": "pk_km",
    "pk_m": "pk_km",
}
REQUIRED_FIELDS = {"commune_name", "latitude", "longitude", "pk_km"}
REMOTE_SNAPSHOT_URLS = [
    "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json",
    "https://yanischaker01-bit.github.io/yanis/dashboard/reports/streamlit_snapshot_latest.json",
]


def normalize_key(k: Any) -> str:
    key = str(k or "").strip().lower()
    key = key.replace(" ", "_").replace("-", "_").replace(".", "_")
    return ALIASES.get(key, key)


def extract_sectors(snapshot: Any) -> list:
    if not isinstance(snapshot, dict):
        return []
    container = snapshot.get("sectors")
    if isinstance(container, dict):
        for candidate in ("sectors", "records", "items", "features"):
            val = container.get(candidate)
            if isinstance(val, list):
                return val
        if isinstance(container.get("data"), list):
            return container["data"]
        return []
    if isinstance(container, list):
        return container
    for candidate in ("sectors", "records", "items", "features", "data", "snapshot"):
        val = snapshot.get(candidate)
        if isinstance(val, list):
            return val
    return []


def _coerce_geometry_point(sector: dict) -> None:
    if not isinstance(sector, dict):
        return
    geom = sector.get("geometry")
    if not isinstance(geom, dict):
        return
    coords = geom.get("coordinates")
    if geom.get("type") == "Point" and isinstance(coords, (list, tuple)) and len(coords) >= 2:
        sector.setdefault("longitude", coords[0])
        sector.setdefault("latitude", coords[1])


def sector_has_required(sector: dict) -> tuple[bool, set]:
    if not isinstance(sector, dict):
        return False, set()
    _coerce_geometry_point(sector)
    keys = {normalize_key(k) for k in sector.keys()}
    if "pk_m" in keys and "pk_km" not in keys:
        keys.add("pk_km")
    has_commune = "commune_name" in keys
    has_lat = "latitude" in keys
    has_lon = "longitude" in keys
    has_pk = "pk_km" in keys
    return has_commune and has_lat and has_lon and has_pk, keys


def count_exploitable_sectors(sectors: list) -> tuple[int, int, set]:
    total = 0
    usable = 0
    available_keys: set[str] = set()
    for sector in sectors:
        if not isinstance(sector, dict):
            continue
        total += 1
        ok, keys = sector_has_required(sector)
        if ok:
            usable += 1
        available_keys.update(keys)
    return total, usable, available_keys


def is_snapshot_valid(snapshot: Any) -> tuple[bool, str, int, int, set]:
    sectors = extract_sectors(snapshot)
    if not isinstance(sectors, list) or len(sectors) == 0:
        return False, "aucun secteur dans sectors", 0, 0, set()
    total, usable, available = count_exploitable_sectors(sectors)
    if total == 0:
        return False, "aucun secteur détecté", 0, 0, available
    if usable == 0:
        return False, "aucun secteur exploitable (colonnes manquantes)", total, usable, available
    for sector in sectors:
        if not isinstance(sector, dict):
            continue
        for field in ("latitude", "longitude", "pk_km"):
            if field in sector:
                try:
                    float(sector[field])
                except (TypeError, ValueError):
                    return False, f"valeur non numérique pour {field}", total, usable, available
    return True, "ok", total, usable, available


def read_json_file(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def fetch_json_url(url: str, timeout: int = 15) -> Optional[dict]:
    try:
        headers = {"Accept": "application/json", "Cache-Control": "no-cache", "Pragma": "no-cache"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def find_last_valid_snapshot() -> tuple[Optional[dict], Optional[str]]:
    candidates = [
        os.path.join("reports", "streamlit_snapshot_latest.json"),
        os.path.join("reports", "streamlit_snapshot_last_valid.json"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        payload = read_json_file(path)
        if payload is None:
            continue
        valid, reason, total, usable, _ = is_snapshot_valid(payload)
        if valid:
            logging.info("Snapshot local valide trouvé: %s (%s secteurs, %s exploitables)", path, total, usable)
            return payload, path
        logging.warning("Snapshot local invalide ignoré: %s (%s)", path, reason)
    for url in REMOTE_SNAPSHOT_URLS:
        payload = fetch_json_url(url)
        if payload is None:
            continue
        valid, reason, total, usable, _ = is_snapshot_valid(payload)
        if valid:
            logging.info("Snapshot distant valide trouvé: %s (%s secteurs, %s exploitables)", url, total, usable)
            return payload, url
        logging.warning("Snapshot distant invalide ignoré: %s (%s)", url, reason)
    return None, None


def atomic_write_json(target_path: str, obj: Any) -> None:
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)


def restore_snapshot(snapshot: Optional[dict], source: Optional[str], target_path: str) -> None:
    if snapshot is None:
        if os.path.exists(target_path):
            os.remove(target_path)
        logging.error("Aucun snapshot valide disponible ; fichier latest supprimé")
        return
    atomic_write_json(target_path, snapshot)
    logging.info("Dernier snapshot valide restauré")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    max_total_seconds = int(os.getenv("LGV_MAX_CYCLE_SECONDS", "720"))
    logging.info("Démarrage du cycle LGV (timeout total: %ss)", max_total_seconds)

    last_snapshot, last_source = find_last_valid_snapshot()
    latest_path = os.path.join("reports", "streamlit_snapshot_latest.json")

    try:
        with timeout_handler(max_total_seconds, "Cycle LGV complet"):
            monitor = LGVSeaMonitor()
            monitor.run_cycle()
        logging.info("Cycle LGV terminé")
    except TimeoutException as exc:
        logging.error("Cycle LGV annulé: %s", exc)
        if last_snapshot is not None:
            restore_snapshot(last_snapshot, last_source, latest_path)
        else:
            if os.path.exists(latest_path):
                os.remove(latest_path)
        sys.exit(1)
    except Exception as exc:
        logging.error("Erreur lors du cycle LGV: %s", exc, exc_info=True)
        if last_snapshot is not None:
            restore_snapshot(last_snapshot, last_source, latest_path)
        else:
            if os.path.exists(latest_path):
                os.remove(latest_path)
        sys.exit(1)

    new_snapshot = read_json_file(latest_path)
    if new_snapshot is None:
        logging.error("Snapshot généré introuvable ou illisible")
        if last_snapshot is not None:
            restore_snapshot(last_snapshot, last_source, latest_path)
        else:
            if os.path.exists(latest_path):
                os.remove(latest_path)
        sys.exit(1)

    valid, reason, total, usable, available = is_snapshot_valid(new_snapshot)
    if valid:
        logging.info("Snapshot valide : %s secteurs, %s exploitables", total, usable)
        atomic_write_json(os.path.join("reports", "streamlit_snapshot_last_valid.json"), new_snapshot)
        sys.exit(0)

    logging.error("Snapshot généré invalide : %s", reason)
    logging.error("Colonnes disponibles : %s", sorted(available))
    if last_snapshot is not None:
        restore_snapshot(last_snapshot, last_source, latest_path)
        logging.info("Dernier snapshot valide restauré")
    else:
        if os.path.exists(latest_path):
            os.remove(latest_path)
        logging.error("Aucun snapshot valide disponible ; fichier latest supprimé")
    sys.exit(1)


if __name__ == "__main__":
    main()
