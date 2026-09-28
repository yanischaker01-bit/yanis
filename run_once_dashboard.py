import json
import logging
import os
import signal
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

import requests

from meteo_test import LGVSeaMonitor


class TimeoutException(Exception):
    pass


@contextmanager
def timeout_handler(seconds: int, task_name: str = "task"):
    """Context manager pour gérer les timeouts."""
    def signal_handler(signum, frame):
        raise TimeoutException(f"{task_name} dépassé le timeout de {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    except TimeoutException:
        logging.warning(f"Timeout: {task_name}")
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# --- Validation utilities ---------------------------------------------------
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


def normalize_key(k: str) -> str:
    k = str(k or "").strip().lower()
    k = k.replace(" ", "_").replace("-", "_").replace(".", "_")
    return ALIASES.get(k, k)


def extract_sectors(snapshot: dict) -> list:
    if not isinstance(snapshot, dict):
        return []
    container = snapshot.get("sectors")
    if isinstance(container, dict):
        for candidate in ("sectors", "records", "items", "features"):
            if candidate in container and isinstance(container[candidate], list):
                return container[candidate]
        # If container itself is a list-like under an unexpected key, try to decode
        if isinstance(container.get("data"), list):
            return container.get("data")
        # If container appears to be already a list-like dict keys->values, bail out
        return []
    if isinstance(container, list):
        return container
    # fallback: some snapshots might put sectors at root keys like 'data' or 'items' or 'features'
    for candidate in ("sectors", "records", "items", "features", "data", "snapshot"):
        v = snapshot.get(candidate)
        if isinstance(v, list):
            return v
    return []


def sector_has_required(sector: dict) -> tuple[bool, set]:
    keys = {normalize_key(k) for k in sector.keys()} if isinstance(sector, dict) else set()
    present = set()
    # Check for latitude/longitude existence including GeoJSON Point handling
    lat_ok = any(k in keys for k in ("latitude",))
    lon_ok = any(k in keys for k in ("longitude",))
    pk_ok = any(k in keys for k in ("pk_km",))

    # GeoJSON feature: properties + geometry
    if not lat_ok or not lon_ok:
        # try geometry.coordinates pattern
        geom = sector.get("geometry") if isinstance(sector, dict) else None
        if isinstance(geom, dict) and isinstance(geom.get("coordinates"), (list, tuple)):
            coords = geom.get("coordinates")
            if len(coords) >= 2:
                lon_ok = True
                lat_ok = True
                keys.update({"longitude", "latitude"})
    # pk_m acceptance
    if not pk_ok and "pk_m" in keys:
        pk_ok = True

    # commune name
    commune_ok = any(k in keys for k in ("commune_name",))

    present = {k for k in keys if k in REQUIRED_FIELDS}
    return (lat_ok and lon_ok and pk_ok and commune_ok), present


def count_exploitable_sectors(sectors: list) -> tuple[int, int, set]:
    total = 0
    usable = 0
    available_keys = set()
    for s in sectors:
        if not isinstance(s, dict):
            continue
        total += 1
        ok, present = sector_has_required(s)
        if ok:
            usable += 1
        available_keys.update({normalize_key(k) for k in s.keys()})
    return total, usable, available_keys


def is_snapshot_valid(snapshot: dict) -> tuple[bool, str, int, int, set]:
    sectors = extract_sectors(snapshot)
    if not isinstance(sectors, list) or len(sectors) == 0:
        return False, "aucun secteur dans sectors", 0, 0, set()
    total, usable, available = count_exploitable_sectors(sectors)
    if total == 0:
        return False, "aucun secteur détecté", 0, 0, available
    if usable == 0:
        return False, "aucun secteur exploitable (colonnes manquantes)", total, usable, available
    # All good
    return True, "ok", total, usable, available


def read_json_file(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def fetch_json_url(url: str, timeout: int = 15) -> Optional[dict]:
    try:
        headers = {"Accept": "application/json", "Cache-Control": "no-cache", "Pragma": "no-cache"}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def find_last_valid_snapshot() -> tuple[Optional[dict], Optional[str]]:
    # 1. local latest
    candidates = [
        os.path.join("reports", "streamlit_snapshot_latest.json"),
        os.path.join("reports", "streamlit_snapshot_last_valid.json"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            snap = read_json_file(p)
            if snap is None:
                logging.debug(f"Impossible de lire {p}")
                continue
            valid, reason, total, usable, available = is_snapshot_valid(snap)
            if valid:
                logging.info(f"Dernier snapshot local valide trouvé: {p} ({total} secteurs, {usable} exploitables)")
                return snap, p
            else:
                logging.info(f"Snapshot local {p} invalide: {reason}")
    # 2. remote candidates
    for url in REMOTE_SNAPSHOT_URLS:
        snap = fetch_json_url(url)
        if snap is None:
            logging.debug(f"Impossible de récupérer {url}")
            continue
        valid, reason, total, usable, available = is_snapshot_valid(snap)
        if valid:
            logging.info(f"Dernier snapshot distant valide trouvé: {url} ({total} secteurs, {usable} exploitables)")
            return snap, url
        else:
            logging.info(f"Snapshot distant {url} invalide: {reason}")
    return None, None


def atomic_write_bytes(target_path: str, data: bytes) -> None:
    tmp = target_path + ".tmp"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target_path)


def atomic_write_json(target_path: str, obj: dict) -> None:
    tmp = target_path + ".tmp"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target_path)


def restore_snapshot(source_snap: dict, source_location: str, target_path: str) -> None:
    logging.info("Restauration du dernier snapshot valide")
    # If source is URL, write downloaded dict; if source is file path we can copy content
    if source_location and source_location.startswith("http"):
        atomic_write_json(target_path, source_snap)
    else:
        # read raw bytes from source file and write atomically
        with open(source_location, "rb") as rf:
            data = rf.read()
        atomic_write_bytes(target_path, data)
    logging.info("Dernier snapshot valide restauré")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    max_total_seconds = int(os.getenv("LGV_MAX_CYCLE_SECONDS", "720"))
    logging.info(f"Démarrage du cycle LGV (timeout total: {max_total_seconds}s)")

    # Find the last valid snapshot before attempting a new run
    last_snap, last_location = find_last_valid_snapshot()
    if last_snap is None:
        logging.warning("Aucun snapshot valide préalable trouvé. Le cycle peut produire le premier snapshot valide.")
    else:
        logging.info("Un snapshot valide préalable a été trouvé et conservé en mémoire.")

    monitor = LGVSeaMonitor()
    # Reduce timeouts for faster collection in CI environments
    monitor.hydro_network_hours = min(getattr(monitor, "hydro_network_hours", 24), 24)
    monitor.synop_cache_ttl_hours = min(getattr(monitor, "synop_cache_ttl_hours", 6), 6)

    latest_path = os.path.join("reports", "streamlit_snapshot_latest.json")

    try:
        with timeout_handler(max_total_seconds, "Cycle LGV complet"):
            monitor.run_cycle()
            logging.info("✓ Cycle LGV terminé — validation du snapshot généré")
    except TimeoutException as e:
        logging.error(f"✗ Cycle LGV annulé: {e}")
        # Do not create fallback empty snapshot. Restore last valid if any.
        if last_snap is not None:
            restore_snapshot(last_snap, last_location, latest_path)
            logging.error("Cycle interrompu : snapshot précédent restauré. Fin avec erreur.")
        else:
            # remove potentially partial latest
            if os.path.isfile(latest_path):
                os.remove(latest_path)
            logging.error("Aucun snapshot valide disponible après timeout. Aucun fichier publié.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"✗ Erreur lors du cycle LGV: {e}", exc_info=True)
        if last_snap is not None:
            restore_snapshot(last_snap, last_location, latest_path)
            logging.error("Cycle en erreur : snapshot précédent restauré. Fin avec erreur.")
        else:
            if os.path.isfile(latest_path):
                os.remove(latest_path)
            logging.error("Aucun snapshot valide disponible après erreur. Aucun fichier publié.")
        sys.exit(1)

    # After successful run, validate the newly generated snapshot
    new_snap = read_json_file(latest_path)
    if new_snap is None:
        logging.error("Snapshot généré introuvable ou illisible. Tentative de restauration.")
        if last_snap is not None:
            restore_snapshot(last_snap, last_location, latest_path)
        sys.exit(1)

    valid, reason, total, usable, available = is_snapshot_valid(new_snap)
    if valid:
        logging.info(f"Snapshot valide : {total} secteurs, {usable} exploitables")
        # copy to last_valid atomically
        try:
            atomic_write_json(os.path.join("reports", "streamlit_snapshot_last_valid.json"), new_snap)
            logging.info("Copie du snapshot valide vers reports/streamlit_snapshot_last_valid.json")
            sys.exit(0)
        except Exception as e:
            logging.error(f"✗ Erreur lors de la sauvegarde du dernier snapshot valide : {e}")
            sys.exit(1)
    else:
        logging.error(f"Snapshot généré invalide : {reason}")
        logging.error(f"Colonnes disponibles (exemples): {sorted(list(available))}")
        # restore last valid if available
        if last_snap is not None:
            restore_snapshot(last_snap, last_location, latest_path)
            logging.error("Dernier snapshot valide restauré")
        else:
            # remove invalid latest
            if os.path.isfile(latest_path):
                os.remove(latest_path)
            logging.error("Aucun snapshot valide disponible — aucun fichier publié")
        sys.exit(1)


if __name__ == "__main__":
    main()
