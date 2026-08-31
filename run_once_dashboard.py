import json
import logging
import os
import signal
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from meteo_test import LGVSeaMonitor


class TimeoutException(Exception):
    pass


@contextmanager
def timeout_handler(seconds: int, task_name: str = "task"):
    """Context manager pour gérer les timeouts."""
    def signal_handler(signum, frame):
        raise TimeoutException(f"{task_name} dépassé le timeout de {seconds}s")
    
    # Sauvegarder le signal handler précédent
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    except TimeoutException:
        logging.warning(f"Timeout: {task_name}")
        raise
    finally:
        signal.alarm(0)  # Désactiver l'alarme
        signal.signal(signal.SIGALRM, old_handler)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    max_total_seconds = int(os.getenv("LGV_MAX_CYCLE_SECONDS", "720"))  # 12 minutes par défaut
    logging.info(f"Démarrage du cycle LGV (timeout total: {max_total_seconds}s)")
    
    try:
        with timeout_handler(max_total_seconds, "Cycle LGV complet"):
            monitor = LGVSeaMonitor()
            # Reduce timeouts for faster collection in CI environments
            monitor.hydro_network_hours = min(monitor.hydro_network_hours, 24)
            monitor.synop_cache_ttl_hours = min(monitor.synop_cache_ttl_hours, 6)
            
            monitor.run_cycle()
            logging.info("✓ Cycle LGV terminé avec succès")
            return
    except TimeoutException as e:
        logging.error(f"✗ Cycle LGV annulé: {e}")
        _create_fallback_snapshot("timeout")
        sys.exit(1)
    except Exception as e:
        logging.error(f"✗ Erreur lors du cycle LGV: {e}", exc_info=True)
        _create_fallback_snapshot("error")
        sys.exit(1)


def _create_fallback_snapshot(error_type: str = "unknown") -> None:
    """Créer un snapshot minimaliste si le cycle échoue."""
    os.makedirs("reports", exist_ok=True)
    
    fallback_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "risk_level": "INDETERMINE",
        "score": 0.0,
        "weather_notice": f"Donnees indisponibles (cycle interruption: {error_type})",
        "alerts": [],
        "recommendations": ["Vérifier les logs du cycle de monitoring"],
        "details": {"status": error_type},
        "map_path": None,
        "lgv_lines": [],
        "weather": [],
        "rivers": [],
        "geotech": {},
        "piezometers": {},
        "hydro_network": {},
        "sectors": {},
        "lgv_communes": {},
        "fr_geography": {"communes_geojson": {"type": "FeatureCollection", "features": []}},
        "metadata": {},
        "commune_ranking": []
    }
    
    latest_path = os.path.join("reports", "streamlit_snapshot_latest.json")
    try:
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(fallback_data, f, ensure_ascii=False, indent=2, default=str)
        logging.info(f"✓ Snapshot de fallback créé: {latest_path}")
    except Exception as e:
        logging.error(f"✗ Erreur lors de la création du fallback: {e}")


if __name__ == "__main__":
    main()
