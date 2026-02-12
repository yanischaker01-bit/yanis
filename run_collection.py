import argparse
import time

from lgv_monitoring.config import DB_PATH, RIVER_POINTS
from lgv_monitoring.database import init_db
from lgv_monitoring.service import LGVMonitoringService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LGV SEA data collection loop.")
    parser.add_argument("--interval-min", type=int, default=60, help="Collection interval in minutes.")
    parser.add_argument(
        "--max-distance-km",
        type=float,
        default=25.0,
        help="Maximum station distance to LGV for pluvio collection.",
    )
    parser.add_argument("--once", action="store_true", help="Run one cycle and stop.")
    args = parser.parse_args()

    init_db(DB_PATH, RIVER_POINTS)
    service = LGVMonitoringService(DB_PATH, RIVER_POINTS)

    if args.once:
        result = service.collect_once(max_station_distance_km=args.max_distance_km)
        print(result)
        return

    interval_seconds = max(args.interval_min, 1) * 60
    while True:
        result = service.collect_once(max_station_distance_km=args.max_distance_km)
        print(result)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()

