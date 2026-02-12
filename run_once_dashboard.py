from meteo_test import LGVSeaMonitor


def main() -> None:
    monitor = LGVSeaMonitor()
    monitor.run_cycle()


if __name__ == "__main__":
    main()
