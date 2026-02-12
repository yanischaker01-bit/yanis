from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class RiverPointConfig:
    river: str
    name: str
    latitude: float
    longitude: float
    station_code: Optional[str]
    threshold_m: Optional[float]
    rapid_rise_mph: float


LGV_COORDINATES_LATLON: List[Tuple[float, float]] = [
    (44.8378, -0.5792),  # Bordeaux
    (45.7167, 0.3667),   # Angouleme (approx)
    (46.3167, 0.4667),   # Poitiers
    (47.3833, 0.6833),   # Tours
]

# Use 1 km as requested by default; can be overridden in UI.
DEFAULT_STATION_MAX_DISTANCE_KM = 1.0

RIVER_POINTS: List[RiverPointConfig] = [
    RiverPointConfig(
        river="Charente",
        name="Charente - secteur OA LGV",
        latitude=45.75,
        longitude=0.10,
        station_code=None,
        threshold_m=2.0,
        rapid_rise_mph=0.12,
    ),
    RiverPointConfig(
        river="Dordogne",
        name="Dordogne - secteur OA LGV",
        latitude=44.90,
        longitude=-0.25,
        station_code=None,
        threshold_m=3.0,
        rapid_rise_mph=0.15,
    ),
    RiverPointConfig(
        river="Vienne",
        name="Vienne - secteur OA LGV",
        latitude=46.82,
        longitude=0.55,
        station_code=None,
        threshold_m=2.5,
        rapid_rise_mph=0.12,
    ),
    RiverPointConfig(
        river="Auxance",
        name="Auxance - secteur OA LGV",
        latitude=46.65,
        longitude=0.30,
        station_code=None,
        threshold_m=1.6,
        rapid_rise_mph=0.10,
    ),
    RiverPointConfig(
        river="Manse",
        name="Manse - secteur OA LGV",
        latitude=47.15,
        longitude=0.65,
        station_code=None,
        threshold_m=1.4,
        rapid_rise_mph=0.10,
    ),
]

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "lgv_monitoring.db"

