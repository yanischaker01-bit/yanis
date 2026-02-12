import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from .config import RiverPointConfig


def get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Path, river_points: Iterable[RiverPointConfig]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS pluvio_stations (
                station_id TEXT PRIMARY KEY,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                source TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pluvio_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id TEXT NOT NULL,
                obs_time_utc TEXT NOT NULL,
                precipitation_mm REAL NOT NULL,
                distance_to_lgv_km REAL NOT NULL,
                source TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                UNIQUE(station_id, obs_time_utc),
                FOREIGN KEY(station_id) REFERENCES pluvio_stations(station_id)
            );

            CREATE TABLE IF NOT EXISTS river_points (
                river_id INTEGER PRIMARY KEY AUTOINCREMENT,
                river TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                station_code TEXT,
                threshold_m REAL,
                rapid_rise_mph REAL NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS hydro_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                river_id INTEGER NOT NULL,
                station_code TEXT NOT NULL,
                obs_time_utc TEXT NOT NULL,
                level_m REAL NOT NULL,
                status TEXT,
                source TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                UNIQUE(river_id, obs_time_utc),
                FOREIGN KEY(river_id) REFERENCES river_points(river_id)
            );

            CREATE TABLE IF NOT EXISTS collection_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_time_utc TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT
            );
            """
        )

        for rp in river_points:
            conn.execute(
                """
                INSERT INTO river_points (
                    river, name, latitude, longitude, station_code, threshold_m, rapid_rise_mph, active
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(river) DO UPDATE SET
                    name = excluded.name,
                    latitude = excluded.latitude,
                    longitude = excluded.longitude,
                    threshold_m = excluded.threshold_m,
                    rapid_rise_mph = excluded.rapid_rise_mph
                """,
                (
                    rp.river,
                    rp.name,
                    rp.latitude,
                    rp.longitude,
                    rp.station_code,
                    rp.threshold_m,
                    rp.rapid_rise_mph,
                ),
            )


def set_station_code(db_path: Path, river: str, station_code: Optional[str]) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE river_points SET station_code = ? WHERE river = ?",
            (station_code, river),
        )

