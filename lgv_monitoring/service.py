import csv
import io
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from .config import LGV_COORDINATES_LATLON, RiverPointConfig
from .database import get_connection
from .geo import build_lgv_line, point_distance_to_line_km


class LGVMonitoringService:
    def __init__(self, db_path: Path, river_points: List[RiverPointConfig]):
        self.db_path = db_path
        self.lgv_line = build_lgv_line(LGV_COORDINATES_LATLON)
        self.river_points = river_points
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "LGV-SEA-Monitor/4.0 (+local dashboard)",
                "Accept": "application/json,text/csv,*/*",
            }
        )
        self.hubeau_base = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"

    def _now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def _log_run(self, status: str, message: str) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                "INSERT INTO collection_runs(run_time_utc, status, message) VALUES (?, ?, ?)",
                (self._now_utc().isoformat(), status, message),
            )

    def _synop_url_for_date(self, d: datetime) -> str:
        return (
            "https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/"
            f"synop.{d.strftime('%Y%m%d')}.csv"
        )

    def fetch_and_store_pluviometry(self, max_distance_km: float) -> pd.DataFrame:
        candidates = [self._now_utc(), self._now_utc() - timedelta(days=1)]
        last_error: Optional[Exception] = None

        for candidate in candidates:
            url = self._synop_url_for_date(candidate)
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code != 200:
                    continue

                reader = csv.DictReader(io.StringIO(response.text), delimiter=";")
                rows = []
                for row in reader:
                    try:
                        # SYNOP lat/lon frequently stored in centidegrees
                        lat = float(row.get("Latitude", "")) / 100.0
                        lon = float(row.get("Longitude", "")) / 100.0
                        station_id = row.get("numer_sta")
                        date_raw = row.get("date")
                        rr = row.get("rr3") or row.get("rr6") or row.get("rr12") or row.get("rr24") or "0"
                        precipitation_mm = float(rr) if rr not in ("mq", "") else 0.0
                        obs_time = pd.to_datetime(date_raw, format="%Y%m%d%H%M%S", utc=True, errors="coerce")
                        if station_id is None or pd.isna(obs_time):
                            continue

                        distance_km = point_distance_to_line_km(lat, lon, self.lgv_line)
                        if distance_km <= max_distance_km:
                            rows.append(
                                {
                                    "station_id": station_id,
                                    "obs_time_utc": obs_time.to_pydatetime().isoformat(),
                                    "precipitation_mm": precipitation_mm,
                                    "distance_to_lgv_km": round(distance_km, 3),
                                    "latitude": lat,
                                    "longitude": lon,
                                    "source": "synop_meteofrance",
                                }
                            )
                    except Exception:
                        continue

                if not rows:
                    # Continue to yesterday fallback if today has no station in corridor.
                    continue

                with get_connection(self.db_path) as conn:
                    now_iso = self._now_utc().isoformat()
                    for item in rows:
                        conn.execute(
                            """
                            INSERT INTO pluvio_stations(station_id, latitude, longitude, source, updated_at)
                            VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT(station_id) DO UPDATE SET
                                latitude = excluded.latitude,
                                longitude = excluded.longitude,
                                source = excluded.source,
                                updated_at = excluded.updated_at
                            """,
                            (
                                item["station_id"],
                                item["latitude"],
                                item["longitude"],
                                item["source"],
                                now_iso,
                            ),
                        )
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO pluvio_observations(
                                station_id, obs_time_utc, precipitation_mm, distance_to_lgv_km, source, created_at_utc
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                item["station_id"],
                                item["obs_time_utc"],
                                item["precipitation_mm"],
                                item["distance_to_lgv_km"],
                                item["source"],
                                now_iso,
                            ),
                        )
                return pd.DataFrame(rows)
            except Exception as exc:
                last_error = exc

        if last_error:
            logging.warning("SYNOP unavailable: %s", last_error)
        return pd.DataFrame()

    def _fetch_hydro_station_obs(self, station_code: str, hours: int = 24) -> pd.DataFrame:
        url = f"{self.hubeau_base}/observations_tr"
        now = self._now_utc()
        start = now - timedelta(hours=hours)
        params = {
            "code_entite": station_code,
            "grandeur_hydro": "H",
            "size": 500,
            "sort": "desc",
            "date_debut_obs": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "date_fin_obs": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        response = self.session.get(url, params=params, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"HubEau observations HTTP {response.status_code}")

        rows = []
        for item in response.json().get("data", []):
            obs_time = pd.to_datetime(item.get("date_obs"), utc=True, errors="coerce")
            result = item.get("resultat_obs")
            if pd.isna(obs_time) or result is None:
                continue
            try:
                rows.append(
                    {
                        "obs_time_utc": obs_time.to_pydatetime().isoformat(),
                        "level_m": float(result) / 1000.0,
                        "status": item.get("statut_observation"),
                    }
                )
            except Exception:
                continue
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df.sort_values("obs_time_utc", ascending=False).reset_index(drop=True)

    def fetch_and_store_hydrometry(self) -> Dict[str, str]:
        status: Dict[str, str] = {}
        with get_connection(self.db_path) as conn:
            rivers = conn.execute(
                """
                SELECT river_id, river, station_code
                FROM river_points
                WHERE active = 1
                """
            ).fetchall()

            now_iso = self._now_utc().isoformat()
            for river in rivers:
                river_id = river["river_id"]
                river_name = river["river"]
                station_code = river["station_code"]
                if not station_code:
                    status[river_name] = "missing_station_code"
                    continue
                try:
                    df = self._fetch_hydro_station_obs(station_code=station_code)
                    if df.empty:
                        status[river_name] = "no_data"
                        continue
                    for _, row in df.iterrows():
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO hydro_observations(
                                river_id, station_code, obs_time_utc, level_m, status, source, created_at_utc
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                river_id,
                                station_code,
                                row["obs_time_utc"],
                                row["level_m"],
                                row["status"],
                                "hubeau_hydrometrie",
                                now_iso,
                            ),
                        )
                    status[river_name] = "ok"
                except Exception as exc:
                    status[river_name] = f"error:{exc}"
        return status

    def collect_once(self, max_station_distance_km: float) -> Dict[str, object]:
        try:
            pluv_df = self.fetch_and_store_pluviometry(max_station_distance_km=max_station_distance_km)
            hydro_status = self.fetch_and_store_hydrometry()
            message = (
                f"pluvio_rows={len(pluv_df)}, hydro={hydro_status}"
            )
            self._log_run("ok", message)
            return {"ok": True, "pluvio_rows": len(pluv_df), "hydro_status": hydro_status}
        except Exception as exc:
            self._log_run("error", str(exc))
            return {"ok": False, "error": str(exc)}

    def get_map_payload(self, pluvio_hours: int = 24, hydro_hours: int = 48) -> Dict[str, object]:
        with get_connection(self.db_path) as conn:
            pluvio = conn.execute(
                """
                WITH latest AS (
                    SELECT station_id, MAX(obs_time_utc) AS max_obs
                    FROM pluvio_observations
                    WHERE obs_time_utc >= datetime('now', ?)
                    GROUP BY station_id
                )
                SELECT p.station_id, p.obs_time_utc, p.precipitation_mm, p.distance_to_lgv_km,
                       s.latitude, s.longitude
                FROM pluvio_observations p
                JOIN latest l
                  ON l.station_id = p.station_id AND l.max_obs = p.obs_time_utc
                JOIN pluvio_stations s
                  ON s.station_id = p.station_id
                ORDER BY p.distance_to_lgv_km
                """,
                (f"-{pluvio_hours} hours",),
            ).fetchall()

            rivers = conn.execute(
                """
                SELECT rp.river_id, rp.river, rp.name, rp.latitude, rp.longitude,
                       rp.station_code, rp.threshold_m, rp.rapid_rise_mph
                FROM river_points rp
                WHERE rp.active = 1
                ORDER BY rp.river
                """
            ).fetchall()

            hydro_details = {}
            for river in rivers:
                obs = conn.execute(
                    """
                    SELECT obs_time_utc, level_m, status
                    FROM hydro_observations
                    WHERE river_id = ?
                      AND obs_time_utc >= datetime('now', ?)
                    ORDER BY obs_time_utc DESC
                    """,
                    (river["river_id"], f"-{hydro_hours} hours"),
                ).fetchall()
                hydro_details[river["river"]] = [dict(x) for x in obs]

            last_runs = conn.execute(
                """
                SELECT run_time_utc, status, message
                FROM collection_runs
                ORDER BY id DESC
                LIMIT 20
                """
            ).fetchall()

        return {
            "pluvio_latest": [dict(x) for x in pluvio],
            "rivers": [dict(x) for x in rivers],
            "hydro_details": hydro_details,
            "last_runs": [dict(x) for x in last_runs],
        }

