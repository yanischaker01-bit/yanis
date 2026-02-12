import csv
import html
import json
import logging
import math
import os
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from io import StringIO
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import schedule


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("lgv_monitoring.log"), logging.StreamHandler()],
)


class DataSource(Enum):
    SYNOP_METEOFRANCE = "synop_meteofrance"
    OPEN_METEO = "open_meteo"
    HUBEAU_HYDRO = "hubeau_hydrometrie"
    VIGICRUES_HYDRO = "vigicrues_hubeau"


@dataclass
class RiverMonitoringPoint:
    river: str
    name: str
    latitude: float
    longitude: float
    station_code: Optional[str] = None
    threshold_m: Optional[float] = None
    rapid_rise_mph: float = 0.10


class LGVSeaMonitor:
    def __init__(self):
        self.default_lgv_coordinates_latlon = [
            (44.8378, -0.5792),
            (45.7167, 0.3667),
            (46.3167, 0.4667),
            (47.3833, 0.6833),
        ]
        self.lgv_geometry_candidates = [
            os.path.join("lgv_monitoring", "assets", "lgv_sea_line.kml"),
            os.path.join("lgv_monitoring", "assets", "LRS_AXES.kmz"),
            r"C:\Users\YCHAKER\Downloads\kmz\LRS_AXES.kmz",
        ]
        self.lgv_lines_latlon = self._load_lgv_lines(self.lgv_geometry_candidates)
        if not self.lgv_lines_latlon:
            self.lgv_lines_latlon = [self.default_lgv_coordinates_latlon]
        self.weather_corridor_km = 1.0
        self.alert_thresholds_mm = {
            "catastrophique_24h": 120.0,
            "extreme_24h": 80.0,
            "forte_24h": 50.0,
            "moderee_24h": 20.0,
        }
        self.soil_sample_step_km = 15.0
        self.soil_max_points = 36
        self.soil_mvt_radius_m = 3000
        self.soil_mvt_page_size = 20
        self.soil_cache_hours = 24
        self.piezometer_corridor_km = 5.0
        self.piezometer_max_stations = 25
        self.piezometer_history_days = 90
        self.piezometer_cache_hours = 6
        self.hydro_network_corridor_km = 8.0
        self.hydro_network_max_stations = 35
        self.hydro_network_hours = 48
        self.hydro_network_cache_hours = 6
        self.sector_length_km = 25.0
        self.sector_radius_km = 10.0

        self.river_points: List[RiverMonitoringPoint] = [
            RiverMonitoringPoint("Dordogne", "Dordogne - secteur OA LGV", 44.90, -0.25, None, 3.0, 0.15),
            RiverMonitoringPoint("Charente", "Charente - secteur Champniers", 45.72, 0.20, None, 2.0, 0.12),
            RiverMonitoringPoint("Vienne", "Vienne - secteur OA LGV", 46.82, 0.55, None, 2.5, 0.12),
            RiverMonitoringPoint("Auxance", "Auxance - secteur OA LGV", 46.65, 0.30, None, 1.6, 0.10),
            RiverMonitoringPoint("Manse", "Manse - secteur OA LGV", 47.15, 0.65, None, 1.4, 0.10),
        ]

        self.headers = {
            "User-Agent": "LGV-SEA-Monitor/4.0 (+contact: internal)",
            "Accept": "application/json",
            "Accept-Language": "fr-FR,fr;q=0.9",
        }
        self.hubeau_base = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"
        self.hubeau_endpoints = {"observations_tr": "/observations_tr", "stations": "/referentiel/stations"}
        self.open_meteo_base = "https://api.open-meteo.com/v1/forecast"
        self.open_meteo_sample_step_km = 5.0
        self.open_meteo_max_points = 80

        os.makedirs("data", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update(self.headers)

        self.station_cache_file = os.path.join("data", "river_station_cache.json")
        self.station_cache = self._load_station_cache()
        self.soil_cache_file = os.path.join("data", "soil_risk_cache.json")
        self.piezometer_cache_file = os.path.join("data", "piezometer_cache.json")
        self.hydro_network_cache_file = os.path.join("data", "hydro_network_cache.json")

    def _now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _safe_float(value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        txt = str(value).strip().replace(",", ".")
        if not txt or txt.lower() in {"mq", "nan", "none", "///"}:
            return None
        try:
            return float(txt)
        except ValueError:
            return None

    @staticmethod
    def _station_pluvio_class(rain_24h: float, rain_7d: Optional[float], rain_30d: Optional[float]) -> str:
        r7 = rain_7d if rain_7d is not None else rain_24h
        r30 = rain_30d if rain_30d is not None else r7
        if rain_24h >= 120 or r7 >= 180 or r30 >= 300:
            return "CRITIQUE"
        if rain_24h >= 80 or r7 >= 120 or r30 >= 220:
            return "ELEVE"
        if rain_24h >= 50 or r7 >= 80 or r30 >= 150:
            return "MODERE"
        if rain_24h >= 20 or r7 >= 40 or r30 >= 90:
            return "VIGILANCE"
        return "NORMAL"

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
        return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _point_to_segment_distance_km(p_lat: float, p_lon: float, a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
        mean_lat = math.radians((a_lat + b_lat + p_lat) / 3.0)

        def to_xy(lat: float, lon: float) -> Tuple[float, float]:
            return lon * 111.320 * math.cos(mean_lat), lat * 110.574

        px, py = to_xy(p_lat, p_lon)
        ax, ay = to_xy(a_lat, a_lon)
        bx, by = to_xy(b_lat, b_lon)
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab2 = abx * abx + aby * aby
        if ab2 == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
        cx, cy = ax + t * abx, ay + t * aby
        return math.hypot(px - cx, py - cy)

    def _point_to_lgv_distance_km(self, lat: float, lon: float) -> float:
        best = float("inf")
        for pts in self.lgv_lines_latlon:
            for i in range(len(pts) - 1):
                a_lat, a_lon = pts[i]
                b_lat, b_lon = pts[i + 1]
                best = min(best, self._point_to_segment_distance_km(lat, lon, a_lat, a_lon, b_lat, b_lon))
        return best

    def _sample_points_along_lgv(self, step_km: float, max_points: int) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        for line in self.lgv_lines_latlon:
            if len(line) < 2:
                continue
            for idx in range(len(line) - 1):
                a_lat, a_lon = line[idx]
                b_lat, b_lon = line[idx + 1]
                seg_km = max(self._haversine_km(a_lat, a_lon, b_lat, b_lon), 0.001)
                count = max(1, int(seg_km / max(step_km, 0.5)))
                for k in range(count):
                    t = k / count
                    lat = a_lat + (b_lat - a_lat) * t
                    lon = a_lon + (b_lon - a_lon) * t
                    points.append((lat, lon))
            points.append(line[-1])

        # Deduplicate lightly by rounded key.
        dedup = []
        seen = set()
        for lat, lon in points:
            key = (round(lat, 4), round(lon, 4))
            if key not in seen:
                seen.add(key)
                dedup.append((lat, lon))
        if not dedup:
            return []

        # Keep a manageable amount of Open-Meteo locations.
        if len(dedup) > max_points:
            stride = max(1, len(dedup) // max_points)
            dedup = dedup[::stride]
            if dedup[-1] != points[-1]:
                dedup.append(points[-1])
        return dedup[:max_points]

    def _bbox_from_lgv_lines(self, pad_deg: float = 0.08) -> Tuple[float, float, float, float]:
        points = [pt for line in self.lgv_lines_latlon for pt in line]
        if not points:
            return (-0.8, 44.7, 0.9, 47.5)
        lats = [lat for lat, _ in points]
        lons = [lon for _, lon in points]
        return (
            min(lons) - pad_deg,
            min(lats) - pad_deg,
            max(lons) + pad_deg,
            max(lats) + pad_deg,
        )

    def _load_fresh_cache(self, cache_path: str, max_age_hours: int) -> Optional[Dict[str, object]]:
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return None
            ts = pd.to_datetime(payload.get("timestamp_utc"), utc=True, errors="coerce")
            if pd.isna(ts):
                return None
            age_h = (self._now_utc() - ts.to_pydatetime()).total_seconds() / 3600.0
            if age_h > max_age_hours:
                return None
            return payload
        except Exception:
            return None

    def _fetch_rga_for_point(self, lat: float, lon: float) -> Dict[str, object]:
        try:
            response = self.session.get(
                "https://georisques.gouv.fr/api/v1/rga",
                params={"latlon": f"{lon:.6f},{lat:.6f}"},
                timeout=20,
            )
            if response.status_code != 200 or not response.text.strip():
                return {}
            payload = response.json()
            if not isinstance(payload, dict):
                return {}
            return payload
        except Exception:
            return {}

    def _fetch_mvt_for_point(self, lat: float, lon: float) -> Dict[str, object]:
        try:
            response = self.session.get(
                "https://georisques.gouv.fr/api/v1/mvt",
                params={
                    "latlon": f"{lon:.6f},{lat:.6f}",
                    "rayon": self.soil_mvt_radius_m,
                    "page_size": self.soil_mvt_page_size,
                },
                timeout=20,
            )
            if response.status_code != 200 or not response.text.strip():
                return {}
            payload = response.json()
            if not isinstance(payload, dict):
                return {}
            return payload
        except Exception:
            return {}

    @staticmethod
    def _classify_geotechnical_point(rga_code: Optional[int], mvt_count: int) -> Dict[str, object]:
        if mvt_count >= 4 or (rga_code == 3 and mvt_count >= 2):
            return {"risk_level": "CRITIQUE", "color": "#7f1d1d", "score": 4}
        if rga_code == 3 or mvt_count >= 1:
            return {"risk_level": "ELEVE", "color": "#b91c1c", "score": 3}
        if rga_code == 2:
            return {"risk_level": "MODERE", "color": "#d97706", "score": 2}
        if rga_code == 1:
            return {"risk_level": "FAIBLE", "color": "#16a34a", "score": 1}
        return {"risk_level": "INDETERMINE", "color": "#6b7280", "score": 1}

    def fetch_geotechnical_context(self) -> Dict[str, object]:
        cached = self._load_fresh_cache(self.soil_cache_file, self.soil_cache_hours)
        if cached:
            return cached

        sampled = self._sample_points_along_lgv(self.soil_sample_step_km, self.soil_max_points)
        points: List[Dict[str, object]] = []

        for idx, (lat, lon) in enumerate(sampled, start=1):
            rga = self._fetch_rga_for_point(lat, lon)
            mvt = self._fetch_mvt_for_point(lat, lon)

            code_raw = rga.get("codeExposition")
            try:
                rga_code = int(str(code_raw)) if code_raw is not None else None
            except Exception:
                rga_code = None
            rga_label = str(rga.get("exposition") or "Indetermine")

            mvt_count = 0
            mvt_data = []
            if isinstance(mvt, dict):
                try:
                    mvt_count = int(mvt.get("results") or 0)
                except Exception:
                    mvt_count = 0
                if isinstance(mvt.get("data"), list):
                    mvt_data = mvt.get("data") or []
            strong_count = sum(1 for item in mvt_data if str(item.get("fiabilite", "")).lower().startswith("fort"))
            top_types = []
            for item in mvt_data[:3]:
                t = item.get("type")
                if t:
                    top_types.append(str(t))

            cls = self._classify_geotechnical_point(rga_code, mvt_count)
            soil_type = "Sols non argileux dominants"
            if rga_code == 3:
                soil_type = "Argiles a forte exposition retrait-gonflement"
            elif rga_code == 2:
                soil_type = "Argiles a exposition moyenne retrait-gonflement"
            elif rga_code == 1:
                soil_type = "Argiles a faible exposition retrait-gonflement"

            points.append(
                {
                    "point_id": idx,
                    "latitude": round(float(lat), 6),
                    "longitude": round(float(lon), 6),
                    "rga_code": rga_code,
                    "rga_label": rga_label,
                    "soil_type": soil_type,
                    "mvt_count": int(mvt_count),
                    "mvt_strong_count": int(strong_count),
                    "mvt_top_types": top_types,
                    "risk_level": cls["risk_level"],
                    "risk_color": cls["color"],
                    "risk_score": cls["score"],
                }
            )

        critical_points = sum(1 for p in points if p["risk_level"] == "CRITIQUE")
        high_points = sum(1 for p in points if p["risk_level"] == "ELEVE")
        moderate_points = sum(1 for p in points if p["risk_level"] == "MODERE")
        low_points = sum(1 for p in points if p["risk_level"] == "FAIBLE")
        unknown_points = sum(1 for p in points if p["risk_level"] == "INDETERMINE")

        alerts = []
        hot = [p for p in points if p["risk_level"] in {"CRITIQUE", "ELEVE"}]
        hot = sorted(hot, key=lambda x: (x["risk_score"], x["mvt_count"], x.get("rga_code") or 0), reverse=True)
        for p in hot[:8]:
            alerts.append(
                {
                    "type": "GEOTECH",
                    "level": "ELEVE" if p["risk_level"] == "ELEVE" else "CRITIQUE",
                    "message": (
                        f"PK echantillon #{p['point_id']}: {p['soil_type']} | "
                        f"RGA={p['rga_label']} | MVT proches={p['mvt_count']}"
                    ),
                }
            )

        payload = {
            "timestamp_utc": self._now_utc().isoformat(),
            "sample_step_km": self.soil_sample_step_km,
            "sample_count": len(points),
            "source": "georisques_api_v1_rga_mvt",
            "summary": {
                "critical_points": critical_points,
                "high_points": high_points,
                "moderate_points": moderate_points,
                "low_points": low_points,
                "unknown_points": unknown_points,
            },
            "alerts": alerts,
            "points": points,
        }
        self._save_json(payload, self.soil_cache_file)
        ts = self._now_utc().strftime("%Y%m%d_%H%M%S")
        self._save_json(payload, os.path.join("data", f"soil_risk_{ts}.json"))
        return payload

    def _fetch_piezometer_chronicles(self, code_bss: str, days: int) -> pd.DataFrame:
        start = (self._now_utc() - timedelta(days=days)).strftime("%Y-%m-%d")
        response = self.session.get(
            "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques",
            params={"code_bss": code_bss, "date_debut_mesure": start, "sort": "desc", "size": 12},
            timeout=25,
        )
        if response.status_code not in {200, 206}:
            return pd.DataFrame()
        payload = response.json()
        rows = []
        for item in payload.get("data", []):
            dt = pd.to_datetime(item.get("date_mesure"), utc=True, errors="coerce")
            depth = self._safe_float(item.get("profondeur_nappe"))
            level = self._safe_float(item.get("niveau_nappe_eau"))
            if pd.isna(dt):
                continue
            if depth is None and level is None:
                continue
            rows.append(
                {
                    "date_mesure": dt,
                    "depth_m": depth,
                    "level_mngf": level,
                    "qualification": item.get("qualification"),
                    "statut": item.get("statut"),
                }
            )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values("date_mesure", ascending=False)
        return df

    @staticmethod
    def _classify_piezometer_risk(depth_m: Optional[float], trend_depth_mpd: Optional[float]) -> Dict[str, object]:
        if depth_m is None:
            risk = "INDETERMINE"
            color = "#6b7280"
        elif depth_m <= 1.0:
            risk = "TRES_ELEVE"
            color = "#dc2626"
        elif depth_m <= 2.0:
            risk = "ELEVE"
            color = "#ea580c"
        elif depth_m <= 3.0:
            risk = "MODERE"
            color = "#eab308"
        else:
            risk = "FAIBLE"
            color = "#2563eb"

        alerts = []
        if depth_m is not None and depth_m <= 1.5:
            alerts.append(f"Nappe tres proche du sol ({depth_m:.2f} m)")
        if trend_depth_mpd is not None and trend_depth_mpd <= -0.12:
            alerts.append(f"Remontee rapide de la nappe ({trend_depth_mpd:.2f} m/j)")
            if risk == "FAIBLE":
                risk, color = "MODERE", "#eab308"
            elif risk == "MODERE":
                risk, color = "ELEVE", "#ea580c"
        return {"risk_level": risk, "color": color, "alert_reasons": alerts}

    def fetch_piezometers_near_lgv(self) -> Dict[str, object]:
        cached = self._load_fresh_cache(self.piezometer_cache_file, self.piezometer_cache_hours)
        if cached:
            return cached

        min_lon, min_lat, max_lon, max_lat = self._bbox_from_lgv_lines(pad_deg=0.08)
        bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
        stations_raw: List[Dict[str, object]] = []
        page = 1
        while page <= 4:
            response = self.session.get(
                "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/stations",
                params={"bbox": bbox, "page": page, "size": 2000},
                timeout=25,
            )
            if response.status_code not in {200, 206}:
                break
            payload = response.json()
            rows = payload.get("data", [])
            if not rows:
                break
            stations_raw.extend(rows)
            if not payload.get("next"):
                break
            page += 1

        candidates = []
        for item in stations_raw:
            code_bss = item.get("code_bss")
            lat = self._safe_float(item.get("y"))
            lon = self._safe_float(item.get("x"))
            if not code_bss or lat is None or lon is None:
                continue
            dist = self._point_to_lgv_distance_km(float(lat), float(lon))
            if dist > self.piezometer_corridor_km:
                continue
            candidates.append(
                {
                    "code_bss": str(code_bss),
                    "name": str(item.get("libelle_pe") or code_bss),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "distance_to_lgv_km": round(float(dist), 3),
                }
            )

        candidates = sorted(candidates, key=lambda x: x["distance_to_lgv_km"])[: self.piezometer_max_stations]
        stations = []
        alerts = []
        for item in candidates:
            obs = self._fetch_piezometer_chronicles(item["code_bss"], self.piezometer_history_days)
            if obs.empty:
                continue

            latest = obs.iloc[0]
            oldest = obs.iloc[-1]
            latest_depth = latest.get("depth_m")
            latest_level = latest.get("level_mngf")
            dt_days = max((latest["date_mesure"] - oldest["date_mesure"]).total_seconds() / 86400.0, 0.0)

            trend_depth_mpd: Optional[float] = None
            if dt_days > 0 and pd.notna(latest_depth) and pd.notna(oldest.get("depth_m")):
                trend_depth_mpd = float(latest_depth - oldest.get("depth_m")) / dt_days

            cls = self._classify_piezometer_risk(latest_depth, trend_depth_mpd)
            station = {
                **item,
                "last_date_utc": latest["date_mesure"].isoformat(),
                "depth_m": None if pd.isna(latest_depth) else round(float(latest_depth), 3),
                "level_mngf": None if pd.isna(latest_level) else round(float(latest_level), 3),
                "trend_depth_mpd": None if trend_depth_mpd is None else round(float(trend_depth_mpd), 3),
                "risk_level": cls["risk_level"],
                "risk_color": cls["color"],
                "alert_reasons": cls["alert_reasons"],
                "n_obs": int(len(obs)),
                "source": "hubeau_niveaux_nappes",
            }
            stations.append(station)
            if station["alert_reasons"]:
                alerts.append(
                    {
                        "type": "NAPPE",
                        "level": "ELEVE" if station["risk_level"] in {"ELEVE", "TRES_ELEVE"} else "MODERE",
                        "message": f"{station['code_bss']} ({station['name']}): " + " | ".join(station["alert_reasons"]),
                    }
                )

        summary = {
            "stations_in_corridor": int(len(candidates)),
            "stations_with_data": int(len(stations)),
            "very_high_risk": int(sum(1 for s in stations if s["risk_level"] == "TRES_ELEVE")),
            "high_risk": int(sum(1 for s in stations if s["risk_level"] == "ELEVE")),
            "moderate_risk": int(sum(1 for s in stations if s["risk_level"] == "MODERE")),
            "rapid_rise_alerts": int(sum(1 for s in stations if s["trend_depth_mpd"] is not None and s["trend_depth_mpd"] <= -0.12)),
        }

        payload = {
            "timestamp_utc": self._now_utc().isoformat(),
            "corridor_km": self.piezometer_corridor_km,
            "max_stations": self.piezometer_max_stations,
            "source": "hubeau_niveaux_nappes",
            "summary": summary,
            "alerts": alerts[:10],
            "stations": stations,
        }
        self._save_json(payload, self.piezometer_cache_file)
        ts = self._now_utc().strftime("%Y%m%d_%H%M%S")
        self._save_json(payload, os.path.join("data", f"piezometers_{ts}.json"))
        return payload

    def _save_json(self, obj: Dict, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)

    def _load_station_cache(self) -> Dict[str, Dict[str, object]]:
        if not os.path.exists(self.station_cache_file):
            return {}
        try:
            with open(self.station_cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_station_cache(self) -> None:
        self._save_json(self.station_cache, self.station_cache_file)

    @staticmethod
    def _downsample_coords(coords: List[Tuple[float, float]], max_points: int = 3500) -> List[Tuple[float, float]]:
        if len(coords) <= max_points:
            return coords
        step = max(1, len(coords) // max_points)
        sampled = coords[::step]
        if sampled[-1] != coords[-1]:
            sampled.append(coords[-1])
        return sampled

    def _line_length_km(self, coords: List[Tuple[float, float]]) -> float:
        if len(coords) < 2:
            return 0.0
        total = 0.0
        for i in range(len(coords) - 1):
            total += self._haversine_km(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
        return total

    def _extract_lgv_lines_from_kml_root(self, root: ET.Element) -> List[List[Tuple[float, float]]]:
        ns = {"k": "http://www.opengis.net/kml/2.2"}
        extracted: List[Tuple[str, List[Tuple[float, float]], float]] = []

        for pm in root.findall(".//k:Placemark", ns):
            name = (pm.findtext("k:name", default="", namespaces=ns) or "").strip().upper()
            coord_text = pm.findtext(".//k:LineString/k:coordinates", default="", namespaces=ns)
            if not coord_text:
                continue
            coords = []
            for token in coord_text.replace("\n", " ").split():
                parts = token.split(",")
                if len(parts) < 2:
                    continue
                try:
                    lon = float(parts[0])
                    lat = float(parts[1])
                except ValueError:
                    continue
                coords.append((lat, lon))
            if len(coords) < 2:
                continue
            extracted.append((name, coords, self._line_length_km(coords)))

        if not extracted:
            return []

        lgv_named = [coords for name, coords, _ in extracted if name.startswith("LGV")]
        if lgv_named:
            return [self._downsample_coords(coords) for coords in lgv_named]

        # Fallback: keep the two longest axes when no explicit LGV naming exists.
        extracted.sort(key=lambda x: x[2], reverse=True)
        longest = [coords for _, coords, length in extracted if length >= 50.0][:2]
        return [self._downsample_coords(coords) for coords in longest]

    def _load_lgv_lines_from_kml(self, kml_path: str) -> List[List[Tuple[float, float]]]:
        if not os.path.exists(kml_path):
            return []
        try:
            with open(kml_path, "rb") as f:
                root = ET.fromstring(f.read())
            lines = self._extract_lgv_lines_from_kml_root(root)
            if lines:
                logging.info("Trace LGV chargee depuis KML: %s (%s ligne(s))", kml_path, len(lines))
            return lines
        except Exception as exc:
            logging.warning("Lecture KML impossible (%s): %s", kml_path, exc)
            return []

    def _load_lgv_lines_from_kmz(self, kmz_path: str) -> List[List[Tuple[float, float]]]:
        if not os.path.exists(kmz_path):
            return []
        try:
            with zipfile.ZipFile(kmz_path) as zf:
                kml_name = "doc.kml" if "doc.kml" in zf.namelist() else zf.namelist()[0]
                root = ET.fromstring(zf.read(kml_name))
            lines = self._extract_lgv_lines_from_kml_root(root)
            if lines:
                logging.info("Trace LGV chargee depuis KMZ: %s (%s ligne(s))", kmz_path, len(lines))
            return lines
        except Exception as exc:
            logging.warning("Lecture KMZ impossible (%s): %s", kmz_path, exc)
            return []

    def _load_lgv_lines(self, candidates: List[str]) -> List[List[Tuple[float, float]]]:
        for path in candidates:
            path = os.path.abspath(path)
            if path.lower().endswith(".kml"):
                lines = self._load_lgv_lines_from_kml(path)
            elif path.lower().endswith(".kmz"):
                lines = self._load_lgv_lines_from_kmz(path)
            else:
                continue
            if lines:
                return lines
        return []

    def _synop_url_for_date(self, day: datetime) -> str:
        return "https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/" + f"synop.{day.strftime('%Y%m%d')}.csv"

    @staticmethod
    def _row_get(row: Dict[str, str], keys: List[str]) -> Optional[str]:
        for key in keys:
            if key in row and row[key] not in (None, ""):
                return row[key]
        return None

    def _extract_synop_record(self, row: Dict[str, str]) -> Optional[Dict[str, object]]:
        station_id = self._row_get(row, ["numer_sta", "station", "id", "ID"])
        date_obs_raw = self._row_get(row, ["date", "Date"])
        lat = self._safe_float(self._row_get(row, ["lat", "latitude", "LAT"]))
        lon = self._safe_float(self._row_get(row, ["lon", "longitude", "LON"]))

        if lat is None or lon is None:
            values = list(row.values())
            if len(values) > 11:
                lat = self._safe_float(values[10])
                lon = self._safe_float(values[11])
        if lat is None or lon is None:
            return None
        if abs(lat) > 90 and abs(lat) <= 9000:
            lat = lat / 100.0
        if abs(lon) > 180 and abs(lon) <= 18000:
            lon = lon / 100.0

        rr1 = self._safe_float(row.get("rr1"))
        rr3 = self._safe_float(row.get("rr3"))
        rr12 = self._safe_float(row.get("rr12"))
        rr24 = self._safe_float(row.get("rr24"))
        generic_rain = self._safe_float(row.get("precipitation"))

        rain_instant = rr1 if rr1 is not None else (rr3 if rr3 is not None else (generic_rain or 0.0))
        rain_12h = rr12 if rr12 is not None else rain_instant
        rain_24h = rr24 if rr24 is not None else rain_12h
        rain_7d = rain_24h
        rain_30d = rain_24h
        rain_month = rain_24h
        rain_forecast = rain_instant  # no forecast from SYNOP row, keep same value
        rain_class = self._station_pluvio_class(float(rain_24h), float(rain_7d), float(rain_30d))

        return {
            "station_id": station_id or "unknown",
            "date_obs_raw": date_obs_raw,
            "latitude": round(float(lat), 6),
            "longitude": round(float(lon), 6),
            "distance_to_lgv_km": round(float(self._point_to_lgv_distance_km(lat, lon)), 3),
            "precipitation_mm": round(float(rain_forecast), 3),
            "rain_24h_mm": round(float(rain_24h), 3),
            "rain_7d_mm": round(float(rain_7d), 3),
            "rain_30d_mm": round(float(rain_30d), 3),
            "rain_month_mm": round(float(rain_month), 3),
            "rain_12h_mm": round(float(rain_12h), 3),
            "rain_instant_mm": round(float(rain_instant), 3),
            "rain_forecast_mm": round(float(rain_forecast), 3),
            "rain_class": rain_class,
            "source": DataSource.SYNOP_METEOFRANCE.value,
        }

    def fetch_pluviometry_synop(self) -> Dict[str, object]:
        logging.info("Pluviometrie: recuperation SYNOP Meteo-France.")
        candidates = [self._now_utc(), self._now_utc() - timedelta(days=1)]
        last_error = None

        for day in candidates:
            url = self._synop_url_for_date(day)
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code != 200:
                    logging.warning("SYNOP non disponible (%s): %s", response.status_code, url)
                    continue
                if not response.text.strip():
                    continue

                reader = csv.DictReader(StringIO(response.text), delimiter=";")
                rows = [rec for rec in (self._extract_synop_record(row) for row in reader) if rec]
                if not rows:
                    continue

                all_df = pd.DataFrame(rows)
                all_df["date"] = pd.to_datetime(all_df["date_obs_raw"], format="%Y%m%d%H%M%S", errors="coerce", utc=True)
                all_df = all_df.dropna(subset=["date"]).sort_values("distance_to_lgv_km")
                if all_df.empty:
                    continue

                selected_df = all_df[all_df["distance_to_lgv_km"] <= self.weather_corridor_km].copy()
                notice = None
                if selected_df.empty:
                    selected_df = all_df.nsmallest(5, "distance_to_lgv_km").copy()
                    selected_df["selection_mode"] = "nearest_fallback_top5"
                    nearest = selected_df.iloc[0]
                    notice = (
                        "Aucune station meteo <= 1 km de la LGV. "
                        f"Fallback: top 5 stations les plus proches (plus proche: {nearest['station_id']} "
                        f"a {nearest['distance_to_lgv_km']:.2f} km)."
                    )
                    logging.warning(notice)
                else:
                    selected_df["selection_mode"] = "within_1km"

                ts = self._now_utc().strftime("%Y%m%d_%H%M%S")
                all_path = os.path.join("data", f"synop_all_stations_{ts}.csv")
                selected_path = os.path.join("data", f"synop_selected_lgv_{ts}.csv")
                all_df.to_csv(all_path, index=False)
                selected_df.to_csv(selected_path, index=False)
                summary_path = os.path.join("data", f"weather_summary_{ts}.json")
                self._save_json(
                    {
                        "timestamp_utc": self._now_utc().isoformat(),
                        "source_url": url,
                        "corridor_km": self.weather_corridor_km,
                        "all_station_count": int(len(all_df)),
                        "selected_station_count": int(len(selected_df)),
                        "notice": notice,
                        "all_csv": all_path,
                        "selected_csv": selected_path,
                    },
                    summary_path,
                )
                return {
                    "all": all_df,
                    "selected": selected_df,
                    "notice": notice,
                    "source_url": url,
                    "summary_path": summary_path,
                }
            except Exception as exc:
                last_error = exc
                logging.warning("Erreur SYNOP (%s): %s", url, exc)

        logging.error("Pluviometrie indisponible. Derniere erreur: %s", last_error)
        empty = pd.DataFrame()
        return {"all": empty, "selected": empty, "notice": "SYNOP indisponible", "source_url": None, "summary_path": None}

    def _fetch_open_meteo_batch(self, batch_points: List[Tuple[float, float]]) -> List[Dict[str, object]]:
        if not batch_points:
            return []
        lats = ",".join(f"{lat:.5f}" for lat, _ in batch_points)
        lons = ",".join(f"{lon:.5f}" for _, lon in batch_points)
        params = {
            "latitude": lats,
            "longitude": lons,
            "hourly": "precipitation",
            "past_days": 35,
            "forecast_days": 1,
            "timezone": "UTC",
        }
        response = self.session.get(self.open_meteo_base, params=params, timeout=45)
        if response.status_code != 200:
            raise RuntimeError(f"Open-Meteo HTTP {response.status_code}")

        payload = response.json()
        entries = payload if isinstance(payload, list) else [payload]
        rows: List[Dict[str, object]] = []

        now_utc = self._now_utc()
        for idx, entry in enumerate(entries):
            lat = self._safe_float(str(entry.get("latitude")))
            lon = self._safe_float(str(entry.get("longitude")))
            if lat is None or lon is None:
                continue

            hourly = entry.get("hourly", {}) or {}
            times = hourly.get("time", []) or []
            precs = hourly.get("precipitation", []) or []
            points: List[Tuple[datetime, float, str]] = []
            for t, p in zip(times, precs):
                p_val = self._safe_float(p)
                if p_val is None:
                    continue
                dt = pd.to_datetime(t, utc=True, errors="coerce")
                if pd.isna(dt):
                    continue
                points.append((dt.to_pydatetime(), float(p_val), str(t)))

            past = [x for x in points if x[0] <= now_utc]
            future = [x for x in points if x[0] > now_utc]
            past.sort(key=lambda x: x[0])
            future.sort(key=lambda x: x[0])

            if past:
                dt_obs, rain_instant, dt_str = past[-1]
            else:
                dt_obs, rain_instant, dt_str = now_utc, 0.0, now_utc.strftime("%Y-%m-%dT%H:%M")

            lower_12h = now_utc - timedelta(hours=12)
            lower_24h = now_utc - timedelta(hours=24)
            lower_7d = now_utc - timedelta(days=7)
            lower_30d = now_utc - timedelta(days=30)
            month_start = datetime(now_utc.year, now_utc.month, 1, tzinfo=timezone.utc)
            rain_12h = sum(v for dt, v, _ in past if dt > lower_12h)
            rain_24h = sum(v for dt, v, _ in past if dt > lower_24h)
            rain_7d = sum(v for dt, v, _ in past if dt > lower_7d)
            rain_30d = sum(v for dt, v, _ in past if dt > lower_30d)
            rain_month = sum(v for dt, v, _ in past if dt >= month_start)
            rain_forecast = future[0][1] if future else rain_instant
            rain_class = self._station_pluvio_class(float(rain_24h), float(rain_7d), float(rain_30d))

            rows.append(
                {
                    "station_id": f"openmeteo_{idx + 1}",
                    "date_obs_raw": dt_str,
                    "latitude": round(float(lat), 6),
                    "longitude": round(float(lon), 6),
                    "distance_to_lgv_km": round(float(self._point_to_lgv_distance_km(float(lat), float(lon))), 3),
                    "precipitation_mm": round(rain_forecast, 3),
                    "rain_24h_mm": round(rain_24h, 3),
                    "rain_7d_mm": round(rain_7d, 3),
                    "rain_30d_mm": round(rain_30d, 3),
                    "rain_month_mm": round(rain_month, 3),
                    "rain_12h_mm": round(rain_12h, 3),
                    "rain_instant_mm": round(rain_instant, 3),
                    "rain_forecast_mm": round(rain_forecast, 3),
                    "rain_class": rain_class,
                    "source": DataSource.OPEN_METEO.value,
                    "selection_mode": "open_meteo_grid",
                }
            )
        return rows

    def fetch_pluviometry_open_meteo(self) -> Dict[str, object]:
        logging.info("Pluviometrie: recuperation Open-Meteo sur grille LGV.")
        sampled = self._sample_points_along_lgv(self.open_meteo_sample_step_km, self.open_meteo_max_points)
        if not sampled:
            empty = pd.DataFrame()
            return {"all": empty, "selected": empty, "notice": "Open-Meteo: echantillonnage LGV vide", "source_url": self.open_meteo_base}

        rows: List[Dict[str, object]] = []
        # Keep URL length reasonable.
        batch_size = 20
        try:
            for i in range(0, len(sampled), batch_size):
                rows.extend(self._fetch_open_meteo_batch(sampled[i : i + batch_size]))
        except Exception as exc:
            logging.warning("Open-Meteo indisponible: %s", exc)
            empty = pd.DataFrame()
            return {"all": empty, "selected": empty, "notice": f"Open-Meteo indisponible: {exc}", "source_url": self.open_meteo_base}

        if not rows:
            empty = pd.DataFrame()
            return {"all": empty, "selected": empty, "notice": "Open-Meteo: aucune donnee retournee", "source_url": self.open_meteo_base}

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date_obs_raw"], errors="coerce", utc=True)
        df = df.dropna(subset=["date"]).sort_values("distance_to_lgv_km")
        ts = self._now_utc().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join("data", f"open_meteo_lgv_grid_{ts}.csv")
        df.to_csv(csv_path, index=False)
        return {
            "all": df.copy(),
            "selected": df.copy(),
            "notice": f"Open-Meteo grille LGV: {len(df)} points.",
            "source_url": self.open_meteo_base,
            "summary_path": csv_path,
        }

    def fetch_pluviometry_combined(self) -> Dict[str, object]:
        synop = self.fetch_pluviometry_synop()
        open_meteo = self.fetch_pluviometry_open_meteo()

        frames_all = [df for df in [synop.get("all"), open_meteo.get("all")] if isinstance(df, pd.DataFrame) and not df.empty]
        frames_sel = [df for df in [synop.get("selected"), open_meteo.get("selected")] if isinstance(df, pd.DataFrame) and not df.empty]
        all_df = pd.concat(frames_all, ignore_index=True) if frames_all else pd.DataFrame()
        selected_df = pd.concat(frames_sel, ignore_index=True) if frames_sel else pd.DataFrame()

        notices = [n for n in [synop.get("notice"), open_meteo.get("notice")] if n]
        notice = " | ".join(notices) if notices else None

        ts = self._now_utc().strftime("%Y%m%d_%H%M%S")
        combined_csv = os.path.join("data", f"weather_combined_{ts}.csv")
        if not selected_df.empty:
            selected_df.to_csv(combined_csv, index=False)

        summary_path = os.path.join("data", f"weather_summary_combined_{ts}.json")
        self._save_json(
            {
                "timestamp_utc": self._now_utc().isoformat(),
                "synop_source": synop.get("source_url"),
                "open_meteo_source": open_meteo.get("source_url"),
                "selected_station_count": int(len(selected_df)) if not selected_df.empty else 0,
                "notice": notice,
                "combined_csv": combined_csv if not selected_df.empty else None,
            },
            summary_path,
        )
        return {
            "all": all_df,
            "selected": selected_df,
            "notice": notice,
            "source_url": {"synop": synop.get("source_url"), "open_meteo": open_meteo.get("source_url")},
            "summary_path": summary_path,
        }

    def _discover_station_for_river(self, rp: RiverMonitoringPoint) -> Optional[Dict[str, object]]:
        pad = 0.65
        min_lon, min_lat = rp.longitude - pad, rp.latitude - pad
        max_lon, max_lat = rp.longitude + pad, rp.latitude + pad
        url = f"{self.hubeau_base}{self.hubeau_endpoints['stations']}"
        params = {"size": 1200, "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}"}

        response = self.session.get(url, params=params, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"Hub'Eau stations HTTP {response.status_code}")

        items = response.json().get("data", [])
        if not items:
            return None

        river_name = rp.river.lower()
        ranked: List[Tuple[int, float, Dict[str, object]]] = []

        for st in items:
            code = st.get("code_station") or st.get("code_entite") or st.get("code_site")
            lat = st.get("latitude_station") or st.get("latitude")
            lon = st.get("longitude_station") or st.get("longitude")
            name = st.get("libelle_station") or st.get("libelle") or st.get("nom_station") or ""
            if not code or lat is None or lon is None:
                continue
            try:
                lat_f, lon_f = float(lat), float(lon)
            except Exception:
                continue

            dist_ref_km = self._haversine_km(rp.latitude, rp.longitude, lat_f, lon_f)
            priority = 0 if river_name in str(name).lower() else 1
            ranked.append(
                (
                    priority,
                    dist_ref_km,
                    {
                        "station_code": code,
                        "station_name": name,
                        "station_latitude": lat_f,
                        "station_longitude": lon_f,
                        "distance_to_river_ref_km": round(dist_ref_km, 2),
                        "auto_selected": True,
                        "match_type": "name+distance" if priority == 0 else "distance_only",
                    },
                )
            )

        if not ranked:
            return None
        ranked.sort(key=lambda x: (x[0], x[1]))
        return ranked[0][2]

    def _discover_station_by_name(self, rp: RiverMonitoringPoint, keyword: str) -> Optional[Dict[str, object]]:
        pad = 1.0
        min_lon, min_lat = rp.longitude - pad, rp.latitude - pad
        max_lon, max_lat = rp.longitude + pad, rp.latitude + pad
        url = f"{self.hubeau_base}{self.hubeau_endpoints['stations']}"
        params = {"size": 2000, "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}"}

        response = self.session.get(url, params=params, timeout=30)
        if response.status_code != 200:
            return None

        key = keyword.lower()
        hits: List[Tuple[float, Dict[str, object]]] = []
        for st in response.json().get("data", []):
            code = st.get("code_station") or st.get("code_entite") or st.get("code_site")
            lat = st.get("latitude_station") or st.get("latitude")
            lon = st.get("longitude_station") or st.get("longitude")
            name = st.get("libelle_station") or st.get("libelle") or st.get("nom_station") or ""
            if not code or lat is None or lon is None:
                continue
            if key not in str(name).lower():
                continue
            try:
                lat_f, lon_f = float(lat), float(lon)
            except Exception:
                continue
            dist_ref_km = self._haversine_km(rp.latitude, rp.longitude, lat_f, lon_f)
            hits.append(
                (
                    dist_ref_km,
                    {
                        "station_code": code,
                        "station_name": name,
                        "station_latitude": lat_f,
                        "station_longitude": lon_f,
                        "distance_to_river_ref_km": round(dist_ref_km, 2),
                        "auto_selected": True,
                        "match_type": f"name:{keyword}",
                    },
                )
            )

        if not hits:
            return None
        hits.sort(key=lambda x: x[0])
        return hits[0][1]

    def _resolve_station_for_river(self, rp: RiverMonitoringPoint) -> Optional[Dict[str, object]]:
        if rp.station_code:
            return {
                "station_code": rp.station_code,
                "station_name": "manual_config",
                "station_latitude": None,
                "station_longitude": None,
                "distance_to_river_ref_km": None,
                "auto_selected": False,
                "match_type": "manual",
            }

        cached = self.station_cache.get(rp.river)
        if cached and cached.get("station_code"):
            cached["auto_selected"] = True
            cached.setdefault("match_type", "cached")
            return cached

        # User requirement: Charente uses Champniers measurement point when available.
        if rp.river.lower() == "charente":
            champniers = self._discover_station_by_name(rp, "champniers")
            if champniers:
                self.station_cache[rp.river] = champniers
                self._save_station_cache()
                return champniers

        discovered = self._discover_station_for_river(rp)
        if discovered:
            self.station_cache[rp.river] = discovered
            self._save_station_cache()
        return discovered

    def fetch_hydrometry_for_station(self, station_code: str, hours: int = 24) -> pd.DataFrame:
        url = f"{self.hubeau_base}{self.hubeau_endpoints['observations_tr']}"
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
            raise RuntimeError(f"Hub'Eau observations_tr HTTP {response.status_code}")

        rows = []
        for item in response.json().get("data", []):
            date_obs = item.get("date_obs")
            result = item.get("resultat_obs")
            if date_obs is None or result is None:
                continue
            try:
                rows.append(
                    {
                        "date_obs": pd.to_datetime(date_obs, utc=True, errors="coerce"),
                        "level_m": float(result) / 1000.0,
                        "status": item.get("statut_observation"),
                    }
                )
            except Exception:
                continue

        df = pd.DataFrame(rows).dropna(subset=["date_obs"])
        if df.empty:
            return df
        return df.sort_values("date_obs", ascending=False).reset_index(drop=True)

    def fetch_all_river_levels(self) -> Dict[str, Dict[str, object]]:
        logging.info("Hydrometrie: recuperation niveaux de cours d'eau.")
        out: Dict[str, Dict[str, object]] = {}

        for rp in self.river_points:
            try:
                station_meta = self._resolve_station_for_river(rp)
                if not station_meta:
                    out[rp.river] = {
                        "configured": False,
                        "river": rp.river,
                        "name": rp.name,
                        "message": "Aucune station Hub'Eau resolue pour cette riviere.",
                        "source": DataSource.VIGICRUES_HYDRO.value,
                    }
                    continue

                station_code = str(station_meta["station_code"])
                obs_df = self.fetch_hydrometry_for_station(station_code=station_code, hours=24)

                if obs_df.empty:
                    out[rp.river] = {
                        "configured": True,
                        "river": rp.river,
                        "name": rp.name,
                        "station_code": station_code,
                        "station_name": station_meta.get("station_name"),
                        "auto_selected": bool(station_meta.get("auto_selected", False)),
                        "match_type": station_meta.get("match_type"),
                        "message": "Aucune observation disponible sur les 24h.",
                        "source": DataSource.VIGICRUES_HYDRO.value,
                    }
                    continue

                latest = obs_df.iloc[0]
                oldest = obs_df.iloc[-1]
                dt_h = (latest["date_obs"] - oldest["date_obs"]).total_seconds() / 3600.0
                trend_mph = (latest["level_m"] - oldest["level_m"]) / dt_h if dt_h > 0 else 0.0

                out[rp.river] = {
                    "configured": True,
                    "river": rp.river,
                    "name": rp.name,
                    "station_code": station_code,
                    "station_name": station_meta.get("station_name"),
                    "station_latitude": station_meta.get("station_latitude"),
                    "station_longitude": station_meta.get("station_longitude"),
                    "distance_to_river_ref_km": station_meta.get("distance_to_river_ref_km"),
                    "auto_selected": bool(station_meta.get("auto_selected", False)),
                    "match_type": station_meta.get("match_type"),
                    "last_level_m": round(float(latest["level_m"]), 3),
                    "trend_mph": round(float(trend_mph), 3),
                    "n_obs": int(len(obs_df)),
                    "threshold_m": rp.threshold_m,
                    "rapid_rise_mph": rp.rapid_rise_mph,
                    "observations": obs_df.head(24).to_dict(orient="records"),
                    "source": DataSource.VIGICRUES_HYDRO.value,
                }

            except Exception as exc:
                out[rp.river] = {
                    "configured": bool(rp.station_code),
                    "river": rp.river,
                    "name": rp.name,
                    "station_code": rp.station_code,
                    "error": str(exc),
                    "source": DataSource.VIGICRUES_HYDRO.value,
                }

        ts = self._now_utc().strftime("%Y%m%d_%H%M%S")
        self._save_json(out, os.path.join("data", f"river_levels_{ts}.json"))
        return out

    @staticmethod
    def _classify_hydro_network_risk(trend_mph: Optional[float]) -> Dict[str, object]:
        if trend_mph is None:
            return {"risk_level": "INDETERMINE", "color": "#6b7280", "score": 1}
        if trend_mph >= 0.12:
            return {"risk_level": "CRITIQUE", "color": "#7f1d1d", "score": 4}
        if trend_mph >= 0.08:
            return {"risk_level": "ELEVE", "color": "#dc2626", "score": 3}
        if trend_mph >= 0.04:
            return {"risk_level": "MODERE", "color": "#ea580c", "score": 2}
        return {"risk_level": "FAIBLE", "color": "#16a34a", "score": 1}

    def fetch_hydro_network_near_lgv(self) -> Dict[str, object]:
        cached = self._load_fresh_cache(self.hydro_network_cache_file, self.hydro_network_cache_hours)
        if cached:
            return cached

        min_lon, min_lat, max_lon, max_lat = self._bbox_from_lgv_lines(pad_deg=0.1)
        response = self.session.get(
            f"{self.hubeau_base}{self.hubeau_endpoints['stations']}",
            params={"size": 2000, "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}"},
            timeout=30,
        )
        if response.status_code != 200:
            payload = {
                "timestamp_utc": self._now_utc().isoformat(),
                "source": DataSource.HUBEAU_HYDRO.value,
                "summary": {"error": f"Hub'Eau stations HTTP {response.status_code}"},
                "alerts": [],
                "stations": [],
            }
            self._save_json(payload, self.hydro_network_cache_file)
            return payload

        rows = response.json().get("data", [])
        candidates: List[Dict[str, object]] = []
        seen_codes = set()
        for st in rows:
            code = st.get("code_station") or st.get("code_entite") or st.get("code_site")
            lat = self._safe_float(st.get("latitude_station") or st.get("latitude"))
            lon = self._safe_float(st.get("longitude_station") or st.get("longitude"))
            if not code or lat is None or lon is None:
                continue
            code = str(code)
            if code in seen_codes:
                continue
            dist_km = self._point_to_lgv_distance_km(float(lat), float(lon))
            if dist_km > self.hydro_network_corridor_km:
                continue
            seen_codes.add(code)
            candidates.append(
                {
                    "station_code": code,
                    "station_name": str(st.get("libelle_station") or st.get("libelle") or code),
                    "river_name": str(st.get("libelle_cours_eau") or "inconnu"),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "distance_to_lgv_km": round(float(dist_km), 3),
                }
            )

        candidates = sorted(candidates, key=lambda x: x["distance_to_lgv_km"])[: self.hydro_network_max_stations]
        stations: List[Dict[str, object]] = []
        alerts: List[Dict[str, object]] = []
        for st in candidates:
            try:
                obs_df = self.fetch_hydrometry_for_station(st["station_code"], hours=self.hydro_network_hours)
            except Exception:
                continue
            if obs_df.empty:
                continue

            latest = obs_df.iloc[0]
            old_idx = min(len(obs_df) - 1, 8)
            old = obs_df.iloc[old_idx]
            dt_h = max((latest["date_obs"] - old["date_obs"]).total_seconds() / 3600.0, 0.0)
            trend_mph = (float(latest["level_m"]) - float(old["level_m"])) / dt_h if dt_h > 0 else None
            cls = self._classify_hydro_network_risk(trend_mph)
            item = {
                **st,
                "last_obs_utc": latest["date_obs"].isoformat(),
                "last_level_m": round(float(latest["level_m"]), 3),
                "trend_mph": None if trend_mph is None else round(float(trend_mph), 3),
                "n_obs": int(len(obs_df)),
                "risk_level": cls["risk_level"],
                "risk_color": cls["color"],
                "risk_score": cls["score"],
                "source": DataSource.HUBEAU_HYDRO.value,
            }
            stations.append(item)
            if item["risk_level"] in {"CRITIQUE", "ELEVE"}:
                alerts.append(
                    {
                        "type": "HYDRO_RESEAU",
                        "level": "CRITIQUE" if item["risk_level"] == "CRITIQUE" else "ELEVE",
                        "message": (
                            f"{item['station_code']} ({item['river_name']}): "
                            f"trend={item['trend_mph']} m/h | niveau={item['last_level_m']} m"
                        ),
                    }
                )

        summary = {
            "candidate_stations": int(len(candidates)),
            "stations_with_data": int(len(stations)),
            "critical_stations": int(sum(1 for s in stations if s["risk_level"] == "CRITIQUE")),
            "high_stations": int(sum(1 for s in stations if s["risk_level"] == "ELEVE")),
            "moderate_stations": int(sum(1 for s in stations if s["risk_level"] == "MODERE")),
        }
        payload = {
            "timestamp_utc": self._now_utc().isoformat(),
            "source": DataSource.HUBEAU_HYDRO.value,
            "corridor_km": self.hydro_network_corridor_km,
            "summary": summary,
            "alerts": alerts[:12],
            "stations": stations,
        }
        self._save_json(payload, self.hydro_network_cache_file)
        ts = self._now_utc().strftime("%Y%m%d_%H%M%S")
        self._save_json(payload, os.path.join("data", f"hydro_network_{ts}.json"))
        return payload

    def build_surveillance_sectors(
        self,
        weather_df: pd.DataFrame,
        geotech: Optional[Dict[str, object]],
        piezometers: Optional[Dict[str, object]],
        hydro_network: Optional[Dict[str, object]],
    ) -> Dict[str, object]:
        centers = self._sample_points_along_lgv(self.sector_length_km, 40)
        sectors = []
        alerts = []

        weather_records = weather_df.to_dict(orient="records") if isinstance(weather_df, pd.DataFrame) and not weather_df.empty else []
        geotech_points = geotech.get("points", []) if isinstance(geotech, dict) and isinstance(geotech.get("points"), list) else []
        piezo_points = piezometers.get("stations", []) if isinstance(piezometers, dict) and isinstance(piezometers.get("stations"), list) else []
        hydro_points = hydro_network.get("stations", []) if isinstance(hydro_network, dict) and isinstance(hydro_network.get("stations"), list) else []

        rain_score_map = {"NORMAL": 1, "VIGILANCE": 2, "MODERE": 2, "ELEVE": 3, "CRITIQUE": 4}
        piezo_score_map = {"FAIBLE": 1, "MODERE": 2, "ELEVE": 3, "TRES_ELEVE": 4}
        hydro_score_map = {"FAIBLE": 1, "MODERE": 2, "ELEVE": 3, "CRITIQUE": 4}

        for idx, (lat, lon) in enumerate(centers, start=1):
            near_weather = [
                r for r in weather_records
                if self._haversine_km(lat, lon, float(r.get("latitude", lat)), float(r.get("longitude", lon))) <= self.sector_radius_km
            ]
            near_geo = [
                g for g in geotech_points
                if self._haversine_km(lat, lon, float(g.get("latitude", lat)), float(g.get("longitude", lon))) <= self.sector_radius_km
            ]
            near_piezo = [
                p for p in piezo_points
                if self._haversine_km(lat, lon, float(p.get("latitude", lat)), float(p.get("longitude", lon))) <= self.sector_radius_km
            ]
            near_hydro = [
                h for h in hydro_points
                if self._haversine_km(lat, lon, float(h.get("latitude", lat)), float(h.get("longitude", lon))) <= self.sector_radius_km
            ]

            if near_weather:
                max24 = max(float(r.get("rain_24h_mm", 0.0) or 0.0) for r in near_weather)
                max7 = max(float(r.get("rain_7d_mm", r.get("rain_24h_mm", 0.0)) or 0.0) for r in near_weather)
                max30 = max(float(r.get("rain_30d_mm", r.get("rain_24h_mm", 0.0)) or 0.0) for r in near_weather)
                rain_class = self._station_pluvio_class(max24, max7, max30)
                weather_score = rain_score_map.get(rain_class, 1)
            else:
                max24 = max7 = max30 = 0.0
                rain_class = "INDETERMINE"
                weather_score = 1

            geotech_score = max((int(g.get("risk_score", 1) or 1) for g in near_geo), default=1)
            piezo_score = max((piezo_score_map.get(str(p.get("risk_level")), 1) for p in near_piezo), default=1)
            hydro_score = max((hydro_score_map.get(str(h.get("risk_level")), 1) for h in near_hydro), default=1)

            component_scores = [weather_score, geotech_score, piezo_score, hydro_score]
            avg_score = sum(component_scores) / len(component_scores)
            worst = max(component_scores)
            if avg_score >= 3.2:
                risk_level = "CRITIQUE"
                color = "#7f1d1d"
            elif worst >= 4 or avg_score >= 2.5:
                risk_level = "ELEVE"
                color = "#dc2626"
            elif avg_score >= 1.8:
                risk_level = "MODERE"
                color = "#ea580c"
            else:
                risk_level = "FAIBLE"
                color = "#16a34a"

            sector = {
                "sector_id": f"S{idx:02d}",
                "latitude": round(float(lat), 6),
                "longitude": round(float(lon), 6),
                "radius_km": self.sector_radius_km,
                "risk_level": risk_level,
                "risk_color": color,
                "score": round(avg_score, 2),
                "weather_class": rain_class,
                "weather_max_24h_mm": round(max24, 1),
                "weather_max_7d_mm": round(max7, 1),
                "weather_max_30d_mm": round(max30, 1),
                "geotech_points": int(len(near_geo)),
                "piezometers": int(len(near_piezo)),
                "hydro_stations": int(len(near_hydro)),
                "under_watch": bool(risk_level in {"CRITIQUE", "ELEVE", "MODERE"}),
            }
            sectors.append(sector)
            if sector["risk_level"] in {"CRITIQUE", "ELEVE"}:
                alerts.append(
                    {
                        "type": "SECTEUR",
                        "level": "CRITIQUE" if sector["risk_level"] == "CRITIQUE" else "ELEVE",
                        "message": (
                            f"{sector['sector_id']}: score={sector['score']} | pluie24h={sector['weather_max_24h_mm']} mm "
                            f"| geotech={sector['geotech_points']} | piezo={sector['piezometers']} | hydro={sector['hydro_stations']}"
                        ),
                    }
                )

        summary = {
            "sector_count": int(len(sectors)),
            "critical": int(sum(1 for s in sectors if s["risk_level"] == "CRITIQUE")),
            "high": int(sum(1 for s in sectors if s["risk_level"] == "ELEVE")),
            "moderate": int(sum(1 for s in sectors if s["risk_level"] == "MODERE")),
            "watch": int(sum(1 for s in sectors if s["under_watch"])),
        }
        payload = {
            "timestamp_utc": self._now_utc().isoformat(),
            "sector_length_km": self.sector_length_km,
            "sector_radius_km": self.sector_radius_km,
            "summary": summary,
            "alerts": alerts[:12],
            "sectors": sectors,
        }
        ts = self._now_utc().strftime("%Y%m%d_%H%M%S")
        self._save_json(payload, os.path.join("data", f"surveillance_sectors_{ts}.json"))
        return payload

    def analyze_risks(
        self,
        weather_df: pd.DataFrame,
        weather_notice: Optional[str],
        rivers: Dict[str, Dict[str, object]],
        geotech: Optional[Dict[str, object]] = None,
        piezometers: Optional[Dict[str, object]] = None,
        hydro_network: Optional[Dict[str, object]] = None,
        sectors: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        risks: Dict[str, object] = {
            "timestamp_utc": self._now_utc().isoformat(),
            "risk_level": "FAIBLE",
            "score": 0.0,
            "alerts": [],
            "details": {},
            "recommendations": [],
            "weather_selection_notice": weather_notice,
        }
        factors: List[int] = []

        if not weather_df.empty:
            metric_col = "rain_24h_mm" if "rain_24h_mm" in weather_df.columns else "precipitation_mm"
            max_rain = float(weather_df[metric_col].max())
            mean_rain = float(weather_df[metric_col].mean())
            max_7d = float(weather_df.get("rain_7d_mm", weather_df[metric_col]).max())
            max_30d = float(weather_df.get("rain_30d_mm", weather_df[metric_col]).max())
            class_counts = weather_df.get("rain_class", pd.Series(dtype=str)).value_counts().to_dict()
            risks["details"]["weather"] = {
                "station_count": int(len(weather_df)),
                "corridor_km": self.weather_corridor_km,
                "metric": metric_col,
                "max_mm": round(max_rain, 2),
                "mean_mm": round(mean_rain, 2),
                "max_7d_mm": round(max_7d, 2),
                "max_30d_mm": round(max_30d, 2),
                "class_counts": class_counts,
            }

            if max_rain >= self.alert_thresholds_mm["catastrophique_24h"]:
                factors.append(4)
                risks["alerts"].append(
                    {
                        "type": "PLUIE_CATASTROPHIQUE",
                        "level": "CRITIQUE",
                        "message": f"Cumul 24h catastrophique: {max_rain:.1f} mm",
                    }
                )
            elif max_rain >= self.alert_thresholds_mm["extreme_24h"]:
                factors.append(3)
                risks["alerts"].append({"type": "PLUIE", "level": "ELEVE", "message": f"Pluie extreme: {max_rain:.1f} mm/24h"})
            elif max_rain >= self.alert_thresholds_mm["forte_24h"]:
                factors.append(3)
                risks["alerts"].append({"type": "PLUIE", "level": "ELEVE", "message": f"Pluie forte: {max_rain:.1f} mm/24h"})
            elif max_rain >= self.alert_thresholds_mm["moderee_24h"]:
                factors.append(2)
                risks["alerts"].append({"type": "PLUIE", "level": "MODERE", "message": f"Pluie notable: {max_rain:.1f} mm/24h"})

            if max_7d >= 120:
                factors.append(3)
                risks["alerts"].append({"type": "PLUIE_CUMUL_7J", "level": "ELEVE", "message": f"Cumul 7 jours eleve: {max_7d:.1f} mm"})
            elif max_7d >= 80:
                factors.append(2)
                risks["alerts"].append({"type": "PLUIE_CUMUL_7J", "level": "MODERE", "message": f"Cumul 7 jours notable: {max_7d:.1f} mm"})

            if max_30d >= 220:
                factors.append(2)
                risks["alerts"].append({"type": "PLUIE_CUMUL_30J", "level": "ELEVE", "message": f"Cumul 30 jours eleve: {max_30d:.1f} mm"})
            elif max_30d >= 150:
                factors.append(1)
                risks["alerts"].append({"type": "PLUIE_CUMUL_30J", "level": "MODERE", "message": f"Cumul 30 jours notable: {max_30d:.1f} mm"})

            critical_stations = weather_df[weather_df.get("rain_24h_mm", weather_df["precipitation_mm"]) >= self.alert_thresholds_mm["extreme_24h"]]
            for _, row in critical_stations.head(5).iterrows():
                risks["alerts"].append(
                    {
                        "type": "STATION_CRITIQUE",
                        "level": "ELEVE",
                        "message": (
                            f"Station {row.get('station_id')} ({row.get('source')}): "
                            f"{row.get('rain_24h_mm', row.get('precipitation_mm'))} mm/24h "
                            f"a {row.get('distance_to_lgv_km')} km de la LGV"
                        ),
                    }
                )

        if weather_notice:
            factors.append(1)
            risks["alerts"].append({"type": "COUVERTURE", "level": "INFO", "message": weather_notice})
            risks["recommendations"].append("Verifier la couverture meteo locale et la qualite des capteurs.")

        if isinstance(geotech, dict):
            summary = geotech.get("summary", {}) if isinstance(geotech.get("summary"), dict) else {}
            risks["details"]["geotech"] = {
                "sample_count": geotech.get("sample_count", 0),
                "critical_points": summary.get("critical_points", 0),
                "high_points": summary.get("high_points", 0),
                "moderate_points": summary.get("moderate_points", 0),
                "source": geotech.get("source"),
            }
            critical_points = int(summary.get("critical_points", 0) or 0)
            high_points = int(summary.get("high_points", 0) or 0)
            if critical_points > 0:
                factors.append(3)
                risks["alerts"].append(
                    {
                        "type": "GEOTECH_CRITIQUE",
                        "level": "CRITIQUE",
                        "message": f"{critical_points} points LGV a risque geotechnique critique (argiles/MVT).",
                    }
                )
            elif high_points >= 3:
                factors.append(2)
                risks["alerts"].append(
                    {
                        "type": "GEOTECH_ELEVE",
                        "level": "ELEVE",
                        "message": f"{high_points} points LGV a risque geotechnique eleve.",
                    }
                )
            for alert in (geotech.get("alerts") or [])[:5]:
                if isinstance(alert, dict):
                    risks["alerts"].append(alert)
            risks["recommendations"].append(
                "Prioriser les inspections genie civil sur les secteurs argileux exposes et points MVT."
            )

        if isinstance(piezometers, dict):
            summary = piezometers.get("summary", {}) if isinstance(piezometers.get("summary"), dict) else {}
            risks["details"]["groundwater"] = {
                "stations_in_corridor": summary.get("stations_in_corridor", 0),
                "stations_with_data": summary.get("stations_with_data", 0),
                "very_high_risk": summary.get("very_high_risk", 0),
                "high_risk": summary.get("high_risk", 0),
                "rapid_rise_alerts": summary.get("rapid_rise_alerts", 0),
                "source": piezometers.get("source"),
            }
            very_high = int(summary.get("very_high_risk", 0) or 0)
            high = int(summary.get("high_risk", 0) or 0)
            rapid = int(summary.get("rapid_rise_alerts", 0) or 0)
            if very_high > 0:
                factors.append(3)
                risks["alerts"].append(
                    {
                        "type": "NAPPE_TRES_HAUTE",
                        "level": "CRITIQUE",
                        "message": f"{very_high} piezometre(s) avec nappe tres proche du sol.",
                    }
                )
            elif high > 0:
                factors.append(2)
                risks["alerts"].append(
                    {
                        "type": "NAPPE_HAUTE",
                        "level": "ELEVE",
                        "message": f"{high} piezometre(s) avec niveau de nappe eleve.",
                    }
                )
            if rapid > 0:
                factors.append(2)
                risks["alerts"].append(
                    {
                        "type": "NAPPE_MONTEE_RAPIDE",
                        "level": "ELEVE",
                        "message": f"{rapid} piezometre(s) en remontee rapide de nappe.",
                    }
                )
            for alert in (piezometers.get("alerts") or [])[:8]:
                if isinstance(alert, dict):
                    risks["alerts"].append(alert)
            risks["recommendations"].append(
                "Verifier reseaux de drainage, exutoires et stabilite des talus dans les secteurs a nappe haute."
            )

        if isinstance(hydro_network, dict):
            summary = hydro_network.get("summary", {}) if isinstance(hydro_network.get("summary"), dict) else {}
            risks["details"]["hydro_network"] = {
                "candidate_stations": summary.get("candidate_stations", 0),
                "stations_with_data": summary.get("stations_with_data", 0),
                "critical_stations": summary.get("critical_stations", 0),
                "high_stations": summary.get("high_stations", 0),
                "moderate_stations": summary.get("moderate_stations", 0),
                "source": hydro_network.get("source"),
            }
            critical = int(summary.get("critical_stations", 0) or 0)
            high = int(summary.get("high_stations", 0) or 0)
            if critical > 0:
                factors.append(3)
                risks["alerts"].append(
                    {
                        "type": "HYDRO_RESEAU_CRITIQUE",
                        "level": "CRITIQUE",
                        "message": f"{critical} station(s) hydro reseau en montee critique.",
                    }
                )
            elif high >= 2:
                factors.append(2)
                risks["alerts"].append(
                    {
                        "type": "HYDRO_RESEAU_ELEVE",
                        "level": "ELEVE",
                        "message": f"{high} station(s) hydro reseau en montee elevee.",
                    }
                )
            for alert in (hydro_network.get("alerts") or [])[:8]:
                if isinstance(alert, dict):
                    risks["alerts"].append(alert)
            risks["recommendations"].append(
                "Suivre en continu les stations Vigicrues proches de la plateforme LGV et valider les tendances."
            )

        if isinstance(sectors, dict):
            summary = sectors.get("summary", {}) if isinstance(sectors.get("summary"), dict) else {}
            risks["details"]["sectors"] = {
                "sector_count": summary.get("sector_count", 0),
                "critical": summary.get("critical", 0),
                "high": summary.get("high", 0),
                "moderate": summary.get("moderate", 0),
                "watch": summary.get("watch", 0),
            }
            critical = int(summary.get("critical", 0) or 0)
            high = int(summary.get("high", 0) or 0)
            if critical > 0:
                factors.append(3)
                risks["alerts"].append(
                    {
                        "type": "SECTEURS_CRITIQUES",
                        "level": "CRITIQUE",
                        "message": f"{critical} secteur(s) de surveillance en criticite.",
                    }
                )
            elif high > 0:
                factors.append(2)
                risks["alerts"].append(
                    {
                        "type": "SECTEURS_ELEVES",
                        "level": "ELEVE",
                        "message": f"{high} secteur(s) de surveillance en risque eleve.",
                    }
                )
            for alert in (sectors.get("alerts") or [])[:8]:
                if isinstance(alert, dict):
                    risks["alerts"].append(alert)
            risks["recommendations"].append(
                "Piloter la surveillance terrain par secteurs (priorite CRITIQUE/ELEVE) sur l'axe 300 km."
            )

        for river, info in rivers.items():
            if not isinstance(info, dict):
                continue
            if not info.get("configured") or info.get("last_level_m") is None:
                risks["details"][river] = {
                    "configured": info.get("configured", False),
                    "message": info.get("message") or info.get("error"),
                }
                continue

            last_level = info.get("last_level_m")
            threshold = info.get("threshold_m")
            trend = info.get("trend_mph")
            rapid = info.get("rapid_rise_mph")

            risks["details"][river] = {
                "station_code": info.get("station_code"),
                "station_name": info.get("station_name"),
                "auto_selected": info.get("auto_selected"),
                "match_type": info.get("match_type"),
                "last_level_m": last_level,
                "trend_mph": trend,
                "threshold_m": threshold,
                "rapid_rise_mph": rapid,
            }

            if threshold is not None and last_level is not None and last_level >= threshold:
                factors.append(3)
                risks["alerts"].append({"type": "CRUE", "level": "ELEVE", "message": f"{river}: {last_level:.2f} m >= seuil {threshold:.2f} m"})

            if rapid is not None and trend is not None and trend >= rapid:
                factors.append(2)
                risks["alerts"].append({"type": "MONTEE_RAPIDE", "level": "MODERE", "message": f"{river}: {trend:.2f} m/h (seuil {rapid:.2f} m/h)"})

        if factors:
            score = sum(factors) / len(factors)
            risks["score"] = round(score, 2)
            if score >= 2.8:
                risks["risk_level"] = "ELEVE"
            elif score >= 1.8:
                risks["risk_level"] = "MODERE"

        if risks["risk_level"] == "ELEVE":
            risks["recommendations"].append("Alerter l'astreinte et declencher inspection terrain ciblee genie civil.")
        elif risks["risk_level"] == "MODERE":
            risks["recommendations"].append("Surveillance renforcee sur secteurs geotechniques et hydrauliques sensibles.")
        else:
            risks["recommendations"].append("Surveillance normale.")

        risks["recommendations"].append("Verifier coherence des mesures (meteo, hydro, piezometres) avant decision travaux.")
        seen = set()
        risks["recommendations"] = [r for r in risks["recommendations"] if not (r in seen or seen.add(r))]
        ts = self._now_utc().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join("reports", f"risk_analysis_{ts}.json")
        self._save_json(risks, report_path)
        risks["report_path"] = report_path
        return risks

    def generate_map(
        self,
        weather_df: pd.DataFrame,
        rivers: Dict[str, Dict[str, object]],
        risks: Dict[str, object],
        geotech: Optional[Dict[str, object]] = None,
        piezometers: Optional[Dict[str, object]] = None,
        hydro_network: Optional[Dict[str, object]] = None,
        sectors: Optional[Dict[str, object]] = None,
    ) -> str:
        ts = self._now_utc().strftime("%Y%m%d_%H%M%S")
        map_path = os.path.join("reports", f"lgv_dashboard_{ts}.html")
        latest_path = os.path.join("reports", "lgv_dashboard_latest.html")

        lgv_lines = [
            [{"lat": lat, "lon": lon} for lat, lon in line]
            for line in self.lgv_lines_latlon
            if len(line) >= 2
        ]

        weather_markers = []
        if not weather_df.empty:
            for _, row in weather_df.iterrows():
                mode = str(row.get("selection_mode", "unknown"))
                source = str(row.get("source", "unknown"))
                rain24 = float(row.get("rain_24h_mm", row.get("precipitation_mm", 0.0)) or 0.0)
                rain7 = float(row.get("rain_7d_mm", rain24) or rain24)
                rain30 = float(row.get("rain_30d_mm", rain7) or rain7)
                rainmonth = float(row.get("rain_month_mm", rain30) or rain30)
                rain_class = str(row.get("rain_class") or self._station_pluvio_class(rain24, rain7, rain30))
                popup = (
                    f"<b>Station:</b> {html.escape(str(row.get('station_id')))}<br>"
                    f"<b>Source:</b> {html.escape(source)}<br>"
                    f"<b>Classe pluvio:</b> {html.escape(rain_class)}<br>"
                    f"<b>Cumul 24h:</b> {rain24:.1f} mm<br>"
                    f"<b>Cumul 7j:</b> {rain7:.1f} mm<br>"
                    f"<b>Cumul 30j:</b> {rain30:.1f} mm<br>"
                    f"<b>Cumul mois:</b> {rainmonth:.1f} mm<br>"
                    f"<b>Cumul 12h:</b> {row.get('rain_12h_mm', row.get('precipitation_mm'))} mm<br>"
                    f"<b>Instantane:</b> {row.get('rain_instant_mm', row.get('precipitation_mm'))} mm<br>"
                    f"<b>Prediction:</b> {row.get('rain_forecast_mm', row.get('precipitation_mm'))} mm<br>"
                    f"<b>Distance LGV:</b> {row.get('distance_to_lgv_km')} km<br>"
                    f"<b>Selection:</b> {html.escape(mode)}<br>"
                    f"<b>Date obs:</b> {html.escape(str(row.get('date_obs_raw')))}"
                )
                weather_markers.append(
                    {
                        "lat": float(row["latitude"]),
                        "lon": float(row["longitude"]),
                        "station_id": str(row.get("station_id")),
                        "source": source,
                        "selection_mode": mode,
                        "rain_24h_mm": round(rain24, 2),
                        "rain_7d_mm": round(rain7, 2),
                        "rain_30d_mm": round(rain30, 2),
                        "rain_month_mm": round(rainmonth, 2),
                        "rain_class": rain_class,
                        "risk_level": rain_class,
                        "popup": popup,
                    }
                )

        river_markers = []
        for rp in self.river_points:
            info = rivers.get(rp.river, {})
            lat = info.get("station_latitude") if info.get("station_latitude") is not None else rp.latitude
            lon = info.get("station_longitude") if info.get("station_longitude") is not None else rp.longitude
            level = info.get("last_level_m")
            threshold = info.get("threshold_m")
            trend = info.get("trend_mph")
            rapid = info.get("rapid_rise_mph")

            if level is not None and threshold is not None and level >= threshold:
                color = "#dc2626"
                risk_level = "ELEVE"
            elif trend is not None and rapid is not None and trend >= rapid:
                color = "#d97706"
                risk_level = "MODERE"
            elif info.get("configured"):
                color = "#16a34a"
                risk_level = "FAIBLE"
            else:
                color = "#6b7280"
                risk_level = "INDETERMINE"

            popup = (
                f"<b>Cours d'eau:</b> {html.escape(rp.river)}<br>"
                f"<b>Source:</b> {html.escape(str(info.get('source', DataSource.VIGICRUES_HYDRO.value)))}<br>"
                f"<b>Station:</b> {html.escape(str(info.get('station_code', 'n/a')))}<br>"
                f"<b>Point mesure:</b> {round(float(lat), 5)}, {round(float(lon), 5)}<br>"
                f"<b>Niveau:</b> {html.escape(str(level))} m<br>"
                f"<b>Tendance:</b> {html.escape(str(trend))} m/h<br>"
                f"<b>Seuil:</b> {html.escape(str(threshold))} m<br>"
                f"<b>Auto-station:</b> {html.escape(str(info.get('auto_selected')))}<br>"
                f"<b>Risque:</b> {risk_level}"
            )
            river_markers.append({"lat": float(lat), "lon": float(lon), "color": color, "risk_level": risk_level, "popup": popup})

        soil_markers = []
        geotech_summary = {}
        if isinstance(geotech, dict):
            if isinstance(geotech.get("summary"), dict):
                geotech_summary = geotech.get("summary") or {}
            for point in geotech.get("points", []) if isinstance(geotech.get("points"), list) else []:
                popup = (
                    f"<b>Point LGV:</b> {point.get('point_id')}<br>"
                    f"<b>Risque geotechnique:</b> {html.escape(str(point.get('risk_level')))}<br>"
                    f"<b>Sol:</b> {html.escape(str(point.get('soil_type')))}<br>"
                    f"<b>RGA:</b> {html.escape(str(point.get('rga_label')))}<br>"
                    f"<b>MVT proches:</b> {html.escape(str(point.get('mvt_count')))}"
                )
                soil_markers.append(
                    {
                        "lat": float(point["latitude"]),
                        "lon": float(point["longitude"]),
                        "risk_level": str(point.get("risk_level", "INDETERMINE")),
                        "color": str(point.get("risk_color", "#6b7280")),
                        "popup": popup,
                    }
                )

        piezo_markers = []
        piezo_summary = {}
        if isinstance(piezometers, dict):
            if isinstance(piezometers.get("summary"), dict):
                piezo_summary = piezometers.get("summary") or {}
            for pz in piezometers.get("stations", []) if isinstance(piezometers.get("stations"), list) else []:
                reasons = pz.get("alert_reasons") or []
                reason_txt = " | ".join(reasons) if reasons else "Aucune alerte"
                popup = (
                    f"<b>Piezometre:</b> {html.escape(str(pz.get('code_bss')))}<br>"
                    f"<b>Nom:</b> {html.escape(str(pz.get('name')))}<br>"
                    f"<b>Risque nappe:</b> {html.escape(str(pz.get('risk_level')))}<br>"
                    f"<b>Profondeur nappe:</b> {html.escape(str(pz.get('depth_m')))} m<br>"
                    f"<b>Tendance profondeur:</b> {html.escape(str(pz.get('trend_depth_mpd')))} m/j<br>"
                    f"<b>Niveau NGF:</b> {html.escape(str(pz.get('level_mngf')))} m<br>"
                    f"<b>Distance LGV:</b> {html.escape(str(pz.get('distance_to_lgv_km')))} km<br>"
                    f"<b>Derniere mesure:</b> {html.escape(str(pz.get('last_date_utc')))}<br>"
                    f"<b>Alerte:</b> {html.escape(reason_txt)}"
                )
                piezo_markers.append(
                    {
                        "lat": float(pz["latitude"]),
                        "lon": float(pz["longitude"]),
                        "risk_level": str(pz.get("risk_level", "INDETERMINE")),
                        "color": str(pz.get("risk_color", "#6b7280")),
                        "popup": popup,
                    }
                )

        hydro_net_markers = []
        hydro_summary = {}
        if isinstance(hydro_network, dict):
            if isinstance(hydro_network.get("summary"), dict):
                hydro_summary = hydro_network.get("summary") or {}
            for st in hydro_network.get("stations", []) if isinstance(hydro_network.get("stations"), list) else []:
                popup = (
                    f"<b>Station Vigicrues:</b> {html.escape(str(st.get('station_code')))}<br>"
                    f"<b>Nom:</b> {html.escape(str(st.get('station_name')))}<br>"
                    f"<b>Cours d'eau:</b> {html.escape(str(st.get('river_name')))}<br>"
                    f"<b>Niveau:</b> {html.escape(str(st.get('last_level_m')))} m<br>"
                    f"<b>Tendance:</b> {html.escape(str(st.get('trend_mph')))} m/h<br>"
                    f"<b>Distance LGV:</b> {html.escape(str(st.get('distance_to_lgv_km')))} km<br>"
                    f"<b>Risque:</b> {html.escape(str(st.get('risk_level')))}"
                )
                hydro_net_markers.append(
                    {
                        "lat": float(st["latitude"]),
                        "lon": float(st["longitude"]),
                        "risk_level": str(st.get("risk_level", "INDETERMINE")),
                        "color": str(st.get("risk_color", "#6b7280")),
                        "popup": popup,
                    }
                )

        sector_markers = []
        sector_summary = {}
        if isinstance(sectors, dict):
            if isinstance(sectors.get("summary"), dict):
                sector_summary = sectors.get("summary") or {}
            for sec in sectors.get("sectors", []) if isinstance(sectors.get("sectors"), list) else []:
                popup = (
                    f"<b>Secteur:</b> {html.escape(str(sec.get('sector_id')))}<br>"
                    f"<b>Risque:</b> {html.escape(str(sec.get('risk_level')))}<br>"
                    f"<b>Score:</b> {html.escape(str(sec.get('score')))}<br>"
                    f"<b>Pluie 24h:</b> {html.escape(str(sec.get('weather_max_24h_mm')))} mm<br>"
                    f"<b>Pluie 7j:</b> {html.escape(str(sec.get('weather_max_7d_mm')))} mm<br>"
                    f"<b>Pluie 30j:</b> {html.escape(str(sec.get('weather_max_30d_mm')))} mm<br>"
                    f"<b>Geo/Piezo/Hydro:</b> {sec.get('geotech_points')}/{sec.get('piezometers')}/{sec.get('hydro_stations')}"
                )
                sector_markers.append(
                    {
                        "lat": float(sec["latitude"]),
                        "lon": float(sec["longitude"]),
                        "risk_level": str(sec.get("risk_level", "INDETERMINE")),
                        "color": str(sec.get("risk_color", "#6b7280")),
                        "popup": popup,
                    }
                )

        geotech_kpi = (
            f"Pts geotech {len(soil_markers)} | critique={geotech_summary.get('critical_points', 0)} "
            f"| eleve={geotech_summary.get('high_points', 0)} | modere={geotech_summary.get('moderate_points', 0)}"
            if soil_markers
            else "Contexte geotechnique non disponible"
        )
        piezo_kpi = (
            f"Piezometres={piezo_summary.get('stations_with_data', 0)} | nappe tres haute={piezo_summary.get('very_high_risk', 0)} "
            f"| nappe haute={piezo_summary.get('high_risk', 0)} | remontee rapide={piezo_summary.get('rapid_rise_alerts', 0)}"
            if piezo_markers
            else "Piezometres non disponibles"
        )
        hydro_kpi = (
            f"Stations hydro reseau={hydro_summary.get('stations_with_data', 0)} | critiques={hydro_summary.get('critical_stations', 0)} "
            f"| elevees={hydro_summary.get('high_stations', 0)}"
            if hydro_net_markers
            else "Reseau Vigicrues etendu non disponible"
        )
        sector_kpi = (
            f"Secteurs={sector_summary.get('sector_count', 0)} | critiques={sector_summary.get('critical', 0)} "
            f"| eleves={sector_summary.get('high', 0)} | sous surveillance={sector_summary.get('watch', 0)}"
            if sector_markers
            else "Secteurs de surveillance non disponibles"
        )

        lgv_json = json.dumps(lgv_lines)
        weather_json = json.dumps(weather_markers)
        river_json = json.dumps(river_markers)
        soil_json = json.dumps(soil_markers)
        piezo_json = json.dumps(piezo_markers)
        hydro_json = json.dumps(hydro_net_markers)
        sector_json = json.dumps(sector_markers)
        risk_level = html.escape(str(risks.get("risk_level")))
        score = html.escape(str(risks.get("score")))
        notice = html.escape(str(risks.get("weather_selection_notice") or "Aucun fallback meteo"))
        geotech_kpi_html = html.escape(geotech_kpi)
        piezo_kpi_html = html.escape(piezo_kpi)
        hydro_kpi_html = html.escape(hydro_kpi)
        sector_kpi_html = html.escape(sector_kpi)

        html_doc = f"""<!doctype html>
<html lang=\"fr\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>LGV SEA - Dashboard Hydrometeo</title>
  <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\" />
  <style>
    :root {{ --bg:#f4f7f9; --panel:#ffffff; --text:#12212c; --muted:#6b7280; }}
    body {{ margin:0; font-family:"Segoe UI",Tahoma,sans-serif; background:var(--bg); color:var(--text); }}
    .wrap {{ display:grid; grid-template-columns:390px 1fr; min-height:100vh; }}
    .panel {{ padding:16px; background:var(--panel); border-right:1px solid #e5e7eb; overflow-y:auto; max-height:100vh; }}
    #map {{ width:100%; height:100vh; }}
    .badge {{ display:inline-block; padding:4px 8px; border-radius:999px; background:#e6fffa; color:#134e4a; font-weight:600; }}
    .kpi {{ margin:10px 0; padding:12px; background:#f8fafc; border-radius:10px; border:1px solid #e5e7eb; }}
    .muted {{ color:var(--muted); font-size:13px; }}
    .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; }}
    .river-pin {{
      width: 14px;
      height: 14px;
      transform: rotate(45deg);
      border-radius: 2px;
      border: 1px solid #0f172a;
      opacity: 0.95;
    }}
    .filter-box {{
      position:absolute;
      top:10px;
      left:10px;
      z-index:1000;
      background:#ffffff;
      border:1px solid #d1d5db;
      border-radius:8px;
      padding:10px;
      font-size:12px;
      min-width:220px;
      box-shadow:0 2px 8px rgba(0,0,0,0.12);
    }}
    .filter-box label {{ display:block; margin-bottom:6px; color:#111827; }}
    .filter-box select {{ width:100%; margin-top:2px; margin-bottom:8px; }}
    @media (max-width:900px) {{ .wrap {{ grid-template-columns:1fr; }} .panel {{ border-right:none; border-bottom:1px solid #e5e7eb; max-height:none; }} #map {{ height:70vh; }} }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <aside class=\"panel\">
      <h2>LGV SEA Monitoring</h2>
      <div class=\"kpi\"><span class=\"badge\">Risque: {risk_level} (score {score}/4)</span></div>
      <div class=\"kpi\"><b>Selection meteo</b><br><span class=\"muted\">{notice}</span></div>
      <div class=\"kpi\"><b>Contexte geotechnique</b><br><span class=\"muted\">{geotech_kpi_html}</span></div>
      <div class=\"kpi\"><b>Niveaux de nappe</b><br><span class=\"muted\">{piezo_kpi_html}</span></div>
      <div class=\"kpi\"><b>Reseau Vigicrues etendu</b><br><span class=\"muted\">{hydro_kpi_html}</span></div>
      <div class=\"kpi\"><b>Secteurs de surveillance</b><br><span class=\"muted\">{sector_kpi_html}</span></div>
      <div class=\"kpi\"><b>Objectif</b><br><span class=\"muted\">Aider les experts genie civil: pluie, hydro, susceptibilite argiles, mouvements de terrain et piezometrie.</span></div>
      <div class=\"muted\">
        <div><span class=\"dot\" style=\"background:#111111\"></span>Pluie catastrophique (>=120 mm/24h)</div>
        <div><span class=\"dot\" style=\"background:#b91c1c\"></span>Pluie extreme (>=80 mm/24h)</div>
        <div><span class=\"dot\" style=\"background:#ea580c\"></span>Pluie forte (>=50 mm/24h)</div>
        <div><span class=\"dot\" style=\"background:#0f766e\"></span>Station meteo <= 1 km</div>
        <div><span class=\"dot\" style=\"background:#1d4ed8\"></span>Open-Meteo (grille LGV)</div>
        <div><span class=\"dot\" style=\"background:#16a34a\"></span>Cours d'eau normal</div>
        <div><span class=\"dot\" style=\"background:#d97706\"></span>Cours d'eau montee rapide</div>
        <div><span class=\"dot\" style=\"background:#dc2626\"></span>Cours d'eau depassement seuil</div>
        <div><span class=\"dot\" style=\"background:#7f1d1d\"></span>Geotech critique (argiles + MVT)</div>
        <div><span class=\"dot\" style=\"background:#b91c1c\"></span>Geotech eleve</div>
        <div><span class=\"dot\" style=\"background:#dc2626\"></span>Nappe tres haute (sol potentiellement gorge d'eau)</div>
        <div><span class=\"dot\" style=\"background:#ea580c\"></span>Nappe haute</div>
        <div><span class=\"dot\" style=\"background:#dc2626\"></span>Secteur surveillance elevee</div>
        <div><span class=\"dot\" style=\"background:#7f1d1d\"></span>Secteur surveillance critique</div>
      </div>
    </aside>
    <section style=\"position:relative;\">
      <div class=\"filter-box\">
        <label>Periode pluie
          <select id=\"periodSelect\">
            <option value=\"rain_24h_mm\">24h</option>
            <option value=\"rain_7d_mm\">7 jours</option>
            <option value=\"rain_30d_mm\">30 jours</option>
            <option value=\"rain_month_mm\">Mois courant</option>
          </select>
        </label>
        <label>Source meteo
          <select id=\"sourceSelect\">
            <option value=\"ALL\">Toutes</option>
            <option value=\"synop_meteofrance\">SYNOP</option>
            <option value=\"open_meteo\">Open-Meteo</option>
          </select>
        </label>
        <label>Niveau risque minimum
          <select id=\"riskSelect\">
            <option value=\"ALL\">Tous</option>
            <option value=\"MODERE\">Modere+</option>
            <option value=\"ELEVE\">Eleve+</option>
            <option value=\"CRITIQUE\">Critique</option>
          </select>
        </label>
      </div>
      <div id=\"map\"></div>
    </section>
  </div>

  <script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
  <script>
    const lgvLines = {lgv_json};
    const weather = {weather_json};
    const rivers = {river_json};
    const soils = {soil_json};
    const piezos = {piezo_json};
    const hydroNet = {hydro_json};
    const sectors = {sector_json};

    const map = L.map('map');
    L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom:18, attribution:'&copy; OpenStreetMap contributors' }}).addTo(map);

    const lgvLayer = L.layerGroup().addTo(map);
    const weatherLayer = L.layerGroup().addTo(map);
    const riverLayer = L.layerGroup().addTo(map);
    const soilLayer = L.layerGroup().addTo(map);
    const piezoLayer = L.layerGroup().addTo(map);
    const hydroLayer = L.layerGroup().addTo(map);
    const sectorLayer = L.layerGroup().addTo(map);

    const lineLayers = [];
    lgvLines.forEach(line => {{
      const lgvLatLng = line.map(p => [p.lat, p.lon]);
      const lyr = L.polyline(lgvLatLng, {{ color:'#0b4f6c', weight:3, opacity:0.85 }}).addTo(lgvLayer);
      lineLayers.push(lyr);
    }});

    function riskRank(level) {{
      const key = String(level || '').toUpperCase();
      const map = {{
        'INDETERMINE': 0,
        'NORMAL': 1, 'FAIBLE': 1,
        'VIGILANCE': 2, 'MODERE': 2,
        'ELEVE': 3,
        'TRES_ELEVE': 4, 'CRITIQUE': 4
      }};
      return map[key] || 0;
    }}

    function weatherClass(value, period) {{
      if (period === 'rain_24h_mm') {{
        if (value >= 120) return 'CRITIQUE';
        if (value >= 80) return 'ELEVE';
        if (value >= 50) return 'MODERE';
        if (value >= 20) return 'VIGILANCE';
        return 'NORMAL';
      }}
      if (period === 'rain_7d_mm') {{
        if (value >= 180) return 'CRITIQUE';
        if (value >= 120) return 'ELEVE';
        if (value >= 80) return 'MODERE';
        if (value >= 40) return 'VIGILANCE';
        return 'NORMAL';
      }}
      if (value >= 300) return 'CRITIQUE';
      if (value >= 220) return 'ELEVE';
      if (value >= 150) return 'MODERE';
      if (value >= 90) return 'VIGILANCE';
      return 'NORMAL';
    }}

    function weatherColor(cls, src, mode) {{
      if (cls === 'CRITIQUE') return '#111111';
      if (cls === 'ELEVE') return '#b91c1c';
      if (cls === 'MODERE') return '#ea580c';
      if (src === 'open_meteo') return '#1d4ed8';
      if (mode === 'within_1km') return '#0f766e';
      return '#f59e0b';
    }}

    function passMinRisk(level, minLevel) {{
      if (minLevel === 'ALL') return true;
      return riskRank(level) >= riskRank(minLevel);
    }}

    function renderLayers() {{
      weatherLayer.clearLayers();
      riverLayer.clearLayers();
      soilLayer.clearLayers();
      piezoLayer.clearLayers();
      hydroLayer.clearLayers();
      sectorLayer.clearLayers();

      const period = document.getElementById('periodSelect').value;
      const source = document.getElementById('sourceSelect').value;
      const minRisk = document.getElementById('riskSelect').value;

      weather.forEach(s => {{
        if (source !== 'ALL' && s.source !== source) return;
        const value = Number(s[period] || 0);
        const cls = weatherClass(value, period);
        if (!passMinRisk(cls, minRisk)) return;
        const color = weatherColor(cls, s.source, s.selection_mode);
        L.circleMarker([s.lat, s.lon], {{
          radius: 7, color: color, fillColor: color, fillOpacity: 0.85
        }}).addTo(weatherLayer).bindPopup(s.popup + `<br><b>Filtre:</b> ${{period}} = ${{value.toFixed(1)}} mm`);
      }});

      rivers.forEach(r => {{
        if (!passMinRisk(r.risk_level, minRisk)) return;
        const icon = L.divIcon({{
          className: '',
          html: `<div class="river-pin" style="background:${{r.color}}"></div>`,
          iconSize: [14, 14],
          iconAnchor: [7, 7]
        }});
        L.marker([r.lat, r.lon], {{ icon }}).addTo(riverLayer).bindPopup(r.popup);
      }});

      soils.forEach(s => {{
        if (!passMinRisk(s.risk_level, minRisk)) return;
        L.circleMarker([s.lat, s.lon], {{
          radius:6, color:s.color, fillColor:s.color, fillOpacity:0.75
        }}).addTo(soilLayer).bindPopup(s.popup);
      }});

      piezos.forEach(p => {{
        if (!passMinRisk(p.risk_level, minRisk)) return;
        L.circleMarker([p.lat, p.lon], {{
          radius:8, color:p.color, fillColor:p.color, fillOpacity:0.85, weight:2
        }}).addTo(piezoLayer).bindPopup(p.popup);
      }});

      hydroNet.forEach(h => {{
        if (!passMinRisk(h.risk_level, minRisk)) return;
        L.circleMarker([h.lat, h.lon], {{
          radius:7, color:h.color, fillColor:h.color, fillOpacity:0.8, weight:2
        }}).addTo(hydroLayer).bindPopup(h.popup);
      }});

      sectors.forEach(s => {{
        if (!passMinRisk(s.risk_level, minRisk)) return;
        L.circleMarker([s.lat, s.lon], {{
          radius: 9, color: s.color, fillColor: s.color, fillOpacity: 0.28, weight: 2
        }}).addTo(sectorLayer).bindPopup(s.popup);
      }});
    }}

    document.getElementById('periodSelect').addEventListener('change', renderLayers);
    document.getElementById('sourceSelect').addEventListener('change', renderLayers);
    document.getElementById('riskSelect').addEventListener('change', renderLayers);

    renderLayers();

    L.control.layers(null, {{
      "Trace LGV": lgvLayer,
      "Stations meteo": weatherLayer,
      "Cours d'eau": riverLayer,
      "Points geotechniques": soilLayer,
      "Piezometres": piezoLayer,
      "Vigicrues reseau": hydroLayer,
      "Secteurs surveillance": sectorLayer
    }}, {{ collapsed:false }}).addTo(map);

    const boundsGroup = [];
    lineLayers.forEach(x => boundsGroup.push(x));
    if (boundsGroup.length > 0) {{
      const allBounds = L.featureGroup(boundsGroup).getBounds();
      map.fitBounds(allBounds.pad(0.10));
    }} else {{
      map.setView([46.2, 0.2], 7);
    }}
  </script>
</body>
</html>
"""

        with open(map_path, "w", encoding="utf-8") as f:
            f.write(html_doc)
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(html_doc)
        return latest_path

    def run_cycle(self) -> None:
        logging.info("=" * 70)
        logging.info("Cycle LGV SEA monitoring")

        weather_pack = self.fetch_pluviometry_combined()
        rivers = self.fetch_all_river_levels()
        geotech = self.fetch_geotechnical_context()
        piezometers = self.fetch_piezometers_near_lgv()
        hydro_network = self.fetch_hydro_network_near_lgv()
        sectors = self.build_surveillance_sectors(weather_pack["selected"], geotech, piezometers, hydro_network)
        risks = self.analyze_risks(
            weather_pack["selected"],
            weather_pack.get("notice"),
            rivers,
            geotech,
            piezometers,
            hydro_network,
            sectors,
        )
        map_path = self.generate_map(weather_pack["selected"], rivers, risks, geotech, piezometers, hydro_network, sectors)

        cycle_output = {
            "timestamp_utc": self._now_utc().isoformat(),
            "risk_level": risks.get("risk_level"),
            "score": risks.get("score"),
            "alerts": risks.get("alerts"),
            "weather_notice": weather_pack.get("notice"),
            "geotech_summary": geotech.get("summary") if isinstance(geotech, dict) else None,
            "piezometer_summary": piezometers.get("summary") if isinstance(piezometers, dict) else None,
            "hydro_network_summary": hydro_network.get("summary") if isinstance(hydro_network, dict) else None,
            "sector_summary": sectors.get("summary") if isinstance(sectors, dict) else None,
            "map_path": map_path,
            "risk_report_path": risks.get("report_path"),
            "weather_summary_path": weather_pack.get("summary_path"),
        }
        ts = self._now_utc().strftime("%Y%m%d_%H%M%S")
        self._save_json(cycle_output, os.path.join("reports", f"cycle_summary_{ts}.json"))
        self._save_json(
            {
                "timestamp_utc": self._now_utc().isoformat(),
                "risk_level": risks.get("risk_level"),
                "alerts_count": len(risks.get("alerts", [])),
                "alerts": risks.get("alerts", []),
            },
            os.path.join("reports", f"alerts_{ts}.json"),
        )

        logging.info("RISK: %s (score %s), alerts=%s", risks["risk_level"], risks["score"], len(risks["alerts"]))
        logging.info("Carte: %s", map_path)
        self._print_console(weather_pack["selected"], rivers, risks, map_path)

    def _print_console(self, weather_df: pd.DataFrame, rivers: Dict[str, Dict[str, object]], risks: Dict[str, object], map_path: str) -> None:
        print("\n" + "=" * 70)
        print(f"LGV SEA - RISQUE HYDROMETEO: {risks['risk_level']} (score {risks['score']}/4)")
        print("=" * 70)

        notice = risks.get("weather_selection_notice")
        if notice:
            print(f"Notice meteo: {notice}")

        if not weather_df.empty:
            print(f"Stations meteo affichees: {len(weather_df)}")
            for _, row in weather_df.head(12).iterrows():
                print(
                    "  - station={sid} pluie={rain}mm dist_lgv={dist}km mode={mode}".format(
                        sid=row.get("station_id"),
                        rain=row.get("precipitation_mm"),
                        dist=row.get("distance_to_lgv_km"),
                        mode=row.get("selection_mode"),
                    )
                )

        print("\nRivieres:")
        for river, info in rivers.items():
            if info.get("last_level_m") is not None:
                print(f"  - {river}: {info['last_level_m']} m | trend {info.get('trend_mph')} m/h | thr {info.get('threshold_m')} | station {info.get('station_code')}")
            else:
                msg = info.get("message") or info.get("error") or "pas de donnees"
                print(f"  - {river}: {msg}")

        geotech = risks.get("details", {}).get("geotech", {})
        if isinstance(geotech, dict) and geotech:
            print(
                "\nGeotech: points={n} critique={c} eleve={h} modere={m}".format(
                    n=geotech.get("sample_count"),
                    c=geotech.get("critical_points"),
                    h=geotech.get("high_points"),
                    m=geotech.get("moderate_points"),
                )
            )

        groundwater = risks.get("details", {}).get("groundwater", {})
        if isinstance(groundwater, dict) and groundwater:
            print(
                "Nappes: stations={n} tres_haute={vh} haute={h} remontee_rapide={r}".format(
                    n=groundwater.get("stations_with_data"),
                    vh=groundwater.get("very_high_risk"),
                    h=groundwater.get("high_risk"),
                    r=groundwater.get("rapid_rise_alerts"),
                )
            )

        hydro_network = risks.get("details", {}).get("hydro_network", {})
        if isinstance(hydro_network, dict) and hydro_network:
            print(
                "Hydro reseau: stations={n} critiques={c} elevees={h} moderees={m}".format(
                    n=hydro_network.get("stations_with_data"),
                    c=hydro_network.get("critical_stations"),
                    h=hydro_network.get("high_stations"),
                    m=hydro_network.get("moderate_stations"),
                )
            )

        sector_summary = risks.get("details", {}).get("sectors", {})
        if isinstance(sector_summary, dict) and sector_summary:
            print(
                "Secteurs: total={n} critiques={c} eleves={h} moderes={m} surveillance={w}".format(
                    n=sector_summary.get("sector_count"),
                    c=sector_summary.get("critical"),
                    h=sector_summary.get("high"),
                    m=sector_summary.get("moderate"),
                    w=sector_summary.get("watch"),
                )
            )

        if risks["alerts"]:
            print("\nAlertes:")
            for alert in risks["alerts"]:
                print(f"  - [{alert.get('level')}] {alert.get('type')}: {alert.get('message')}")
        else:
            print("\nAucune alerte active")

        print(f"\nCarte HTML: {map_path}")
        print("=" * 70 + "\n")


def main() -> None:
    monitor = LGVSeaMonitor()
    monitor.run_cycle()
    schedule.every().hour.do(monitor.run_cycle)
    print("Monitoring actif (cycle horaire). Ctrl+C pour arreter.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        logging.info("Arret demande par l'utilisateur.")


if __name__ == "__main__":
    main()
