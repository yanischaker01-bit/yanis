from typing import Iterable, Tuple

from shapely.geometry import LineString, Point


def build_lgv_line(latlon_points: Iterable[Tuple[float, float]]) -> LineString:
    # Shapely expects lon/lat.
    return LineString([(lon, lat) for lat, lon in latlon_points])


def point_distance_to_line_km(lat: float, lon: float, line: LineString) -> float:
    # Planar approximation in degrees then converted to km.
    point = Point(lon, lat)
    distance_deg = point.distance(line)
    return distance_deg * 111.0

