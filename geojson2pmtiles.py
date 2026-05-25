#!/usr/bin/env python3
"""Convert GeoJSON files to PMTiles for carte_lgv.
Usage: python geojson2pmtiles.py [layer_name ...]
  No args = convert all layers.
  With args = only convert specified layers (e.g. bois n2).
"""

import gzip
import json
import os
import sys
from pathlib import Path

import fiona
import mapbox_vector_tile
import mercantile
from shapely.geometry import mapping, shape, box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from pmtiles.tile import TileType, Compression, zxy_to_tileid
from pmtiles.writer import Writer

DATA_DIR = Path(__file__).parent / "data"

LAYERS = [
    # (geojson_file, layer_name, min_zoom, max_zoom, encoding)
    ("bois.geojson",    "bois",    8,  17, "utf-8"),
    ("n2.geojson",      "n2",      6,  14, None),      # auto-detect utf-8/cp1252
    ("old.geojson",     "old",     7,  15, "utf-8"),
    ("mc.geojson",      "mc",      8,  16, "utf-8"),
    ("eco.geojson",     "eco",     8,  14, "utf-8"),
    ("oa_poly.geojson", "oa_poly", 10, 18, "utf-8"),
]


def read_geojson(path, encoding):
    """Read GeoJSON with encoding fallback for n2."""
    features, geoms = [], []
    encodings = [encoding] if encoding else ["utf-8", "cp1252"]

    for enc in encodings:
        try:
            with fiona.open(str(path), encoding=enc) as src:
                for feat in src:
                    try:
                        geom = shape(feat["geometry"])
                        if not geom.is_valid:
                            geom = geom.buffer(0)
                        if geom.is_empty:
                            continue
                        props = {k: v for k, v in (feat["properties"] or {}).items() if v is not None}
                        features.append({"geometry": None, "properties": props})
                        geoms.append(geom)
                    except Exception:
                        pass
            break  # success
        except Exception as e:
            if enc == encodings[-1]:
                raise
            print(f"  Encoding {enc} failed, trying next...")

    # Store geometry as mapping (needed for mapbox_vector_tile)
    for i, geom in enumerate(geoms):
        features[i]["geometry"] = mapping(geom)

    return features, geoms


def convert_layer(geojson_file, layer_name, min_zoom, max_zoom, encoding):
    geojson_path = DATA_DIR / geojson_file
    pmtiles_path = DATA_DIR / f"{layer_name}.pmtiles"

    if not geojson_path.exists():
        print(f"SKIP  {geojson_file} (not found)")
        return

    print(f"\n{'='*60}")
    print(f"  {geojson_file}  ->  {layer_name}.pmtiles  (z{min_zoom}-{max_zoom})")

    features, geoms = read_geojson(geojson_path, encoding)
    print(f"  {len(features)} features loaded")

    if not features:
        print("  No features — skipping.")
        return

    tree = STRtree(geoms)
    bounds = unary_union(geoms).bounds   # (minx, miny, maxx, maxy)
    print(f"  Bounds: {[round(b, 4) for b in bounds]}")

    total_tiles = 0

    with open(str(pmtiles_path), "wb") as f:
        writer = Writer(f)

        for zoom in range(min_zoom, max_zoom + 1):
            tiles = list(mercantile.tiles(bounds[0], bounds[1], bounds[2], bounds[3], zooms=zoom))
            written = 0

            for tile in tiles:
                tb = mercantile.bounds(tile)
                tile_box = box(tb.west, tb.south, tb.east, tb.north)

                candidate_idx = tree.query(tile_box, predicate="intersects")

                tile_feats = []
                for idx in candidate_idx:
                    geom = geoms[idx]
                    try:
                        clipped = geom.intersection(tile_box)
                    except Exception:
                        continue
                    if clipped.is_empty:
                        continue
                    # Skip degenerate results (points/lines from polygon intersection)
                    if clipped.geom_type not in ("Polygon", "MultiPolygon"):
                        # for line layers keep LineString too
                        if clipped.geom_type not in ("LineString", "MultiLineString"):
                            continue
                    tile_feats.append({
                        "geometry": mapping(clipped),
                        "properties": features[idx]["properties"],
                    })

                if not tile_feats:
                    continue

                try:
                    mvt = mapbox_vector_tile.encode(
                        {"name": layer_name, "features": tile_feats},
                        default_options={
                            "extents": 4096,
                            "quantize_bounds": (tb.west, tb.south, tb.east, tb.north),
                            "y_coord_down": False,
                        },
                    )
                    tileid = zxy_to_tileid(tile.z, tile.x, tile.y)
                    writer.write_tile(tileid, gzip.compress(mvt))
                    written += 1
                    total_tiles += 1
                except Exception as e:
                    pass  # skip bad tiles silently

            print(f"  z{zoom:2d}: {len(tiles):5d} candidate tiles  ->  {written} written")

        writer.finalize(
            {
                "tile_type": TileType.MVT,
                "tile_compression": Compression.GZIP,
                "min_lon_e7": int(bounds[0] * 1e7),
                "min_lat_e7": int(bounds[1] * 1e7),
                "max_lon_e7": int(bounds[2] * 1e7),
                "max_lat_e7": int(bounds[3] * 1e7),
            },
            {"name": layer_name, "format": "pbf"},
        )

    size_mb = pmtiles_path.stat().st_size / 1024 / 1024
    print(f"  => {total_tiles} tiles  |  {size_mb:.2f} MB  ->  {pmtiles_path.name}")


def main():
    wanted = set(sys.argv[1:])

    for geojson_file, layer_name, min_zoom, max_zoom, encoding in LAYERS:
        if wanted and layer_name not in wanted:
            continue
        convert_layer(geojson_file, layer_name, min_zoom, max_zoom, encoding)

    print("\n\nAll done!")


if __name__ == "__main__":
    main()
