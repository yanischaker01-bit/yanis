#!/usr/bin/env python3
"""Build pk_metric.pmtiles directly from LRS PK 1M shapefile (EPSG:2154 → WGS84).
Usage: python build_pk_metric.py
Output: data/pk_metric.pmtiles  (z17-19, labels PK métriques)
"""
import gzip, sys
from pathlib import Path
import fiona
import mapbox_vector_tile
import mercantile
from pyproj import Transformer
from shapely.geometry import Point, box
from shapely.strtree import STRtree
from pmtiles.tile import TileType, Compression, zxy_to_tileid
from pmtiles.writer import Writer

SHP   = Path(r"C:\Users\YCHAKER\Desktop\SIG\pk\LRS PK 1M.shp")
OUT   = Path(__file__).parent / "data" / "pk_metric.pmtiles"
LAYER = "pk_metric"
MIN_Z, MAX_Z = 17, 19   # uniquement zoom terrain

def main():
    if not SHP.exists():
        sys.exit(f"Introuvable : {SHP}")

    print("Reprojection EPSG:2154 -> WGS84 ...")
    tf = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    features, geoms = [], []
    with fiona.open(str(SHP)) as src:
        total = len(src)
        for i, feat in enumerate(src):
            if i % 100000 == 0:
                print(f"  {i}/{total} …")
            coords = feat["geometry"]["coordinates"]
            lng, lat = tf.transform(coords[0], coords[1])
            pk_m = int(round(feat["properties"]["pk"] or 0))
            voie = str(feat["properties"]["voie"] or "")
            label = f"{pk_m}"
            geom = Point(lng, lat)
            features.append({"geometry": None,
                              "properties": {"pk": pk_m, "voie": voie, "_pk_label": label}})
            geoms.append(geom)

    print(f"  {len(features)} points chargés.")

    # Bounding box
    from shapely.ops import unary_union
    all_geom = unary_union(geoms)
    bounds = all_geom.bounds  # (minx,miny,maxx,maxy)
    print(f"  Bounds WGS84 : {[round(b,4) for b in bounds]}")

    tree = STRtree(geoms)
    total_tiles = 0

    with open(str(OUT), "wb") as f:
        writer = Writer(f)

        for zoom in range(MIN_Z, MAX_Z + 1):
            tiles = list(mercantile.tiles(bounds[0], bounds[1], bounds[2], bounds[3], zooms=zoom))
            written = 0

            for tile in tiles:
                tb  = mercantile.bounds(tile)
                tbox = box(tb.west, tb.south, tb.east, tb.north)
                idxs = tree.query(tbox, predicate="intersects")
                if not len(idxs):
                    continue

                tile_feats = []
                for idx in idxs:
                    from shapely.geometry import mapping
                    tile_feats.append({
                        "geometry": mapping(geoms[idx]),
                        "properties": features[idx]["properties"],
                    })

                try:
                    mvt = mapbox_vector_tile.encode(
                        {LAYER: {"features": tile_feats}},
                        default_options={
                            "extents": 4096,
                            "quantize_bounds": (tb.west, tb.south, tb.east, tb.north),
                            "y_coord_down": False,
                        },
                    )
                    writer.write_tile(zxy_to_tileid(tile.z, tile.x, tile.y), gzip.compress(mvt))
                    written += 1
                    total_tiles += 1
                except Exception:
                    pass

            print(f"  z{zoom:2d}: {len(tiles):6d} candidats  ->  {written} ecrits")

        writer.finalize(
            {"tile_type": TileType.MVT, "tile_compression": Compression.GZIP,
             "min_lon_e7": int(bounds[0]*1e7), "min_lat_e7": int(bounds[1]*1e7),
             "max_lon_e7": int(bounds[2]*1e7), "max_lat_e7": int(bounds[3]*1e7)},
            {"name": LAYER, "format": "pbf"},
        )

    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"\nDone: {total_tiles} tuiles  |  {size_mb:.1f} MB  ->  {OUT.name}")

if __name__ == "__main__":
    main()
