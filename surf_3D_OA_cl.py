import geopandas as gpd
import pandas as pd
from openpyxl.utils import get_column_letter

# === Chemins ===
path_surfaces = r"C:\Users\ychaker\Desktop\TEMP\Partie_nord__.shp"
path_genie_civil = r"C:\Users\ychaker\Downloads\shp\TEC_T_GENIECIVIL_PO.shp"
output_excel = r"C:\Users\ychaker\Desktop\SIG\rapport_SURF3D_ordre_spatial____Nord_____28_01.xlsx"


def main():

    # =========================================================
    # 1. Chargement des surfaces
    # =========================================================
    print("📍 Chargement des surfaces...")
    surf = gpd.read_file(path_surfaces).to_crs(epsg=2154)

    if surf.empty:
        raise ValueError("❌ Aucune surface trouvée.")

    print(f"✅ {len(surf)} surfaces chargées")

    # =========================================================
    # 2. Génie civil (référentiel PK / commune)
    # =========================================================
    print("\n📍 Chargement de la couche Génie Civil...")
    gc = gpd.read_file(path_genie_civil)

    required_cols = ["LIBELLE", "PK_DEB", "COMMUNE"]
    missing = [c for c in required_cols if c not in gc.columns]
    if missing:
        raise ValueError(f"❌ Colonnes manquantes : {missing}")

    gc_ref = (
        gc[["LIBELLE", "PK_DEB", "COMMUNE"]]
        .drop_duplicates("LIBELLE")
        .rename(columns={"LIBELLE": "oa_d"})
    )

    gc_ref["PK_D"] = gc_ref["PK_DEB"] / 1000
    print(f"✅ {len(gc_ref)} OA de référence chargés")

    # =========================================================
    # 3. Ordre spatial Nord → Sud
    # =========================================================
    surf["Y_centroid"] = surf.geometry.centroid.y
    surf["ordre_spatial"] = surf["Y_centroid"].rank(ascending=False, method="first")

    def add_spatial_order(df):
        order_mapping = (
            surf[["oa_d", "ordre_spatial"]]
            .drop_duplicates("oa_d")
            .set_index("oa_d")["ordre_spatial"]
        )
        df["ordre_spatial"] = df["oa_d"].map(order_mapping)
        return df.sort_values("ordre_spatial").drop(columns="ordre_spatial")

    feuilles = {}

    # =========================================================
    # 4. SURFACE 3D CLÔTURE (modèle_du == 'CL')
    # =========================================================
    if {"modèle_du", "surface_3d", "oa_d", "oa_f"}.issubset(surf.columns):
        surf_cl = (
            surf[surf["modèle_du"] == "CL"]
            .groupby(["oa_d", "oa_f"], observed=True)
            .agg(SURF_3D_Cloture=("surface_3d", "sum"))
            .reset_index()
        )
    else:
        surf_cl = pd.DataFrame(columns=["oa_d", "oa_f", "SURF_3D_Cloture"])

    # =========================================================
    # 5. RAPPORT 1 – par OA / int_ext / voie
    # =========================================================
    cols1 = ["oa_d", "oa_f", "int_ext", "voie_cos"]
    if all(c in surf.columns for c in cols1):

        feuilles["Surface_3D_par_OA"] = (
            surf.groupby(cols1, observed=True)
            .agg(
                SURF_3D_Totale=("surface_3d", "sum"),
                SURF_3D_Moyenne=("surface_3d", "mean"),
                Nombre_Elements=("surface_3d", "count"),
            )
            .reset_index()
            .pipe(add_spatial_order)
        )

        if "surface_2d" in surf.columns:
            feuilles["Surface_2D_par_OA"] = (
                surf.groupby(cols1, observed=True)
                .agg(
                    SURF_2D_Totale=("surface_2d", "sum"),
                    SURF_2D_Moyenne=("surface_2d", "mean"),
                    Nombre_Elements=("surface_2d", "count"),
                )
                .reset_index()
                .pipe(add_spatial_order)
            )

    # =========================================================
    # 6. RAPPORT 2 – par modèle
    # =========================================================
    cols2 = ["oa_d", "oa_f", "modèle_du", "int_ext", "voie_cos"]
    if all(c in surf.columns for c in cols2):

        feuilles["Surface_3D_par_modele"] = (
            surf.groupby(cols2, observed=True)
            .agg(
                SURF_3D_Totale=("surface_3d", "sum"),
                SURF_3D_Moyenne=("surface_3d", "mean"),
                Count=("surface_3d", "count"),
            )
            .reset_index()
            .pipe(add_spatial_order)
        )

        if "surface_2d" in surf.columns:
            feuilles["Surface_2D_par_modele"] = (
                surf.groupby(cols2, observed=True)
                .agg(
                    SURF_2D_Totale=("surface_2d", "sum"),
                    SURF_2D_Moyenne=("surface_2d", "mean"),
                    Count=("surface_2d", "count"),
                )
                .reset_index()
                .pipe(add_spatial_order)
            )

    # =========================================================
    # 7. RAPPORT 3 – par couple OA
    # =========================================================
    cols3 = ["oa_d", "oa_f"]
    if all(c in surf.columns for c in cols3):

        feuilles["Surface_3D_par_OAD_OAF"] = (
            surf.groupby(cols3, observed=True)
            .agg(
                SURF_3D_Totale=("surface_3d", "sum"),
                SURF_3D_Moyenne=("surface_3d", "mean"),
                Nombre_Elements=("surface_3d", "count"),
            )
            .reset_index()
            .pipe(add_spatial_order)
        )

        if "surface_2d" in surf.columns:
            feuilles["Surface_2D_par_OAD_OAF"] = (
                surf.groupby(cols3, observed=True)
                .agg(
                    SURF_2D_Totale=("surface_2d", "sum"),
                    SURF_2D_Moyenne=("surface_2d", "mean"),
                    Nombre_Elements=("surface_2d", "count"),
                )
                .reset_index()
                .pipe(add_spatial_order)
            )

    # =========================================================
    # 8. RAPPORT 4 – Comparaison 3D / 2D
    # =========================================================
    if {"surface_3d", "surface_2d"}.issubset(surf.columns):

        feuilles["Comparaison_3D_2D"] = (
            surf.groupby(["oa_d", "oa_f"], observed=True)
            .agg(
                SURF_3D_Totale=("surface_3d", "sum"),
                SURF_2D_Totale=("surface_2d", "sum"),
                Ratio_3D_2D=(
                    "surface_3d",
                    lambda x: x.sum()
                    / surf.loc[x.index, "surface_2d"].sum()
                    if surf.loc[x.index, "surface_2d"].sum() != 0
                    else 0,
                ),
                Nombre_Elements=("surface_3d", "count"),
            )
            .reset_index()
            .pipe(add_spatial_order)
        )

    # =========================================================
    # 9. Ajout PK / COMMUNE + Surface Clôture
    # =========================================================
    corrections_pk = {
        "PK 0+700 RCO1": {"PK_DEB": 700, "COMMUNE": "LA COURONNE"},
        "PRA CE1 011": {"PK_DEB": 1200, "COMMUNE": "LA COURONNE"},
        "PRO CE0 0022": {"PK_DEB": 2300, "COMMUNE": "LA COURONNE"},
        "PRA CE2 033": {"PK_DEB": 3700, "COMMUNE": "LA COURONNE"},
    }

    for nom, df in feuilles.items():

        if not {"oa_d", "oa_f"}.issubset(df.columns):
            continue

        # ➕ Surface 3D Clôture
        df = df.merge(surf_cl, on=["oa_d", "oa_f"], how="left")
        df["SURF_3D_Cloture"] = df["SURF_3D_Cloture"].fillna(0)

        # ➕ PK / commune
        df = df.merge(gc_ref, on="oa_d", how="left")

        mask_zero = df["PK_D"].fillna(0) == 0
        for oa, vals in corrections_pk.items():
            idx = mask_zero & (df["oa_d"] == oa)
            df.loc[idx, "PK_DEB"] = vals["PK_DEB"]
            df.loc[idx, "COMMUNE"] = vals["COMMUNE"]
            df.loc[idx, "PK_D"] = vals["PK_DEB"] / 1000

        cols = ["PK_D"] + [c for c in df.columns if c != "PK_D"]
        feuilles[nom] = df[cols]

    # =========================================================
    # 10. EXPORT EXCEL
    # =========================================================
    print(f"\n💾 Export vers {output_excel}")
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        for nom_feuille, df in feuilles.items():
            sheet_name = (nom_feuille[:28] + "..") if len(nom_feuille) > 31 else nom_feuille
            sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in ["_", " "])

            df.to_excel(writer, sheet_name=sheet_name, index=False)

            ws = writer.sheets[sheet_name]
            for i, col in enumerate(df.columns, 1):
                col_letter = get_column_letter(i)
                width = max(df[col].astype(str).map(len).max(), len(col)) + 2
                ws.column_dimensions[col_letter].width = min(width, 50)

    print("✅ Excel généré avec Surface 3D Clôture par couple OA.")


if __name__ == "__main__":
    main()
