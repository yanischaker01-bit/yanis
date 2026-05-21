import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from openpyxl.utils import get_column_letter

# === Chemins ===
path_surfaces = r"C:\Users\ychaker\Downloads\COS_GMAO\partie_nord.shp"
path_oa = r"C:\Users\ychaker\Downloads\shp\TEC_T_GENIECIVIL_PO.shp"
output_excel = r"C:\Users\ychaker\Desktop\SIG\rapport_SURF3D_ordre_spatial____nord.xlsx"

def main():
    # 1. Chargement des surfaces avec leur géométrie
    print("📍 Chargement des surfaces...")
    surf = gpd.read_file(path_surfaces).to_crs(epsg=2154)

    if surf.empty:
        raise ValueError("❌ Aucune surface trouvée dans le fichier.")
    print(f"✅ {len(surf)} surfaces chargées")

    # 2. Calcul de l'ordre spatial nord-sud
    print("🔢 Calcul de l'ordre spatial nord-sud...")
    surf["Y_centroid"] = surf.geometry.centroid.y
    surf["ordre_spatial"] = surf["Y_centroid"].rank(ascending=False, method="first")

    # 3. Chargement de la couche OA (PK_DEB + COMMUNE)
    print("📍 Chargement des OA génie civil...")
    oa = gpd.read_file(path_oa)

    oa_ref = (
        oa[["LIBELLE", "PK_DEB", "COMMUNE"]]
        .drop_duplicates("LIBELLE")
        .rename(columns={"LIBELLE": "OA_D"})
    )

    # 4. Fonction pour conserver l'ordre spatial dans les regroupements
    def add_spatial_order(df):
        order_mapping = (
            surf[["OA_D", "ordre_spatial"]]
            .drop_duplicates("OA_D")
            .set_index("OA_D")["ordre_spatial"]
        )
        df["ordre_spatial"] = df["OA_D"].map(order_mapping)
        return df.sort_values("ordre_spatial").drop(columns="ordre_spatial")

    feuilles = {}

    # Rapport 1: Par OA_D, OA_F, Int_Ext, Voie_COS
    cols1 = ["OA_D", "OA_F", "Int_Ext", "Voie_COS"]
    if all(col in surf.columns for col in cols1):
        df1 = (
            surf.groupby(cols1, observed=True)
            .agg(
                SURF_3D_Totale=("Surface_3D", "sum"),
                Y_moyen=("Y_centroid", "mean"),
            )
            .reset_index()
            .pipe(add_spatial_order)
            .merge(oa_ref, on="OA_D", how="left")   # ✅ AJOUT
        )
        feuilles["Surface_3D_par_OA"] = df1

    # Rapport 2: Par Modèle_du, Int_Ext, Voie_COS
    cols2 = ["OA_D", "OA_F", "Modèle_du", "Int_Ext", "Voie_COS"]
    if all(col in surf.columns for col in cols2):
        df2 = (
            surf.groupby(cols2, observed=True)
            .agg(
                SURF_3D_Totale=("Surface_3D", "sum"),
                Count=("Surface_3D", "count"),
            )
            .reset_index()
        )
        order_mapping = (
            surf[["OA_D", "ordre_spatial"]]
            .drop_duplicates("OA_D")
            .set_index("OA_D")["ordre_spatial"]
        )
        df2["ordre_spatial"] = df2["OA_D"].map(order_mapping)
        df2 = (
            df2.sort_values("ordre_spatial")
            .drop(columns="ordre_spatial")
            .merge(oa_ref, on="OA_D", how="left")   # ✅ AJOUT
        )
        feuilles["Surface_3D_par_modele"] = df2

    # Rapport 3: Par OA_D et OA_F uniquement
    cols3 = ["OA_D", "OA_F", "Surface_3D"]
    if all(col in surf.columns for col in cols3):
        df3 = (
            surf.groupby(["OA_D", "OA_F"], observed=True)
            .agg(SURF_3D_Totale=("Surface_3D", "sum"))
            .reset_index()
        )
        order_mapping = (
            surf[["OA_D", "ordre_spatial"]]
            .drop_duplicates("OA_D")
            .set_index("OA_D")["ordre_spatial"]
        )
        df3["ordre_spatial"] = df3["OA_D"].map(order_mapping)
        df3 = (
            df3.sort_values("ordre_spatial")
            .drop(columns="ordre_spatial")
            .merge(oa_ref, on="OA_D", how="left")   # ✅ AJOUT
        )
        feuilles["Surface_3D_par_OAD_OAF"] = df3

    # Export Excel (inchangé)
    if feuilles:
        print(f"💾 Export vers {output_excel}")
        with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
            for nom_feuille, df in feuilles.items():
                sheet_name = (
                    (nom_feuille[:28] + "..")
                    if len(nom_feuille) > 31
                    else nom_feuille
                )
                sheet_name = "".join(
                    c for c in sheet_name if c.isalnum() or c in ["_", " "]
                )

                df.to_excel(writer, sheet_name=sheet_name, index=False)

                worksheet = writer.sheets[sheet_name]
                for idx, col in enumerate(df.columns, 1):
                    col_letter = get_column_letter(idx)
                    max_len = (
                        max(df[col].astype(str).map(len).max(), len(col)) + 2
                    )
                    worksheet.column_dimensions[col_letter].width = min(
                        max_len, 50
                    )

        print("✅ Rapport généré avec PK_DEB et COMMUNE ajoutés (sans modification de l’existant).")
    else:
        print("❌ Aucune donnée à exporter.")

if __name__ == "__main__":
    main()
