import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from openpyxl.utils import get_column_letter

# === Chemins ===
path_surfaces = r"C:\Users\ychaker\Desktop\TEMP\Partie_nord__.shp" 
output_excel = r"C:\Users\ychaker\Desktop\SIG\rapport_SURF3D_ordre_spatial____Nord____.xlsx"

def main():
    # 1. Chargement des surfaces avec leur géométrie
    print("📍 Chargement des surfaces...")
    surf = gpd.read_file(path_surfaces).to_crs(epsg=2154)
    
    if surf.empty:
        raise ValueError("❌ Aucune surface trouvée dans le fichier.")
    print(f"✅ {len(surf)} surfaces chargées")
    
    # Afficher les colonnes disponibles pour débogage
    print(f"📋 Colonnes disponibles: {list(surf.columns)}")
    
    # Normaliser les noms de colonnes (s'assurer qu'on utilise les bonnes)
    # Les colonnes dans votre shapefile sont en minuscules
    print("\n🔍 Noms de colonnes identifiés:")
    print(f"  - surface_3d: {'surface_3d' in surf.columns}")
    print(f"  - surface_2d: {'surface_2d' in surf.columns}")
    print(f"  - oa_d: {'oa_d' in surf.columns}")
    print(f"  - oa_f: {'oa_f' in surf.columns}")
    print(f"  - int_ext: {'int_ext' in surf.columns}")
    print(f"  - voie_cos: {'voie_cos' in surf.columns}")
    print(f"  - modèle_du: {'modèle_du' in surf.columns}")
    
    # 2. Calcul de l'ordre spatial nord-sud
    print("\n🔢 Calcul de l'ordre spatial nord-sud...")
    surf["Y_centroid"] = surf.geometry.centroid.y
    surf["ordre_spatial"] = surf["Y_centroid"].rank(ascending=False, method='first')
    
    # 3. Fonction pour conserver l'ordre spatial dans les regroupements
    def add_spatial_order(df):
        # Créer un mapping oa_d -> ordre_spatial
        order_mapping = surf[['oa_d', 'ordre_spatial']].drop_duplicates('oa_d').set_index('oa_d')['ordre_spatial']
        df['ordre_spatial'] = df['oa_d'].map(order_mapping)
        return df.sort_values('ordre_spatial').drop(columns='ordre_spatial')

    feuilles = {}

    # Rapport 1: Par oa_d, oa_f, int_ext, voie_cos (pour surface_3d et surface_2d)
    cols1 = ["oa_d", "oa_f", "int_ext", "voie_cos"]
    if all(col in surf.columns for col in cols1):
        print("\n📊 Création du rapport 1...")
        # Pour surface_3d
        df1_3d = (surf.groupby(cols1, observed=True)
                  .agg(SURF_3D_Totale=('surface_3d', 'sum'),
                       SURF_3D_Moyenne=('surface_3d', 'mean'),
                       Y_moyen=('Y_centroid', 'mean'),
                       Nombre_Elements=('surface_3d', 'count'))
                  .reset_index()
                  .pipe(add_spatial_order))
        feuilles["Surface_3D_par_OA"] = df1_3d
        print(f"  ✅ Surface_3D_par_OA: {len(df1_3d)} lignes")
        
        # Pour surface_2d si la colonne existe
        if 'surface_2d' in surf.columns:
            df1_2d = (surf.groupby(cols1, observed=True)
                      .agg(SURF_2D_Totale=('surface_2d', 'sum'),
                           SURF_2D_Moyenne=('surface_2d', 'mean'),
                           Y_moyen=('Y_centroid', 'mean'),
                           Nombre_Elements=('surface_2d', 'count'))
                      .reset_index()
                      .pipe(add_spatial_order))
            feuilles["Surface_2D_par_OA"] = df1_2d
            print(f"  ✅ Surface_2D_par_OA: {len(df1_2d)} lignes")
    else:
        print(f"⚠️ Colonnes manquantes pour rapport 1: {[col for col in cols1 if col not in surf.columns]}")

    # Rapport 2: Par modèle_du, int_ext, voie_cos
    cols2 = ["oa_d", "oa_f", "modèle_du", "int_ext", "voie_cos"]
    if all(col in surf.columns for col in cols2):
        print("\n📊 Création du rapport 2...")
        # Pour surface_3d
        df2_3d = (surf.groupby(cols2, observed=True)
                  .agg(SURF_3D_Totale=('surface_3d', 'sum'),
                       SURF_3D_Moyenne=('surface_3d', 'mean'),
                       Count=('surface_3d', 'count'))
                  .reset_index())
        
        # Appliquer l'ordre spatial
        order_mapping = surf[['oa_d', 'ordre_spatial']].drop_duplicates('oa_d').set_index('oa_d')['ordre_spatial']
        df2_3d['ordre_spatial'] = df2_3d['oa_d'].map(order_mapping)
        df2_3d = df2_3d.sort_values('ordre_spatial').drop(columns='ordre_spatial')
        feuilles["Surface_3D_par_modele"] = df2_3d
        print(f"  ✅ Surface_3D_par_modele: {len(df2_3d)} lignes")
        
        # Pour surface_2d si la colonne existe
        if 'surface_2d' in surf.columns:
            df2_2d = (surf.groupby(cols2, observed=True)
                      .agg(SURF_2D_Totale=('surface_2d', 'sum'),
                           SURF_2D_Moyenne=('surface_2d', 'mean'),
                           Count=('surface_2d', 'count'))
                      .reset_index())
            
            df2_2d['ordre_spatial'] = df2_2d['oa_d'].map(order_mapping)
            df2_2d = df2_2d.sort_values('ordre_spatial').drop(columns='ordre_spatial')
            feuilles["Surface_2D_par_modele"] = df2_2d
            print(f"  ✅ Surface_2D_par_modele: {len(df2_2d)} lignes")
    else:
        print(f"⚠️ Colonnes manquantes pour rapport 2: {[col for col in cols2 if col not in surf.columns]}")

    # Rapport 3: Par oa_d et oa_f uniquement
    cols3 = ["oa_d", "oa_f"]
    if all(col in surf.columns for col in cols3):
        print("\n📊 Création du rapport 3...")
        # Pour surface_3d
        df3_3d = (surf.groupby(cols3, observed=True)
                  .agg(SURF_3D_Totale=('surface_3d', 'sum'),
                       SURF_3D_Moyenne=('surface_3d', 'mean'),
                       Nombre_Elements=('surface_3d', 'count'))
                  .reset_index()
                  .pipe(add_spatial_order))
        feuilles["Surface_3D_par_OAD_OAF"] = df3_3d
        print(f"  ✅ Surface_3D_par_OAD_OAF: {len(df3_3d)} lignes")
        
        # Pour surface_2d si la colonne existe
        if 'surface_2d' in surf.columns:
            df3_2d = (surf.groupby(cols3, observed=True)
                      .agg(SURF_2D_Totale=('surface_2d', 'sum'),
                           SURF_2D_Moyenne=('surface_2d', 'mean'),
                           Nombre_Elements=('surface_2d', 'count'))
                      .reset_index()
                      .pipe(add_spatial_order))
            feuilles["Surface_2D_par_OAD_OAF"] = df3_2d
            print(f"  ✅ Surface_2D_par_OAD_OAF: {len(df3_2d)} lignes")
    else:
        print(f"⚠️ Colonnes manquantes pour rapport 3: {[col for col in cols3 if col not in surf.columns]}")

    # Rapport 4: Comparaison surface_3d vs surface_2d (si les deux existent)
    if 'surface_3d' in surf.columns and 'surface_2d' in surf.columns and 'oa_d' in surf.columns:
        print("\n📊 Création du rapport de comparaison...")
        df4 = (surf.groupby(["oa_d", "oa_f"], observed=True)
               .agg(SURF_3D_Totale=('surface_3d', 'sum'),
                    SURF_2D_Totale=('surface_2d', 'sum'),
                    Ratio_3D_2D=('surface_3d', lambda x: x.sum() / surf.loc[x.index, 'surface_2d'].sum() if surf.loc[x.index, 'surface_2d'].sum() != 0 else 0),
                    Nombre_Elements=('surface_3d', 'count'))
               .reset_index()
               .pipe(add_spatial_order))
        feuilles["Comparaison_3D_2D"] = df4
        print(f"  ✅ Comparaison_3D_2D: {len(df4)} lignes")

    # Export Excel
    if feuilles:
        print(f"\n💾 Export de {len(feuilles)} feuilles vers {output_excel}")
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            for nom_feuille, df in feuilles.items():
                # Nettoyer le nom de la feuille pour Excel
                sheet_name = (nom_feuille[:28] + '..') if len(nom_feuille) > 31 else nom_feuille
                sheet_name = ''.join(c for c in sheet_name if c.isalnum() or c in ['_', ' '])
                
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Ajuster la largeur des colonnes
                worksheet = writer.sheets[sheet_name]
                for idx, col in enumerate(df.columns, 1):
                    col_letter = get_column_letter(idx)
                    max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.column_dimensions[col_letter].width = min(max_len, 50)
        
        print(f"\n✅ Rapport généré avec succès! {len(feuilles)} feuilles créées.")
        print(f"📊 Feuilles créées:")
        for i, name in enumerate(feuilles.keys(), 1):
            print(f"  {i}. {name}: {len(feuilles[name])} lignes")
    else:
        print("\n❌ Aucune donnée à exporter. Vérifiez les noms de colonnes dans votre shapefile.")

if __name__ == "__main__":
    main()