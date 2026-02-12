import pandas as pd

input_file = r"D:\test_Leroy\NDVI_18_03.csv"
output_file = r"D:\test_Leroy\NDVI_18_03_filtered_pk_267_270.csv"

# Paramètres
sep = ";"  # Séparateur confirmé
target_column = "KmInit"  # Colonne contenant les kilomètres
pk_min = 267.0  # Nouvelle borne inférieure
pk_max = 270.0  # Nouvelle borne supérieure

try:
    # 1. Vérification initiale avec analyse approfondie
    df_sample = pd.read_csv(input_file, nrows=1000, sep=sep)
    print("Colonnes détectées:", df_sample.columns.tolist())
    
    if target_column not in df_sample.columns:
        # Essayer d'autres noms de colonnes possibles
        possible_columns = ['pk_deb', 'pk_fin', 'KmInit', 'PK', 'PointKilometrique', 'LRS_PK']
        for col in possible_columns:
            if col in df_sample.columns:
                target_column = col
                print(f"Colonne alternative trouvée: {target_column}")
                break
        else:
            raise ValueError(f"Aucune colonne PK trouvée. Colonnes disponibles:\n{df_sample.columns.tolist()}")

    # Analyse préliminaire
    print(f"\nAnalyse de la colonne {target_column} avant traitement:")
    print("Exemples de valeurs:", df_sample[target_column].head(10).tolist())
    print("Valeurs manquantes:", df_sample[target_column].isna().sum(), "/", len(df_sample))
    print("Valeurs uniques:", df_sample[target_column].nunique())
    
    # Conversion pour l'analyse des plages
    df_sample[target_column] = pd.to_numeric(
        df_sample[target_column].astype(str).str.replace(',', '.'), 
        errors='coerce'
    )
    print(f"Plage actuelle: {df_sample[target_column].min()} à {df_sample[target_column].max()}")
    
    # 2. Traitement du fichier
    filtered_data = []
    total_processed = 0
    conversion_errors = 0
    
    for chunk in pd.read_csv(input_file, chunksize=100000, sep=sep, 
                          engine='python', error_bad_lines=False):
        total_processed += len(chunk)
        
        # Conversion numérique robuste
        chunk[target_column] = (chunk[target_column]
                              .astype(str)
                              .str.replace('[^\d.,-]', '', regex=True)  # Nettoyage
                              .str.replace(',', '.')  # Standardisation décimale
                              .apply(pd.to_numeric, errors='coerce'))  # Conversion
        
        # Statistiques
        conversion_errors += chunk[target_column].isna().sum()
        current_converted = chunk[target_column].notna().sum()
        
        # Filtrage
        mask = (chunk[target_column] >= pk_min) & (chunk[target_column] <= pk_max)
        filtered = chunk[mask].copy()
        filtered_data.append(filtered)
        
        print(f"Lot traité: {len(chunk):,} lignes | Converties: {current_converted:,} | Filtrees: {len(filtered):,}")

    # 3. Consolidation des résultats
    if filtered_data:
        result = pd.concat(filtered_data)
        print(f"\nRÉSULTATS FINAUX:")
        print(f"- Lignes traitées totales: {total_processed:,}")
        print(f"- Erreurs de conversion: {conversion_errors:,}")
        print(f"- Lignes filtrées retenues: {len(result):,}")
        
        if len(result) > 0:
            # Statistiques détaillées
            print("\nStatistiques des données filtrées:")
            print(f"PK min: {result[target_column].min():.3f}")
            print(f"PK max: {result[target_column].max():.3f}")
            print(f"Moyenne PK: {result[target_column].mean():.3f}")
            
            # Export avec les colonnes les plus utiles
            cols_to_export = [target_column, 'NDVI', 'Code', 'Libellé', 'Voie_COS', 'Surface_3D', 'type_ot']
            cols_to_export = [col for col in cols_to_export if col in result.columns]
            
            result[cols_to_export].to_csv(output_file, index=False, sep=sep, encoding='utf-8-sig')
            print(f"\nExport réussi vers {output_file}")
            print("Colonnes exportées:", cols_to_export)
            print("\nAperçu des données exportées:")
            print(result[cols_to_export].head())
        else:
            print("\nAUCUNE DONNÉE dans l'intervalle spécifié.")
            print("Suggestions:")
            print(f"1. Vérifiez la plage PK dans vos données (actuelle: {df_sample[target_column].min()} à {df_sample[target_column].max()})")
            print("2. Essayez d'autres colonnes PK disponibles:", [col for col in df_sample.columns if 'pk' in col.lower() or 'km' in col.lower()])
            print("3. Ajustez les bornes si nécessaire (actuelles: {pk_min} à {pk_max})")
    else:
        print("Aucune donnée n'a pu être filtrée.")

except Exception as e:
    print(f"ERREUR CRITIQUE: {str(e)}")
    if 'df_sample' in locals():
        print("\nCOLONNES DISPONIBLES:", df_sample.columns.tolist())
        print("\nSTATISTIQUES DES COLONNES PK:")
        pk_cols = [col for col in df_sample.columns if 'pk' in col.lower() or 'km' in col.lower()]
        if pk_cols:
            print(df_sample[pk_cols].describe())