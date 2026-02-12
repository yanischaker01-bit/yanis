import pandas as pd
from datetime import datetime
import os

# Configuration
input_file = r"D:\test_Leroy\NDVI_30_05_25.csv"
output_dir = r"D:\test_Leroy\organisé"
output_file = os.path.join(output_dir, f"NDVI_organise_{datetime.now().strftime('%d%m%Y')}.csv")
separator = ";"

# Créer le répertoire de sortie si nécessaire
os.makedirs(output_dir, exist_ok=True)

try:
    # Lecture du fichier CSV
    df = pd.read_csv(input_file, sep=separator, encoding='utf-8')
    
    print(f"Fichier original chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Nettoyage des données (exemples)
    # 1. Supprimer les doublons
    df = df.drop_duplicates()
    
    # 2. Supprimer les lignes avec valeurs manquantes importantes
    df = df.dropna(thresh=len(df.columns)-2)  # Garde les lignes avec au moins N-2 valeurs
    
    # 3. Trier les données (exemple par date si colonne existe)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date')
    
    # 4. Réorganiser les colonnes (ajuster selon vos besoins)
    cols = df.columns.tolist()
    # Mettre les colonnes importantes en premier (exemple)
    priority_cols = ['id', 'date', 'ndvi', 'geometry']  # A adapter
    other_cols = [col for col in cols if col not in priority_cols]
    new_col_order = priority_cols + other_cols
    df = df[new_col_order]
    
    # 5. Formater les nombres (exemple NDVI)
    if 'ndvi' in df.columns:
        df['ndvi'] = df['ndvi'].round(4)  # 4 décimales pour NDVI
    
    # Sauvegarde du fichier organisé
    df.to_csv(output_file, sep=separator, index=False, encoding='utf-8')
    
    print(f"Fichier organisé sauvegardé : {output_file}")
    print(f"Résumé : {df.shape[0]} lignes, {df.shape[1]} colonnes conservées")

except Exception as e:
    print(f"Erreur lors du traitement : {str(e)}")