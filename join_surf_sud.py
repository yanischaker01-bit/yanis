import pandas as pd

# === Chemins ===
rapport_path = r"C:\Users\ychaker\Desktop\SIG\rapport_SURF3D_ordre_spatial___test.xlsx"
test_path = r"C:\Users\ychaker\Desktop\SIG\test.xlsx"
output_path = r"C:\Users\ychaker\Desktop\SIG\test_rempli.xlsx"

# === Feuilles à traiter et clés de jointure ===
sheets_config = {
    'Surface_3D_par_OAD_OAF': ['OA_D', 'OA_F'],
    'Surface_3D_par_modele': ['OA_D', 'OA_F', 'Modèle_du', 'Int_Ext', 'Voie_COS'],
    'Surface_3D_par_OA': ['OA_D', 'OA_F', 'Int_Ext', 'Voie_COS']
}

# === Charger fichiers Excel ===
rapport = pd.read_excel(rapport_path, sheet_name=None)
test = pd.read_excel(test_path, sheet_name=None)

# === Nettoyage des noms de colonnes ===
def clean_columns(df):
    df.columns = [c.strip().replace('.', '_').replace(' ', '_') for c in df.columns]
    return df

rapport = {k: clean_columns(v) for k, v in rapport.items()}
test = {k: clean_columns(v) for k, v in test.items()}

# === Traitement feuille par feuille ===
updated = {}

for sheet_name, merge_keys in sheets_config.items():
    df_rapport = rapport[sheet_name].copy()
    df_test = test.get(sheet_name, pd.DataFrame()).copy()

    # Colonnes à remplir
    fields_to_fill = ['SURF_3D_Totale']
    if sheet_name == 'Surface_3D_par_modele':
        fields_to_fill.append('Count')

    # Supprimer les champs à remplir dans test
    df_test = df_test.drop(columns=[f for f in fields_to_fill if f in df_test.columns], errors='ignore')

    # Fusionner avec rapport pour récupérer les bonnes valeurs
    df_filled = df_rapport[merge_keys].copy()
    for field in fields_to_fill:
        if field in df_rapport.columns:
            df_filled[field] = df_rapport[field]
        else:
            df_filled[field] = None

    # Merge test avec les nouvelles valeurs selon les clés
    df_merged = df_rapport[merge_keys].merge(df_test, on=merge_keys, how='left')
    for field in fields_to_fill:
        df_merged[field] = df_rapport[field]  # forcer à prendre la valeur du rapport

    # Replacer dans l'ordre du rapport
    ordered_cols = list(df_rapport.columns)
    df_final = df_merged[[col for col in ordered_cols if col in df_merged.columns]]

    updated[sheet_name] = df_final

# === Sauvegarde du résultat ===
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    for sheet, df in updated.items():
        df.to_excel(writer, sheet_name=sheet, index=False)

print(f"✅ Données remplies avec succès dans : {output_path}")
