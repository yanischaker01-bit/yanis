import pandas as pd

# Chemin et noms des feuilles
fichier = r"C:\Users\ychaker\Desktop\SIG\2025 - PROD VEG - 148+990 au  300+820_MAJ.xlsx"
feuille_1 = '149 -225'
feuille_2 = '225 -300 '

# Colonnes à comparer
colonnes_cibles = ['OA_D', 'OA_F', 'Modèle_du', 'Int_Ext', 'Voie_COS']

# Fonction pour charger et nettoyer une feuille
def charger_et_nettoyer(feuille):
    df = pd.read_excel(fichier, sheet_name=feuille, dtype=str)
    df.columns = [col.strip() for col in df.columns]  # Nettoyage noms de colonnes
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)  # Nettoyage contenu
    return df

# Chargement et nettoyage
df1 = charger_et_nettoyer(feuille_1)
df2 = charger_et_nettoyer(feuille_2)

# Filtrage des colonnes de comparaison
df1_filtré = df1[colonnes_cibles].dropna()
df2_filtré = df2[colonnes_cibles].dropna()

# Fusion pour trouver les doublons stricts entre les deux feuilles
doublons = pd.merge(df1_filtré, df2_filtré, how='inner', on=colonnes_cibles).drop_duplicates()

# Résultat
if not doublons.empty:
    print("✅ Doublons trouvés entre les deux feuilles (mêmes valeurs sur OA_D, OA_F, Modèle_du, Int_Ext, Voie_COS) :")
    print(doublons)
else:
    print("❌ Aucun doublon trouvé sur les colonnes spécifiées.")
