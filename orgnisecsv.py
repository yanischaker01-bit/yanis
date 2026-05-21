import csv

input_file = r"C:\Users\ychaker\Downloads\PAM_NO.csv"
output_file = r"C:\Users\ychaker\Downloads\PAM_NO_clean.csv"

headers = [
    "LIBELLE", "AXE", "PK_EF", "PK_GC", "DECALAGE",
    "GPS_DMS_LAT", "GPS_DMS_LONG",
    "GPS_DD_LAT", "GPS_DD_LONG", "COMMENTAIRE"
]

records = []
current_block = []
line_counter = 0

with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        text = line.strip()
        line_counter += 1
        
        # Ignorer les lignes vides
        if not text:
            continue
        
       
        # Ajouter la ligne au bloc courant
        current_block.append(text)
        
        # Dès que nous avons 10 éléments, on traite le bloc
        if len(current_block) == 10:
            libelle = current_block[0]
            commentaire = current_block[9]
            
            # Vérifier si le LIBELLE commence par PAM mais n'est pas PAM Bassin
            if libelle.startswith("PAM") and not libelle.startswith("PAM Bassin"):
                # Traitement du COMMENTAIRE
                if "PAM Bassin" in commentaire:
                    current_block[9] = "PAM Bassin"
                else:
                    current_block[9] = ""  # Laisser vide si pas PAM Bassin
                
                # Convertir les virgules en points pour les coordonnées GPS décimales
                # GPS_DD_LAT (index 7) et GPS_DD_LONG (index 8)
                for i in [7, 8]:
                    if i < len(current_block):
                        if ',' in current_block[i]:
                            current_block[i] = current_block[i].replace(',', '.')
                
                # Ajouter le bloc finalisé
                records.append(current_block[:])
            # Si le libelle n'est pas valide, on ignore simplement l'enregistrement
            
            # Réinitialiser le bloc pour la prochaine ligne
            current_block = []

print(f"\n{'='*60}")
print(f"ANALYSE TERMINÉE")
print(f"{'='*60}")
print(f"Lignes traitées: {line_counter}")
print(f"Enregistrements exportés: {len(records)}")

# Écriture du fichier nettoyé
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(headers)
    writer.writerows(records)

print(f"\n✅ Fichier créé : {output_file}")

# Afficher un aperçu des données exportées
if records:
    print(f"\n{'='*60}")
    print("APERÇU DES DONNÉES EXPORTÉES (3 premiers)")
    print(f"{'='*60}")
    
    for i, record in enumerate(records[:3], 1):
        print(f"\nEnregistrement {i}:")
        for header, value in zip(headers, record):
            print(f"  {header}: {value}")
    
    # Statistiques sur les commentaires
    print(f"\n{'='*60}")
    print("STATISTIQUES SUR LES COMMENTAIRES")
    print(f"{'='*60}")
    
    commentaires_pam_bassin = sum(1 for r in records if r[9] == "PAM Bassin")
    commentaires_vides = sum(1 for r in records if r[9] == "")
    
    print(f"Commentaires 'PAM Bassin': {commentaires_pam_bassin}")
    print(f"Commentaires vides: {commentaires_vides}")
    print(f"Total enregistrements: {len(records)}")
    
    if len(records) > 3:
        print(f"\n... et {len(records) - 3} autres enregistrements")
else:
    print("\n⚠️ Aucun enregistrement exporté !")
    print("   Vérifiez que votre fichier contient des enregistrements avec:")
    print("   - LIBELLE commençant par 'PAM'")
    print("   - LIBELLE ne commençant pas par 'PAM Bassin'")