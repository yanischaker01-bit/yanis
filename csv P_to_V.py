import csv

# Chemins
input_path = r"E:\test_Leroy\NDVI_19__09.csv"
output_path = r"E:\test_Leroy\NDVI_19__09_.csv"

# Lecture + traitement
with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, delimiter=';')  # optionnel : change le séparateur si besoin

    for row in reader:
        new_row = []
        for value in row:
            try:
                # Si c'est un nombre, remplacer le . par ,
                float(value)  # vérifie si c’est bien un nombre
                value = value.replace('.', ',')
            except ValueError:
                pass  # si ce n'est pas un nombre, on ne touche pas
            new_row.append(value)
        writer.writerow(new_row)

print("✅ Fichier converti avec succès :", output_path)
