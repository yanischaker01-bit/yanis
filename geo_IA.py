import csv
import sys

# Augmenter la limite de taille des champs CSV
csv.field_size_limit(sys.maxsize)

with open(r"D:\test_Leroy\NDVI_11_04_25.csv", newline='', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile, delimiter=";")
    nb_lignes = sum(1 for row in reader)

print(f"Nombre de lignes totales : {nb_lignes}")
