import subprocess
import os

dwg_path = r"C:\Users\ychaker\Downloads\LEVE BUSES AU PORTAIL R1FS 2+500.dwg"
gpkg_path = r"C:\Users\ychaker\Downloads\LEVE_BUSES_AU_PORTAIL_R1FS_2+500.gpkg"

if not os.path.exists(dwg_path):
    print(f"❌ Fichier DWG introuvable : {dwg_path}")
else:
    print(f"✅ Fichier DWG trouvé : {dwg_path}")

    command = [
        "ogr2ogr",
        "-f", "GPKG",
        gpkg_path,
        dwg_path,
        "-overwrite",
    ]

    try:
        print("🛠️ Conversion en cours...")
        subprocess.run(command, check=True)
        print(f"✅ Conversion terminée : {gpkg_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de la conversion : {e}")
    except FileNotFoundError:
        print("❌ 'ogr2ogr' non trouvé. Assure-toi que GDAL est bien installé et que son dossier 'bin' est dans le PATH.")
