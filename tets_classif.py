import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.mask import mask
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.ndimage import median_filter
import joblib
import warnings

warnings.filterwarnings("ignore")

# === PARAMÈTRES UTILISATEUR ===
IMG_PATH        = r"C:/Users/ychaker/Downloads/478000_6562000.tif"
ROI_PATH        = r"C:/Users/ychaker/Desktop/SIG/ROI_classif.shp"
OUTPUT_RASTER   = r"C:/Users/ychaker/Desktop/vegetation_rf.tif"
TILE_SIZE       = 512
RANDOM_STATE    = 42
N_JOBS          = -1  # tous les coeurs CPU

# === 1. CHARGER L’IMAGE ET CALCULER NDVI (si possible) ===
def load_image(path):
    with rasterio.open(path) as src:
        meta   = src.meta.copy()
        data   = src.read().astype(np.float32) / 10000.  # échelle éventuelle
        bands, h, w = data.shape

        # Calcul de l’NDVI si on a >= 4 bandes (ex: Sentinel-2)
        if bands >= 4:
            red = data[2]   # B3
            nir = data[3]   # B4
            ndvi = (nir - red) / (nir + red + 1e-10)
            # on concatène en tant que bande supplémentaire
            data = np.vstack([data, ndvi[np.newaxis, :, :]])
            bands += 1

        return data, meta

img, meta = load_image(IMG_PATH)

# === 2. EXTRAIRE LES ÉCHANTILLONS D’APPRENTISSAGE ===
def extract_samples(img, meta, shapefile):
    """Renvoie X (pixels×bands) et y (labels) extraits des polygones du shapefile."""
    gdf = gpd.read_file(shapefile).to_crs(meta['crs'])
    X_list, y_list = [], []

    for _, row in gdf.iterrows():
        geom = [row.geometry.__geo_interface__]
        label = row['Classe']
        if label is None:
            continue

        # masque la zone
        masked, _ = mask(rasterio.io.MemoryFile().open(**meta, data=img), geom, crop=False)
        mask_bool = masked[0] != 0  # on prend la première bande pour le masque

        # extraire les pixels valides
        for b in range(masked.shape[0]):
            band_vals = masked[b][mask_bool]
            if b == 0:
                pixel_stack = band_vals[:, None]
            else:
                pixel_stack = np.hstack([pixel_stack, band_vals[:, None]])
        X_list.append(pixel_stack)
        y_list.extend([label] * pixel_stack.shape[0])

    if not X_list:
        raise RuntimeError("Aucun échantillon d’apprentissage trouvé.")
    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y

X, y = extract_samples(img, meta, ROI_PATH)
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"✅ {len(y)} pixels d’entraînement, classes : {list(le.classes_)}")

# === 3. SÉPARATION TRAIN/TEST ET OPTIMISATION RF ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_STATE
)

# grille aléatoire d’hyperparamètres
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth':    [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'class_weight': ['balanced', None]
}

rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)
search = RandomizedSearchCV(
    rf, param_distributions=param_dist,
    n_iter=20, cv=5, scoring='f1_weighted',
    random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=1
)
print("🔍 Recherche des meilleurs hyperparamètres...")
search.fit(X_train, y_train)
best_rf = search.best_estimator_
print("✨ Meilleurs paramètres :", search.best_params_)

# évaluation
y_pred = best_rf.predict(X_test)
print("\n🎯 Rapport de classification sur le jeu de test :")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# on peut sauvegarder le modèle
joblib.dump({'model': best_rf, 'encoder': le}, "rf_vegetation_model.joblib")

# === 4. PRÉDICTION SUR L’IMAGE COMPLÈTE PAR TUILES ===
def predict_tiles(img, meta, model):
    bands, h, w = img.shape
    out = np.zeros((h, w), dtype=np.uint8)

    for i in range(0, h, TILE_SIZE):
        for j in range(0, w, TILE_SIZE):
            win_h = min(TILE_SIZE, h - i)
            win_w = min(TILE_SIZE, w - j)
            tile = img[:, i:i+win_h, j:j+win_w].reshape(bands, -1).T
            preds = model.predict(tile).reshape(win_h, win_w)
            out[i:i+win_h, j:j+win_w] = preds

    return out

print("🔍 Prédiction sur l’image complète...")
pred_map = predict_tiles(img, meta, best_rf)
pred_map = median_filter(pred_map, size=3)

# === 5. SAUVEGARDE DU RASTER CLASSIFIÉ ===
meta.update({
    'count': 1,
    'dtype': 'uint8',
    'compress': 'lzw',
    'tiled': True
})
with rasterio.open(OUTPUT_RASTER, 'w', **meta) as dst:
    dst.write(pred_map, 1)

print(f"💾 Résultat sauvegardé dans {OUTPUT_RASTER}")

# === 6. OPTIONNEL : AFFICHAGE RAPIDE ===
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.title("Carte de végétation (classes encodées)")
plt.imshow(pred_map, cmap='Greens')
plt.axis('off')
plt.show()
