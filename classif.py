import numpy as np
from osgeo import gdal, ogr, osr
import matplotlib.pyplot as plt
from skimage import exposure
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.ndimage import median_filter
import os

# === PARAMÈTRES ===
img_path = r"C:/Users/ychaker/Downloads/478000_6562000.tif"
roi_path = r"C:/Users/ychaker/Desktop/SIG/ROI_classif.shp"
out_raster_path = r"C:/Users/ychaker/Desktop/classification_finale.tif"
tile_size = 512

# === 1. Charger l’image et calculer NDVI ===
def load_image(img_path):
    ds = gdal.Open(img_path)
    if ds is None:
        raise FileNotFoundError(f"Image introuvable : {img_path}")
    geo = ds.GetGeoTransform()
    proj = ds.GetProjection()
    bands = ds.RasterCount
    img = np.stack([exposure.rescale_intensity(ds.GetRasterBand(i+1).ReadAsArray(), out_range=(0, 1)) for i in range(bands)], axis=2)
    
    if bands >= 4:
        red, nir = img[:, :, 0], img[:, :, 3]
        ndvi = (nir - red) / (nir + red + 1e-10)
        img = np.concatenate([img, ndvi[:, :, np.newaxis]], axis=2)
    return img, geo, proj

img, geotransform, projection = load_image(img_path)

# === 2. Extraire les échantillons d’apprentissage depuis le shapefile ===
def extract_training_samples(roi_path, img, geotransform):
    drv = ogr.GetDriverByName("ESRI Shapefile")
    ds = drv.Open(roi_path, 0)
    lyr = ds.GetLayer()
    
    X, y = [], []
    for feat in lyr:
        classe = feat.GetField("Classe")
        if not classe:
            continue
        mem_drv = ogr.GetDriverByName("Memory")
        mem_ds = mem_drv.CreateDataSource("")
        mem_lyr = mem_ds.CreateLayer("poly", srs=lyr.GetSpatialRef(), geom_type=ogr.wkbPolygon)
        mem_lyr.CreateFeature(feat.Clone())
        
        mask_ds = gdal.GetDriverByName("MEM").Create("", img.shape[1], img.shape[0], 1, gdal.GDT_Byte)
        mask_ds.SetGeoTransform(geotransform)
        mask_ds.SetProjection(projection)
        gdal.RasterizeLayer(mask_ds, [1], mem_lyr, burn_values=[1])
        mask = mask_ds.GetRasterBand(1).ReadAsArray()
        
        idx = np.where(mask == 1)
        if idx[0].size > 0:
            X.append(img[idx])
            y.extend([classe] * idx[0].size)
    
    if not X:
        raise ValueError("Aucun pixel d’apprentissage trouvé.")
    
    return np.vstack(X), np.array(y)

X, y = extract_training_samples(roi_path, img, geotransform)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"✅ Apprentissage sur {len(y)} pixels répartis en {len(le.classes_)} classes")

# === 3. Entraîner le modèle IA ===
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

clf = HistGradientBoostingClassifier(
    max_iter=200,
    max_depth=12,
    learning_rate=0.1,
    l2_regularization=0.2,
    early_stopping=True,
    random_state=42
)

clf.fit(X_train, y_train)
print("\n🎯 Évaluation sur jeu de test :")
print(classification_report(y_test, clf.predict(X_test), target_names=le.classes_))

# === 4. Prédiction par blocs ===
def predict_image_blocks(img, model, tile_size=512):
    h, w, bands = img.shape
    prediction = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            tile = img[i:i+tile_size, j:j+tile_size, :]
            flat = tile.reshape(-1, bands)
            pred = model.predict(flat)
            prediction[i:i+tile.shape[0], j:j+tile.shape[1]] = pred.reshape(tile.shape[0], tile.shape[1])
    return prediction

print("🔍 Prédiction globale...")
pred = predict_image_blocks(img, clf)
pred_filtered = median_filter(pred, size=3)

# === 5. Sauvegarder le raster classifié ===
def save_raster(path, array, geotransform, projection):
    driver = gdal.GetDriverByName("GTiff")
    out_ras = driver.Create(path, array.shape[1], array.shape[0], 1, gdal.GDT_Byte)
    out_ras.SetGeoTransform(geotransform)
    out_ras.SetProjection(projection)
    out_ras.GetRasterBand(1).WriteArray(array)
    out_ras.FlushCache()
    out_ras = None

save_raster(out_raster_path, pred_filtered, geotransform, projection)
print(f"💾 Raster classifié sauvegardé : {out_raster_path}")

# === 6. Visualisation rapide ===
plt.figure(figsize=(15, 5))
plt.subplot(131); plt.title("Image RGB"); plt.imshow(img[:, :, :3])
plt.subplot(132); plt.title("Classification brute"); plt.imshow(pred, cmap='tab20')
plt.subplot(133); plt.title("Classification filtrée"); plt.imshow(pred_filtered, cmap='tab20')
plt.tight_layout()
plt.show()
