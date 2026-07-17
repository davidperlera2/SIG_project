import rasterio
import numpy as np
import joblib
from scipy.ndimage import generic_filter, label

model = joblib.load("modelo_rf_2.pkl")

sentinel2_path = "../Imagenes_Satelitales/Sentinel2_2025_a_2.tif"
sentinel1_path = "../Imagenes_Satelitales/SAR_2025_a.tif"

with rasterio.open(sentinel2_path) as src:
    img = src.read()
    profile = src.profile

with rasterio.open(sentinel1_path) as src:
    sar = src.read()

with rasterio.open(sentinel2_path) as src:
    img = src.read()
    profile = src.profile

with rasterio.open(sentinel1_path) as src:
    sar = src.read()

B2 = img[0]
B3 = img[1]
B4 = img[2]
B8 = img[3]
B11 = img[4]

epsilon = 1e-10

NDVI = (B8 - B4) / (B8 + B4 + epsilon)
NDBI = (B11 - B8) / (B11 + B8 + epsilon)
NDWI = (B3 - B8) / (B3 + B8 + epsilon)

VH = sar[0]
VV = sar[1]
ratio = sar[2]

stack = np.stack([
    B2, B3, B4, B8, B11,
    NDVI, NDBI, NDWI,
    VV, VH, ratio
], axis=0)

n_features, rows, cols = stack.shape

X_full = stack.reshape(n_features, rows * cols).T


print("Prediciendo nueva área...")

y_pred_full = model.predict(X_full)

pred = y_pred_full.reshape(rows, cols)

print("Aplicando filtro espacial...")

# -------------------------
# Majority Filter
# -------------------------

def majority_filter(values):

    values = values.astype(int)

    values = values[values != -1]

    if len(values) == 0:
        return -1

    return np.bincount(values).argmax()

filtered = generic_filter(
    pred,
    function=majority_filter,
    size=3
)

# -------------------------
# Remove small regions
# -------------------------

min_pixels = 20

cleaned = filtered.copy()

classes = np.unique(filtered)

for cls in classes:

    if cls == -1:
        continue

    mask = filtered == cls

    labeled, num_features = label(mask)

    for region_id in range(1, num_features + 1):

        region = labeled == region_id

        if np.sum(region) < min_pixels:
            cleaned[region] = 0


profile.update(
    dtype=rasterio.int16,
    count=1,
    nodata=-1
)

output_path = "../Results/clasificacion_2025_ap_t_3.tif"

with rasterio.open(output_path, "w", **profile) as dst:
    dst.write(cleaned.astype(rasterio.int16), 1)

print("Mapa guardado:", output_path)