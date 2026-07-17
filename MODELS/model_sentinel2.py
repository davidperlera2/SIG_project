import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.ndimage import generic_filter, label
import joblib

# =========================
# 1. LOAD DATA
# =========================

sentinel2_path = "../Imagenes_Satelitales/Sentinel2_2025.tif"
gt_path = "../gt_2.tif"

with rasterio.open(sentinel2_path) as src:
    img = src.read()
    profile = src.profile

with rasterio.open(gt_path) as src:
    gt = src.read(1)

print("Shape Sentinel:", img.shape)
print("Shape GroundTruth:", gt.shape)

# =========================
# 2. PREPARE FEATURES
# =========================

B2 = img[0]
B3 = img[1]
B4 = img[2]
B8 = img[3]
B11 = img[4]

epsilon = 1e-10

NDVI = (B8 - B4) / (B8 + B4 + epsilon)
NDBI = (B11 - B8) / (B11 + B8 + epsilon)
NDWI = (B3 - B8) / (B3 + B8 + epsilon)

stack = np.stack([
    NDVI, NDBI, NDWI
], axis=0)

n_features, rows, cols = stack.shape

# =========================
# 3. SPATIAL BLOCK SPLIT
# =========================


block_size = 256

train_mask = np.zeros((rows, cols), dtype=bool)
test_mask = np.zeros((rows, cols), dtype=bool)


for i in range(0, rows, block_size):
    for j in range(0, cols, block_size):

        block_row = i // block_size
        block_col = j // block_size

        if (block_row + block_col) % 2 == 0:
            train_mask[i:i+block_size, j:j+block_size] = True
        else:
            test_mask[i:i+block_size, j:j+block_size] = True

# =============================
# 4. PREPARE TRAIN Y TEST DATA
# =============================

X_full = stack.reshape(n_features, rows * cols).T
y_full = gt.reshape(rows * cols)

train_mask_flat = train_mask.reshape(rows * cols)
test_mask_flat = test_mask.reshape(rows * cols)

train_valid = (y_full != -1) & train_mask_flat

X_train = X_full[train_valid]
y_train = y_full[train_valid]

test_valid = (y_full != -1) & test_mask_flat

X_test = X_full[test_valid]
y_test = y_full[test_valid]

print("Train:", X_train.shape)
print("Test:", X_test.shape)

print("\n===== PIXELS DISTRIBUTION =====")

print(f"Total training pixels: {len(y_train)}")
print(f"Total evaluation pixels: {len(y_test)}")

print("\nTraining pixels per class:")
unique_train, counts_train = np.unique(y_train, return_counts=True)

for clase, cantidad in zip(unique_train, counts_train):
    print(f"Class {clase}: {cantidad} pixels")

print("\nEvaluation pixels per class:")
unique_test, counts_test = np.unique(y_test, return_counts=True)

for clase, cantidad in zip(unique_test, counts_test):
    print(f"Class {clase}: {cantidad} pixels")

# =========================
# 5. TRAIN MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# =========================================
# 6. EVALUATE MODEL
# =========================================

y_pred_test = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred_test))

print("\nClassification report:\n")
print(classification_report(y_test, y_pred_test))

print("\nPredicting full map...")

print("\nKappa:", cohen_kappa_score(y_test, y_pred_test))

print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_pred_test)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')

plt.title("Confusion Matrix")
plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)
print("Confusion matrix saved as 'confusion_matrix.png'")

plt.show()

y_pred_full = model.predict(X_full)

pred = y_pred_full.reshape(rows, cols)


# =========================
# 7. SAVE CLASSIFICATION
# =========================

profile.update(
    dtype=rasterio.int16,
    count=1,
    nodata=-1
)

output_path = "final_classification_sentinel.tif"

with rasterio.open(output_path, "w", **profile) as dst:
    dst.write(pred.astype(rasterio.int16), 1)

print("\nMap saved in:", output_path)

# =========================
# 8. SAVE SPLIT MAP
# =========================

split_map = np.full((rows, cols), -1)

split_map[train_mask] = 1
split_map[test_mask] = 2

split_map[gt == -1] = -1

split_path = "split_blocks.tif"

with rasterio.open(split_path, "w", **profile) as dst:
    dst.write(split_map.astype(rasterio.int16), 1)

print("Split Map saved in:", split_path)

# =========================
# 9. IMPORTANCE OF FEATURES
# =========================

features = [
    "NDVI","NDBI","NDWI",
    "VV","VH","VV/VH"
]

importances = model.feature_importances_

for f, imp in zip(features, importances):
    print(f, imp)

# =========================
# 10. SAVE MODEL
# =========================
joblib.dump(model, "model_sentinel2.pkl")