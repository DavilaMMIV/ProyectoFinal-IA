import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RUTA_DATOS = Path(__file__).resolve().parent.parent / "datos_senas"
ARCHIVO = RUTA_DATOS / "dataset.npz"

if not ARCHIVO.exists():
    raise FileNotFoundError(
        f"No existe {ARCHIVO}. Ejecuta primero capturar_datos.py."
    )

with np.load(ARCHIVO, allow_pickle=False) as data:
    X = data["X"].astype(np.float32)
    y = data["y"].astype(str)

if X.ndim != 2 or X.shape[1] != 63:
    raise ValueError(f"Se esperaban datos con forma (muestras, 63); recibidos {X.shape}.")
if len(X) != len(y):
    raise ValueError(f"X y y tienen diferente cantidad de muestras: {len(X)} y {len(y)}.")

encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)
print(f"Muestras: {len(X)} | Clases: {list(encoder.classes_)}")

conteos = np.bincount(y_enc)
if len(encoder.classes_) < 2 or np.any(conteos < 2):
    raise ValueError("Necesitas al menos 2 muestras para cada una de 2 o más clases.")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

modelo = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
)
modelo.fit(X_tr, y_tr)

precision = modelo.score(X_te, y_te)
print(f"\nPrecisión en test: {precision:.2%}")

joblib.dump(modelo, RUTA_DATOS / "modelo_lsm.joblib", compress=3)
joblib.dump(encoder, RUTA_DATOS / "encoder.joblib", compress=3)
print(f"Modelo estático guardado en: {RUTA_DATOS}")
