import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def seleccionar_features_por_importancia(X, y, modelo_tipo="random_forest", top_k=5):
    if modelo_tipo != "random_forest":
        raise ValueError("Solo se admite modelo_tipo='random_forest'")

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    if np.issubdtype(y_arr.dtype, np.integer) or np.issubdtype(y_arr.dtype, np.bool_):
        modelo = RandomForestClassifier(random_state=42)
    else:
        modelo = RandomForestRegressor(random_state=42)

    modelo.fit(X_arr, y_arr)
    importancias = modelo.feature_importances_

    indices_seleccionados = np.argsort(importancias)[::-1][:top_k]
    X_reducido = X_arr[:, indices_seleccionados]

    return {
        "indices_seleccionados": indices_seleccionados,
        "X_reducido": X_reducido,
    }
