"""Generador de casos de uso para imputar_mediana_y_devolver (pregunta 0004)."""

import random

import numpy as np
from sklearn.impute import SimpleImputer


def generar_caso_de_uso_imputar_mediana_y_devolver():
    """Cada llamada produce un par (input_dict, output_esperado) distinto."""
    n_rows = random.randint(10, 35)
    n_features = random.randint(2, 7)
    X = np.random.randn(n_rows, n_features).astype(np.float64)

    n_nan = random.randint(1, max(1, n_rows * n_features // 4))
    for _ in range(n_nan):
        r, c = random.randrange(n_rows), random.randrange(n_features)
        X[r, c] = np.nan

    # Al menos un valor finito por columna (mediana definida para SimpleImputer)
    for c in range(n_features):
        if np.isnan(X[:, c]).all():
            X[random.randrange(n_rows), c] = random.uniform(-1, 1)

    input_data = {"X": X.copy()}
    imputer = SimpleImputer(strategy="median")
    output_data = imputer.fit_transform(X)
    return input_data, output_data


if __name__ == "__main__":
    inp, out = generar_caso_de_uso_imputar_mediana_y_devolver()
    assert out.shape == inp["X"].shape
    assert not np.isnan(out).any()
    print("OK 0004 SimpleImputer median:", out.shape)
