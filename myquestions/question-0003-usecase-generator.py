"""Generador de casos de uso para escalar_robusto (pregunta 0003)."""

import random

import numpy as np
from sklearn.preprocessing import RobustScaler


def generar_caso_de_uso_escalar_robusto():
    """Cada llamada produce un par (input_dict, output_esperado) distinto."""
    n_rows = random.randint(12, 45)
    n_features = random.randint(2, 8)
    X = np.random.randn(n_rows, n_features) * random.uniform(0.5, 2.5)
    X = X + random.uniform(-1.5, 1.5)

    # Añadir algunos outliers para que RobustScaler sea no trivial
    for _ in range(random.randint(1, n_features)):
        r, c = random.randrange(n_rows), random.randrange(n_features)
        X[r, c] += random.choice([-1, 1]) * random.uniform(8, 20)

    input_data = {"X": X.copy()}
    scaler = RobustScaler()
    output_data = scaler.fit_transform(X)
    return input_data, output_data


if __name__ == "__main__":
    inp, out = generar_caso_de_uso_escalar_robusto()
    assert out.shape == inp["X"].shape
    assert np.isfinite(out).all()
    print("OK 0003 RobustScaler:", out.shape)
