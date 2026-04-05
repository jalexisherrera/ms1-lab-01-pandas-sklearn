"""Generador de casos de uso para contar_nulos_por_columna (pregunta 0002)."""

import random

import numpy as np
import pandas as pd


def generar_caso_de_uso_contar_nulos_por_columna():
    """Cada llamada produce un par (input_dict, output_esperado) distinto."""
    n_rows = random.randint(5, 20)
    n_cols = random.randint(2, 6)
    cols = [f"f_{i}" for i in range(n_cols)]
    df = pd.DataFrame(np.random.randn(n_rows, n_cols), columns=cols)

    # Introducir NaN aleatorios
    for _ in range(random.randint(1, max(1, n_rows * n_cols // 3))):
        r, c = random.randrange(n_rows), random.randrange(n_cols)
        df.iat[r, c] = np.nan

    df_arg = df.copy()
    input_data = {"df": df_arg}
    output_data = df.isnull().sum()
    return input_data, output_data


if __name__ == "__main__":
    inp, out = generar_caso_de_uso_contar_nulos_por_columna()
    assert out.shape[0] == inp["df"].shape[1]
    print("OK 0002:", out.to_dict())
