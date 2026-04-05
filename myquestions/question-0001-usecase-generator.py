"""Generador de casos de uso para seleccionar_columnas (pregunta 0001)."""

import random

import numpy as np
import pandas as pd


def generar_caso_de_uso_seleccionar_columnas():
    """Cada llamada produce un par (input_dict, output_esperado) distinto."""
    n_rows = random.randint(4, 15)
    n_cols = random.randint(3, 7)
    all_cols = [f"col_{i}" for i in range(n_cols)]
    df = pd.DataFrame(np.random.randn(n_rows, n_cols), columns=all_cols)

    k = random.randint(1, n_cols)
    columnas = random.sample(all_cols, k)

    df_arg = df.copy()
    input_data = {"df": df_arg, "columnas": columnas}
    output_data = df[list(columnas)].copy()
    return input_data, output_data


if __name__ == "__main__":
    inp, out = generar_caso_de_uso_seleccionar_columnas()
    assert inp["df"].shape[0] == out.shape[0]
    assert list(out.columns) == inp["columnas"]
    print("OK 0001:", out.shape)
