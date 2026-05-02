from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


def procesar_calidad_cafe(df_muestras, humedad_min):
    df_step = df_muestras[df_muestras["humedad"] >= humedad_min].copy()

    imputer = SimpleImputer(strategy="median")
    df_step["altitud"] = imputer.fit_transform(df_step[["altitud"]])

    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(df_step[["altitud", "densidad"]])

    y_target = df_step["puntuacion"]
    return x_scaled, y_target
