import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


def rankear_importancia_features(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_imputed, y)

    result = (
        pd.DataFrame({"feature": X.columns, "importancia": model.feature_importances_})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    return result
