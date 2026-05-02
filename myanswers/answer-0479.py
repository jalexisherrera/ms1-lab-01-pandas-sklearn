from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def entrenar_clasificador_robusto(X, y):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=50)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline
