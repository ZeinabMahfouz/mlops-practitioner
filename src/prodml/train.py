import pickle

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import engineer_features


def main() -> None:
    df_raw = load_data(str(settings.DATA_PATH))
    df = engineer_features(df_raw)
    df_train, df_val = split_data(df)

    categorical = ["PU_DO"]
    numerical = ["trip_distance"]

    dicts_train = df_train[categorical + numerical].to_dict(orient="records")
    dicts_val = df_val[categorical + numerical].to_dict(orient="records")

    dv = DictVectorizer()
    X_train = dv.fit_transform(dicts_train)
    X_val = dv.transform(dicts_val)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")

    settings.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.MODEL_PATH, "wb") as f_out:
        pickle.dump((dv, model), f_out)
    print(f"Model saved to {settings.MODEL_PATH}")


if __name__ == "__main__":
    main()
