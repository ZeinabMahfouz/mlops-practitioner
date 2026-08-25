import logging
import pickle

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import engineer_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    # 1. Load and prepare data
    df_raw = load_data(str(settings.DATA_PATH))
    df = engineer_features(df_raw)
    df_train, df_val = split_data(df)

    categorical = ["PU_DO"]
    numerical = ["trip_distance"]

    # Define dictionary features
    dicts_train = df_train[categorical + numerical].to_dict(orient="records")
    dicts_val = df_val[categorical + numerical].to_dict(orient="records")

    # 2. Vectorize and train model
    dv = DictVectorizer()
    X_train = dv.fit_transform(dicts_train)
    X_val = dv.transform(dicts_val)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))

    logger.info(f"Validation MAE: {mae:.4f}")
    logger.info(f"Validation RMSE: {rmse:.4f}")

    settings.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 3. Save standard pickle model artifact
    with open(settings.MODEL_PATH, "wb") as f_out:
        pickle.dump((dv, model), f_out)
    logger.info(f"Model saved to {settings.MODEL_PATH}")

    # 4. Export Ridge model to ONNX
    num_features = X_train.shape[1]
    initial_type = [("float_input", FloatTensorType([None, num_features]))]

    try:
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        onnx_path = settings.MODEL_PATH.with_suffix(".onnx")
        with open(onnx_path, "wb") as f_onnx:
            f_onnx.write(onnx_model.SerializeToString())
        logger.info(f"ONNX model saved to {onnx_path}")
    except (RuntimeError, ValueError, TypeError) as e:
        logger.warning(f"Could not export ONNX model: {e}")


if __name__ == "__main__":
    main()
