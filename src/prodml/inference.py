import mlflow
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer

from prodml.config import settings
from prodml.data import load_data
from prodml.features import engineer_features


def predict_duration():
    # 1. Set MLflow tracking URI
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URL)

    # 2. Load the model from Model Registry (Staging stage)
    model_name = "ride-duration-predictor"
    model_uri = f"models:/{model_name}/Staging"

    print(f"Loading model from: {model_uri}")
    model = mlflow.xgboost.load_model(model_uri)

    # 3. Load and prepare sample data for inference
    df_raw = load_data(str(settings.DATA_PATH))
    df = engineer_features(df_raw)

    categorical = ["PU_DO"]
    numerical = ["trip_distance"]

    dicts = df[categorical + numerical].head(5).to_dict(orient="records")

    dv = DictVectorizer(sparse=True)
    X_inference = dv.fit_transform(dicts)

    # 4. Create DMatrix and run predictions
    dmatrix = xgb.DMatrix(X_inference)
    predictions = model.predict(dmatrix)

    for i, pred in enumerate(predictions):
        print(f"Trip {i+1} - Predicted Duration: {pred:.2f} minutes")


if __name__ == "__main__":
    predict_duration()
