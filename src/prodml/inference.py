import mlflow
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer

from prodml.config import settings
from prodml.data import load_data
from prodml.features import engineer_features


def load_inference_data(data_path: str, num_samples: int = 5):
    """تحميل وتجهيز البيانات لعملية التنبؤ"""
    df_raw = load_data(data_path)
    df = engineer_features(df_raw)
    return df.head(num_samples)


def predict_duration(
    model_uri: str = "models:/ride-duration-predictor/Staging",
    data_path: str = str(settings.DATA_PATH),
):
    # 1. Set MLflow tracking URI
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URL)

    # 2. Load the model from Model Registry
    print(f"Loading model from: {model_uri}")
    model = mlflow.xgboost.load_model(model_uri)

    # 3. Load and prepare sample data for inference
    df = load_inference_data(data_path)

    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    dicts = df[categorical + numerical].to_dict(orient="records")

    # تنبيه: في بيئة الإنتاج الحقيقية يجب تحميل الـ DictVectorizer المحفوظ
    # بدلاً من عمل fit من جديد لضمان توافق الميزات مع النموذج.
    dv = DictVectorizer(sparse=True)
    X_inference = dv.fit_transform(dicts)

    # 4. Create DMatrix and run predictions
    dmatrix = xgb.DMatrix(X_inference)
    predictions = model.predict(dmatrix)

    for i, pred in enumerate(predictions):
        print(f"Trip {i+1} - Predicted Duration: {pred:.2f} minutes")

    return predictions


if __name__ == "__main__":
    predict_duration()
