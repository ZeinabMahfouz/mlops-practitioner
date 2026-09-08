import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(model, dv, test_df):
    """حساب مقاييس الأداء للنموذج على بيانات الاختبار"""
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]

    dicts_test = test_df[categorical + numerical].to_dict(orient="records")
    X_test = dv.transform(dicts_test)
    y_test = test_df["duration"].values

    test_dmatrix = xgb.DMatrix(X_test)
    y_pred = model.predict(test_dmatrix)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    return {"test_mae": float(mae), "test_rmse": rmse}


def main():
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/model.pkl")
    input_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/features")
    reports_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. تحميل النموذج والـ DictVectorizer المحفوظين معاً
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    dv = artifact["dv"]

    # 2. تحميل بيانات الاختبار
    test_df = pd.read_parquet(input_dir / "test.parquet")

    # 3. حساب المقاييس باستخدام الدالة المستقلة
    metrics = evaluate_model(model, dv, test_df)

    # 4. حفظ المقاييس بصيغة JSON لكي يقرأها DVC
    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(
        f"Evaluation metrics saved to {metrics_path} | MAE: {metrics['test_mae']:.4f}, RMSE: {metrics['test_rmse']:.4f}"
    )


if __name__ == "__main__":
    main()
