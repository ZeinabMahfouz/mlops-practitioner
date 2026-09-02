import sys
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

# Add the src directory to sys.path to ensure prodml package can be imported
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(autouse=True)
def setup_dummy_model():
    model_path = Path("models/model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove corrupted or existing model file to prevent unpickling errors
    if model_path.exists():
        model_path.unlink()

    # Create fitted model and vectorizer
    model = LinearRegression()
    dv = DictVectorizer()
    X = dv.fit_transform(
        [
            {
                "PULocationID": 130,
                "DOLocationID": 205,
                "trip_distance": 3.5,
                "PU_DO": "130_205",
            }
        ]
    )
    y = [10.0]
    model.fit(X, y)

    # Dump the tuple safely using joblib
    joblib.dump((model, dv), model_path)
