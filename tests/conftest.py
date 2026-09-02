import sys
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

# add the src directory to the sys.path to ensure that the prodml package can be imported
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(autouse=True)
def setup_dummy_model():
    # Automatically create a valid fitted dummy model file at the default path before tests run
    model_path = Path("models/model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)

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

    joblib.dump((model, dv), model_path)
