from prodml.predict import DurationPredictor


def test_duration_predictor():
    predictor = DurationPredictor(model_path="models/model.pkl")
    predictor.load()

    sample_features = {"PU_DO": "130_205", "trip_distance": 3.5}

    pred = predictor.predict_one(sample_features)
    assert isinstance(pred, float)
    assert pred > 0
