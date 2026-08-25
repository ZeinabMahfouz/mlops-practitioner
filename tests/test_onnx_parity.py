import pathlib
import joblib
import numpy as np
import pytest


def test_onnx_parity():
    ort = pytest.importorskip("onnxruntime")

    models_dir = pathlib.Path(__file__).parent.parent / "models"
    pkl_path = models_dir / "model.pkl"
    onnx_path = models_dir / "model.onnx"

    if not onnx_path.exists() or not pkl_path.exists():
        pytest.skip("Model artifacts not found.")

    # 1. Load vectorizer & models
    dv, pkl_model = joblib.load(pkl_path)
    ort_session = ort.InferenceSession(str(onnx_path))

    # 2. Transform sample data
    sample_dict = [{"PU_DO": "130_205", "trip_distance": 2.5}]
    X_sample = dv.transform(sample_dict).astype(np.float32).toarray()

    # 3. Model predictions
    expected_pred = pkl_model.predict(X_sample)

    input_name = ort_session.get_inputs()[0].name
    actual_pred = ort_session.run(None, {input_name: X_sample})[0]

    # 4. Parity assertion
    np.testing.assert_allclose(
        actual_pred.flatten(), expected_pred.flatten(), rtol=1e-4, atol=1e-5
    )
