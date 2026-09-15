import pickle

import bentoml

with open("models/model.pkl", "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]  # xgb.Booster
dv = artifact["dv"]
bento_model = bentoml.xgboost.save_model(
    "ride_duration_xgb",
    model,
    signatures={"predict": {"batchable": True, "batch_dim": 0}},
    custom_objects={"dv": dv},
)
print(f"Saved: {bento_model.tag}")
