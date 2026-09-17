import random

from locust import HttpUser, between, task

LOCATION_IDS = list(range(1, 265))


def _random_trip():
    return {
        "PULocationID": random.choice(LOCATION_IDS),
        "DOLocationID": random.choice(LOCATION_IDS),
        "trip_distance": round(random.uniform(0.5, 20.0), 2),
    }


class RideDurationUser(HttpUser):
    wait_time = between(1, 3)

    @task(80)
    def predict_single(self):
        self.client.post("/predict", json=_random_trip(), name="/predict")

    @task(15)
    def predict_batch(self):
        n = random.randint(5, 50)
        trips = [_random_trip() for _ in range(n)]
        self.client.post("/predict_batch", json={"trips": trips}, name="/predict_batch")

    @task(5)
    def metadata(self):
        self.client.post("/metadata", json={}, name="/metadata")
