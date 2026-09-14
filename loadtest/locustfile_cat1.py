"""Step 3 load test: CAT 1 (FastAPI eager) baseline.

    locust -f loadtest/locustfile_cat1.py --host http://localhost:8000

Open http://localhost:8089, set 50 users / spawn rate ~5, run for a
minute or two once p95 stabilizes, then record p50/p95/p99 and RPS.
"""

import random

from locust import HttpUser, between, task

LOCATION_IDS = list(range(1, 265))


class RideDurationUser(HttpUser):
    wait_time = between(0, 0.1)

    @task
    def predict(self):
        payload = {
            "PULocationID": random.choice(LOCATION_IDS),
            "DOLocationID": random.choice(LOCATION_IDS),
            "trip_distance": round(random.uniform(0.5, 20.0), 2),
        }
        self.client.post("/predict", json=payload)
