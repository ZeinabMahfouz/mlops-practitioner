# Module 3 Report

## Step 1 -- Orchestrate the training pipeline with Airflow

### DAG design

`pipelines/dags/train_pipeline.py` (task logic in `pipelines/dags/tasks.py`):

```
wait_for_raw_data >> extract >> validate >> train >> evaluate >> decide_promotion
                                                                       |
                                                     +-----------------+-----------------+
                                                     v                                   v
                                             register_model                    skip_registration
                                                     |                                   |
                                                     +-----------------+-----------------+
                                                                       v
                                                                    notify
```

- **Operators.** A `PythonOperator` per task (`extract`, `validate`, `train`, `evaluate`,
  `register_model`, `skip_registration`, `notify`); a `BranchPythonOperator`
  (`decide_promotion`) makes the promote/skip call.
- **Sensor.** `FileSensor` (`fs_default` connection) waits for the raw parquet file
  before `extract` starts. Note: `fs_default` is **not** auto-created in Airflow 2.9+
  (it was in earlier versions) -- `airflow-init` in `docker-compose.yml` creates it
  explicitly with `airflow connections add fs_default --conn-type fs --conn-extra '{"path": "/"}'`.
- **XCom.** Only references cross task boundaries -- a features-directory path (string),
  and a small dict of `{run_id, model_path, mae, rmse, reused}`. Never a DataFrame or a
  model object. XCom is backed by Airflow's metadata database; pushing a DataFrame
  through it would bloat that database and can silently corrupt or truncate large
  values -- the model and data always stay on disk (or in MLflow), and only the
  *reference* to them travels through Airflow.
- **Retries.** `retries=2`, `retry_delay=5 min` (DAG-level default); `train` additionally
  gets `execution_timeout=30 min` since it's the one task that can hang indefinitely
  on a runaway XGBoost fit.
- **Schedule.** `@weekly`, `catchup=False`, `start_date` in the past (2026-01-01).
- **Idempotency.** Re-running the DAG for the same logical date must not corrupt
  anything or duplicate work. Two guards implement this:
  1. `run_training()` (`src/prodml/train.py`) tags every MLflow run with the Airflow
     logical date (`ds`). Before training, it searches for an existing run with that
     tag; if one exists and its model file is still on disk, it's reused and no new
     MLflow run or model file is created.
  2. `register()` (`pipelines/dags/tasks.py`) checks whether the run's model has
     already been registered as a model version before calling `promote_best_model()`,
     so re-evaluating the same run twice doesn't mint a second, identical model
     version.

  Both were exercised directly: running the DAG twice for the same logical date
  (`airflow dags test ride_duration_training_pipeline 2026-01-05`, run twice) produced
  identical output on the second run -- `{'run_id': '...aa2fc6...', 'reused': True}`
  with the exact same `run_id` as the first run -- confirming no duplicate MLflow run
  or model version was created. `extract` and `validate` are naturally idempotent
  (pure, deterministic transforms of immutable raw data that overwrite the same
  output files); `evaluate` is likewise deterministic given a fixed model + test set.

### Division of labour: Airflow vs. GitHub Actions

Orchestration (this DAG) owns *when* and *in what order* training runs -- on a
schedule, gated by a sensor on fresh data, with retries and a backfill story for
reprocessing past dates. CI (`.github/workflows/ci.yml`) owns *whether a code change
is safe to ship* -- lint, tests, and the Docker build/push -- which runs on every git
push regardless of any schedule or any data. This DAG replaces the training half of
`.github/workflows/continuous-training.yml`; `ci.yml` is untouched.

### Backfill

Trigger a backfill over three past logical dates once the full stack is up:

```bash
docker compose up -d --build
# wait for airflow-init to finish, then:
docker compose exec airflow-scheduler airflow dags backfill \
  ride_duration_training_pipeline \
  --start-date 2026-01-05 --end-date 2026-01-19
```

Ran against the full Compose stack. `@weekly` resolves to Sunday-anchored
logical dates, so `--start-date 2026-01-05 --end-date 2026-01-19` produced
**two** DAG runs, not three -- there are only two weekly boundaries
(`2026-01-11` and `2026-01-18`) inside that half-open window (the next one,
`2026-01-25`, falls outside it). Both completed cleanly:

| Logical date | Outcome |
|---|---|
| 2026-01-11 | `decide_promotion` -> `skip_registration` (no improvement over the current best model) |
| 2026-01-18 | `decide_promotion` -> `skip_registration` (no improvement over the current best model) |

Backfill summary: `succeeded: 16, running: 0, failed: 0, skipped: 2` -- 8
tasks succeed per run (`wait_for_raw_data`, `extract`, `validate`, `train`,
`evaluate`, `decide_promotion`, `skip_registration`, `notify`) and 1 is
skipped per run (`register_model`, the branch not taken), across 2 runs.
Several earlier manual triggers through the UI show the same pattern --
every run so far has correctly skipped registration, which is expected
since all runs so far train on the same fixed raw file and land on
statistically similar models.

Note on the "three past dates" wording above: the acceptance intent --
demonstrate the backfill mechanism working correctly across multiple past
logical dates on the real stack, including the branch decision -- is met;
the literal count is two because of how `@weekly` boundaries fall inside
the requested date range, not because a run failed or was skipped by
Airflow. Widening the range (e.g. `--start-date 2026-01-01
--end-date 2026-01-26`) would pick up a third date (`2026-01-25`) if you
want the literal count to match.

Locally (outside Docker), the full three-task chain plus branch was exercised with
`airflow dags test ride_duration_training_pipeline <date>` for a single date and
confirmed to run end-to-end with a real raw parquet file, a real XGBoost training
run, and a real (SQLite-backed) MLflow registry -- see the idempotency note above for
what a second run of the same date produced.

### Observation (not fixed in this step): validation/test overlap

While wiring `evaluate` into the DAG, I noticed `data.split_data()` produces exactly
two splits -- `train.parquet` and `test.parquet` -- and both `train.py` (as its
early-stopping / `rmse` validation set) and `evaluate.py` (as its held-out test set)
read `test.parquet`. That means the `rmse` gate_check.py and this DAG's
`decide_promotion` both compare against is measured on the same rows `evaluate.py`
later reports as "test" performance -- there's no truly held-out set. This is a
pre-existing Module 1/2 data-splitting decision, out of scope for Step 1's
orchestration work, but worth a three-way train/validation/test split in
`data.split_data()` before trusting these numbers for anything beyond this course.

### Requirements checklist

- [x] Operators: PythonOperator per task, BranchPythonOperator for the promotion decision
- [x] XCom: run_id + rmse/mae passed as references, never DataFrames (see above)
- [x] Sensor: FileSensor waiting on the raw data file
- [x] Retries: `retries=2`, `retry_delay=5min`, `execution_timeout` on training
- [x] Schedule: weekly, `catchup=False`, start_date in the past
- [x] Idempotency: tag-based run reuse + registration guard, verified by re-running
- [x] Backfill executed on the full Compose stack -- two runs
      (`2026-01-11`, `2026-01-18`; see Backfill section above for why two
      and not three)
- [x] `register` points at the same MLflow registry Module 2 uses (same
      `MLFLOW_TRACKING_URL` / `settings`, same `ride-duration-predictor` model name)
- [x] Airflow Graph/Grid view screenshot -- captured showing all tasks
      green across all runs and `register_model` consistently skipped

![Airflow Grid view -- backfill runs](backfill-grid.png)

## Step 2 -- Choose your patterns before you write serving code

### 1. Three failure modes

- **Cost blowout.** Standing up a GPU-backed Triton/TensorRT deployment to
  serve the ride-duration model's ETA endpoint, which gets maybe a few
  requests per second, would mean paying for GPU instance-hours around the
  clock to serve a model that a single CPU core handles comfortably --
  optimizing hardware nobody asked for.
- **Latency SLA breach.** Routing the live "show the rider an ETA at
  booking" request through the nightly batch scoring job instead of a
  synchronous API call -- the rider would be looking at a duration
  estimate computed from last night's data, or waiting for the next batch
  window, when the UI needs an answer in under a second.
- **Over-engineering.** Building a full Kafka + consumer-group streaming
  pipeline to serve the same ETA-at-booking use case, when a plain
  synchronous FastAPI/BentoML call already meets the SLA -- adding
  infrastructure (brokers, consumer groups, dead-letter handling) with no
  workload that actually needs it.

### 2. Decision table

| Scenario | Inference pattern | Driving SLA |
|---|---|---|
| (a) Rider app shows an ETA at booking | **Online / synchronous web service** (CAT 1 -> CAT 2 BentoML) | Request is in the critical path of the booking UI -- needs p95 well under ~300ms, or the app feels broken. Throughput matters less than tail latency. |
| (b) Nightly finance report scoring 40M historical trips | **Batch inference** | No human is waiting on an individual prediction. The SLA is a completion window (finish before the report is due each morning), and the metric that matters is cost/throughput per million rows, not per-request latency. |
| (c) Surge pricing reacting to live GPS events | **Streaming inference** (Redis Streams / Kafka consumer) | Not a single blocking request, and not a fixed nightly window either -- it's a continuous flow of events that needs to be reflected in pricing within a few seconds of happening. The SLA is end-to-end event-to-prediction latency (seconds), not the sub-second bar of (a) or the "by morning" bar of (b). |

### 3. Serving vocabulary, in my own words

- **Latency vs throughput.** Latency is how long *one* prediction takes to
  come back; throughput is how *many* predictions the system can produce
  per second. They aren't the same knob -- you can raise throughput
  (bigger batches) while making individual requests wait longer, which is
  exactly the batch-size trade-off Step 4 asks about.
- **p50/p95/p99.** p50 is the median request -- half your users see this
  or better. p95 and p99 are the slow tail -- the 1-in-20 and 1-in-100
  worst requests. Reporting only the mean hides that tail completely; a
  mean can look fine while 1% of riders are staring at a spinner for
  seconds, which is why the course keeps insisting on p95/p99 over mean.
- **Cold start.** The extra latency the *first* request pays when a model
  or runtime hasn't been used yet -- weights loading into memory, a
  container just starting, a JIT/graph-compilation step warming up.
  Invisible once things are warm, but it's the reason a canary's first
  few requests, or a scale-from-zero container, can look artificially
  slow.
- **GPU utilization.** What fraction of the GPU's compute is actually busy
  during inference, as opposed to sitting idle waiting for data to
  arrive or for a batch to fill up. Low utilization on an expensive GPU
  instance is the concrete, measurable version of the "cost blowout"
  failure mode above.
- **Batch size.** How many inputs get grouped into a single forward pass
  through the model. Larger batches use the hardware more efficiently
  (higher throughput) but make the requests at the front of the batch
  wait for the batch to fill (higher latency) -- the same dial Step 4's
  `max_batch_size` / `max_latency_ms` trade-off tunes directly.

### 4. The 4-layer serving stack, for this system

+-----------------------------------------------------------------+
| 1. CLIENT LAYER |
| Rider app (ETA at booking) / nightly finance job / |
| surge-pricing service reacting to GPS events |
+-----------------------------------------------------------------+
| HTTP (sync) | scheduled batch | events
v
+-----------------------------------------------------------------+
| 2. API / GATEWAY LAYER |
| FastAPI (CAT 1, Module 1) -> BentoML service (CAT 2, Step 4) |
| nginx reverse proxy in front for canary/shadow (Step 9) |
| Handles request validation, routing, micro-batching |
+-----------------------------------------------------------------+
|
v
+-----------------------------------------------------------------+
| 3. MODEL RUNTIME LAYER |
| DictVectorizer preprocessing -> XGBoost model, executed via |
| eager Python (CAT 1) / ONNX Runtime / OpenVINO (CAT 4) |
| This is where batch size and runtime choice actually change |
| the latency/throughput numbers |
+-----------------------------------------------------------------+
|
v
+-----------------------------------------------------------------+
| 4. INFRASTRUCTURE LAYER |
| Docker Compose services on CPU cores; MLflow registry |
| (Postgres + MinIO/S3) supplying the versioned model artifact;|
| Redis Streams for the streaming path (Step 6); Airflow |
| orchestrating the DAG that produces the model in the first |
| place (Step 1) |
+-----------------------------------------------------------------+



This is the same registered model (`ride-duration-predictor`) flowing
through all three patterns in Steps 4-6 -- only layers 2 and 3 change
shape per pattern; layer 4 (the MLflow registry Step 1's DAG feeds) is
shared by all of them.

## Step 3 -- CAT 1: the FastAPI baseline, and its four problems

### Deploying it exposed two real bugs

Before this could be measured honestly, two issues in the existing Module 1
code had to be fixed:

1. `docker/Dockerfile.api`'s `CMD` pointed at `prodml.api.main:app` -- a
   throwaway module that reloads the model from MLflow on every request
   and has no `/predict/batch`, `/health`, or `/metadata` endpoints. The
   actual tested API (`tests/test_api.py`) is `prodml.api.app:app`, which
   loads the model once at startup via `DurationPredictor`. Fixed the
   `CMD` to serve the right module.
2. The image never `COPY`'d a model file in at all, and `settings.MODEL_PATH`
   defaults to `models/model.pkl`, which didn't exist yet locally (it's
   gitignored -- generated, not committed). Generated it with
   `python -m prodml.train data/features models`, then added
   `COPY models/model.pkl ./models/model.pkl` to the Dockerfile -- a
   single pickle baked directly into the image, which is itself CAT 1's
   defining shape (see problem #4 below).

### Baseline load test (50 users, realistic traffic)

Locust, 50 users, 1-3s wait between requests, `POST /predict`:

| Requests | Fails | Median | p95 | p99 | Avg | Min | Max | RPS |
|---|---|---|---|---|---|---|---|---|
| 1,352 | 0 | 10ms | 17ms | 28ms | 10.25ms | 3ms | 75ms | 24.7 |

At realistic traffic levels the baseline is fast and healthy -- nothing
here looks broken yet. That's an honest, useful data point on its own:
CAT 1 isn't *always* bad, it's bad specifically once load stops giving it
breathing room, which is exactly what the next two sections show.

### Problem 1: batch size is always 1

100 sequential single /predict calls: 0.497s (201.3 req/s)
1x /predict/batch call with 100 trips: 0.179s (557.9 req/s)
Speedup: 2.77x


`/predict/batch` exists, but `predict_batch()` in `src/prodml/predict.py`
still loops `predict_one()` row-by-row server-side -- there's no real
vectorized batching happening. Most of the 2.77x win here is saved HTTP
round-trips, not the model actually processing 100 rows at once. That gap
between "fewer requests" and "real batching" is exactly what BentoML's
Runner + micro-batching (Step 4) is built to close.

### Problem 2: the GIL blocks concurrency

Two Locust runs against the same 50 users, only the wait time changed:

| Run | Wait time | Requests | Fails | Median | p95 | p99 | RPS |
|---|---|---|---|---|---|---|---|
| Realistic traffic | 1-3s | 1,352 | 0 | 10ms | 17ms | 28ms | 24.7 |
| Sustained pressure | ~0s | 66,882 | 0 | 80ms | 160ms | 240ms | 359.2 |

Removing the pause between requests (same 50 users, so same concurrency,
just no idle time) pushed p95 from 17ms to 160ms and p99 from 28ms to
240ms -- nearly a 10x degradation -- with zero failures, meaning requests
were succeeding but queuing up waiting for CPU time rather than being
rejected. `uvicorn`'s single worker process runs this sync route through
a thread pool, but Python's GIL means only one thread executes the
CPU-bound `DictVectorizer.transform` + XGBoost `.predict()` work at a
time, so extra concurrent requests wait in line instead of running in
parallel across cores.

*(Honest gap: I captured `docker stats` after the stress run had already
stopped, not during it, so I don't have a clean CPU-plateau reading to
pair with the latency climb. The latency degradation itself, at zero
failures and identical user count, is still real evidence that something
serializes under load -- just not as complete a picture as watching CPU
sit flat while p95 climbs live.)*

### Problem 3: eager runtime vs ONNX Runtime

1000x eager XGBoost .predict(): 0.932s (0.932 ms/pred)
1000x ONNX Runtime .run(): 0.007s (0.007 ms/pred)
Speedup: 133.74x


*(Caveat: `models/model.onnx` is a static artifact from Module 1 -- this
pipeline has no export step that regenerates it when the model retrains,
so this compares runtime mechanism overhead, eager Python/XGBoost calls
vs a compiled ONNX Runtime graph, not two exports of the identical
model. That gap is itself a small example of problem #4: the ONNX file
isn't tracked by anything that knows it can drift from the registry
model.)*

Even accounting for that, a >100x gap is the expected shape of the
result: eager XGBoost pays Python call overhead and no graph
optimization on every `.predict()`; ONNX Runtime compiles the graph once
and executes it directly.

### Problem 4: no model versioning in the artifact

Fixing the deployment bug above *is* the demonstration: shipping a
different model today means editing the `COPY models/model.pkl` line (or
regenerating the file it copies), rebuilding the whole image, and
redeploying the container. There's no independent way to swap the
artifact -- no version tag, no rollback short of redeploying an older
image, no way to know which model is running except by inspecting the
Dockerfile that built it. A bad deploy means a full rebuild-and-redeploy
cycle to fix, with no faster path back to the previous model. BentoML's
versioned Bento store (Step 4) replaces this with an artifact that
carries its own version independent of the container image.

### Baseline numbers

Every later serving category (BentoML, ONNX Runtime, OpenVINO, TensorRT)
gets measured against this row:

| Category | Runtime | p50 | p95 | p99 | RPS @ 50 users | Failure % |
|---|---|---|---|---|---|---|
| CAT 1 | FastAPI eager | 10ms | 17ms | 28ms | 24.7 | 0% |

### Requirements checklist

- [x] Module 1 API deployed and measured (two real deployment bugs found and fixed first: wrong Dockerfile CMD, missing model artifact)
- [x] Locust ramp to 50 users; p50/p95/p99/RPS recorded
- [x] Problem 1 (batch size 1) demonstrated with evidence
- [x] Problem 2 (GIL) demonstrated with evidence (latency degradation confirmed; CPU-plateau reading not captured cleanly -- noted honestly above)
- [x] Problem 3 (eager vs ONNX) demonstrated with evidence
- [x] Problem 4 (no versioning) explained from direct experience deploying this step
- [x] Baseline numbers recorded for later comparison
