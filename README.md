# Course Recommender System & Production Deployment Pipeline


An interactive, end-to-end Machine Learning web application designed to evaluate and serve personalised course recommendations. This project demonstrates an integration of statistical modelling, natural language feature engineering, and robust, production-ready **MLOps containerisation**.

---

## Architecture & Technical Highlights

- **Machine Learning Core**: Evaluated 14 distinct recommendation strategies utilising a Bag-of-Words feature matrix and Cosine Similarity to capture semantic item alignment.
- **Exploratory Data Analysis**: Isolated distinct user segments by identifying and breaking down a bimodal student enrolment distribution.
- **Enterprise Containerisation**: Structured a secure, high-performance, multi-stage Docker workflow to separate heavy compilation dependencies from the minimal runtime image.
- **Orchestration Parity**: Configured a local `docker-compose` framework utilising advanced orchestration design patterns (PID 1 initialisers, automated liveness loops) that translate directly to Kubernetes infrastructure objects.

---

## Infrastructure & MLOps Blueprint

The deployment engine is split into a multi-stage architecture engineered for high speed, minimal layer overhead, and secure local isolation.

### 1. Multi-Stage Docker Optimisation (`Dockerfile`)
- **Stage 1 (Builder)**: Leverages `python:3.9-slim` to compile heavy machine learning wheels with automated pip cache mounts (`--mount=type=cache`), optimising layer rebuilding times.
- **Stage 2 (Runtime)**: Port-maps an isolated runtime footprint. Injects OpenMP (`libgomp1`) directly to support optimised tree-ensemble model processing.
- **Least-Privilege Security**: Enforces non-root user execution (`USER appuser`, UID `10001`) to eliminate container breakout risks in cluster environments.
- **Automated Telemetry**: Employs a built-in `HEALTHCHECK` query loop tracking the internal `_stcore/health` Streamlit endpoint to guarantee seamless self-healing capabilities.

### 2. Local Composition Framework (`docker-compose.yml`)
The docker-compose setup provides complete functional parity with production cluster deployment patterns:
- **Process Initialiser (`init: true`)**: Integrates `tini` as PID 1 to capture signal interrupts and clean up zombie child processes, preventing systemic memory leakages.
- **Persistent Volume Mapping**: Mounts localised analytics artifacts (`ratings.csv`) safely to prevent data evaporation during container recycles.

---

## Local Deployment Instructions

To spin up the containerised application environment locally, ensure you have Docker installed and execute:

```bash
# Clone the repository
git clone https://github.com
cd Streamlit2

# Build the runtime image and launch the local orchestration layer
docker compose up --build -d
```

Once initialised, the containerised engine will automatically perform self-tests and serve the interactive platform at `http://localhost:8501`.

---

## Core Technical Stack
- **Languages & Modelling**: Python, NumPy, Pandas, Scikit-Learn, Cosine Similarity
- **Infrastructure & Delivery**: Docker, Docker Compose, Linux CLI, Streamlit Cloud
