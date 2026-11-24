# Airflow with this project's package installed, for the ctr_daily DAG.
# Built and run by docker-compose.yml at the repo root.

FROM apache/airflow:3.4.0-python3.11

ARG AIRFLOW_CONSTRAINTS=https://raw.githubusercontent.com/apache/airflow/constraints-3.4.0/constraints-3.11.txt

# The pipeline's runtime needs, pinned by Airflow's constraints where it pins them.
# PyTorch is left out: the DAG trains LightGBM by default, and ctr only imports
# torch when a DCN model is used.
RUN pip install --no-cache-dir --constraint "${AIRFLOW_CONSTRAINTS}" \
    lightgbm pandas pyarrow scikit-learn scipy matplotlib joblib

COPY --chown=airflow pyproject.toml README.md /opt/ctr-src/
COPY --chown=airflow src /opt/ctr-src/src
RUN pip install --no-cache-dir --no-deps /opt/ctr-src
