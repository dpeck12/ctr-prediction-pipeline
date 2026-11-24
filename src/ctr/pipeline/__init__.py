"""Daily retraining and monitoring: the logic behind the Airflow DAG.

Everything here is plain Python so it can be tested and replayed without
Airflow. The DAG in airflow/dags/ only decides when each step runs.
"""
