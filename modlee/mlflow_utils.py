
import mlflow
from mlflow.client import MlflowClient

from modlee.rep import Rep

client = MlflowClient()

def get_run(run_idx=0):
    experiments = client.search_experiments()
    runs = client.search_runs(experiments[0].experiment_id)
    return runs[run_idx]

def get_model(run):
    rep = Rep.from_run(run)
    return rep.model