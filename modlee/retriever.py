from importlib.machinery import SourceFileLoader
import os

import numpy as np
import modlee
from modlee import logging

from mlflow.client import MlflowClient
client = MlflowClient()


def run_dir_exists(run_dir):
    if not os.path.exists(run_dir):
        logging.warning(f"Run directory {run_dir} does not exist")
        return False
    return True


def get_runs(run_dir, experiment_id=None, run_id=None, **kwargs):
    if not run_dir_exists(run_dir):
        return []

    modlee.set_run_dir(run_dir)

    experiments = client.search_experiments()
    if len(experiments) == 0:
        logging.warning(f"No experiments found in {run_dir}")
        return []
    runs = []
    if experiment_id is not None:
        experiments = [experiments[experiment_id]]
    filter_string = ''
    if run_id is not None:
        filter_string = f"run_id='{run_id}'"
    for experiment in experiments:
        _exp_runs = client.search_runs(
            experiment.experiment_id, filter_string, **kwargs)
        runs = [*runs, *_exp_runs]

    return runs


def get_model(run_dir):
    if not run_dir_exists(run_dir):
        return None
    model_path = f"{run_dir}/artifacts/model.py"
    model = SourceFileLoader(
        'modlee_mod',
        model_path
    ).load_module().ModleeModel()
    return model


def get_data_snapshot(run_dir):
    if not run_dir_exists(run_dir):
        return None
    data_snapshot_path = f"{run_dir}/artifacts/data_snapshot.npy"
    if not os.path.exists(data_snapshot_path):
        return None
    return np.load(data_snapshot_path)
