from importlib.machinery import SourceFileLoader
import inspect
import json
import os
import ast

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
    model = SourceFileLoader(
        'modlee_mod',
        f"{run_dir}/artifacts/model.py"
    ).load_module()

    # retrieve the variables for the object signature
    model_kwargs = dict(inspect.signature(model.ModleeModel).parameters)
    model_kwargs.pop('args'), model_kwargs.pop('kwargs')
    cached_vars = get_cached_vars(run_dir)
    keys_to_pop = []
    for model_key, model_val in model_kwargs.items():
        cached_val = cached_vars.get(model_key, None)
        if cached_val:
            model_kwargs.update({model_key: cached_val})
        elif model_val.default != inspect._empty:
            model_kwargs.update({model_key: model_val.default})
        else:
            keys_to_pop.append(model_key)
    for key_to_pop in keys_to_pop:
        model_kwargs.pop(key_to_pop)

    # recreate the model
    return model.ModleeModel(
        **model_kwargs
    )


def get_cached_vars(run_dir):
    if not run_dir_exists(run_dir):
        return {}
    with open(f"{run_dir}/artifacts/cached_vars", 'r') as vars_file:
        return json.loads(vars_file.read())
    

def get_data_snapshot(run_dir):
    if not run_dir_exists(run_dir):
        return None
    data_snapshot_path = f"{run_dir}/artifacts/data_snapshot.npy"
    if not os.path.exists(data_snapshot_path):
        return None
    return np.load(data_snapshot_path)
