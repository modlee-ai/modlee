""" 
Retriever for experiment assets.
"""
from importlib.machinery import SourceFileLoader
import inspect
import json
import os
import ast
import numpy as np
import modlee
from modlee import logging
import mlflow
from mlflow.client import MlflowClient


def run_path_exists(run_path):
    """
    Chek if a run path exists.

    :param run_path: The run path to check.
    :return: Whether the path exists or not.
    """
    if not os.path.exists(run_path):
        logging.warning(f"Run directory {run_path} does not exist")
        return False
    return True


def get_runs(run_path, experiment_id=None, run_id=None, **kwargs):
    """
    Get the runs in a given run path.

    :param run_path: The path to search.
    :param experiment_id: The experiment ID to retrieve, defaults to None and retrieves all experiments.
    :param run_id: The run ID to retrieve, defaults to None to retrieves all runs.
    :return: A list of runs.
    """
    if not run_path_exists(run_path):
        return []

    modlee.set_run_path(run_path)

    client = MlflowClient()
    experiments = client.search_experiments()

    if len(experiments) == 0:
        logging.warning(f"No experiments found in {run_path}")
        return []
    runs = []
    if experiment_id is not None:
        experiments = [experiments[experiment_id]]
    filter_string = ""
    if run_id is not None:
        filter_string = f"run_id='{run_id}'"
    for experiment in experiments:
        _exp_runs = client.search_runs(
            experiment.experiment_id, filter_string, **kwargs
        )
        runs = [*runs, *_exp_runs]
    return runs


def get_model(run_path):
    """
    Get the model at a run path.

    :param run_path: The run path.
    :return: The model as a ModleeModel object.
    """
    if not run_path_exists(run_path):
        return None
    model = SourceFileLoader(
        "modlee_mod", f"{run_path}/artifacts/model.py"
    ).load_module()

    # retrieve the variables for the object signature
    model_kwargs = dict(inspect.signature(model.ModleeModel).parameters)
    model_kwargs.pop("args"), model_kwargs.pop("kwargs")
    cached_vars = get_cached_vars(run_path)
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
    return model.ModleeModel(**model_kwargs)


def get_cached_vars(run_path):
    """
    Get the cached variables required to rebuild a model from a run path.

    :param run_path: The run path.
    :return: A dictionary of the cached variables.
    """
    if not run_path_exists(run_path):
        return {}
    with open(f"{run_path}/artifacts/cached_vars", "r") as vars_file:
        return json.loads(vars_file.read())

