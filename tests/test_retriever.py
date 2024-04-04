""" 
Test retriever.
"""
import os
import unittest
import pathlib


import numpy as np
import torch
import yaml

import modlee
import mlflow
from mlflow.client import MlflowClient

with open(
    os.path.join(os.path.dirname(__file__), "test_retriever.yaml"), "r"
) as test_retriever_file:
    ret_dict = yaml.safe_load(test_retriever_file)
globals().update(
    dict(mlruns_paths=ret_dict["mlruns_paths"], run_paths=ret_dict["run_paths"])
)

run_paths = [os.path.join(os.path.dirname(__file__), "test_mlruns")]


class _RepTest(unittest.TestCase):
    """
    Deprecated, this functionality is still required on the client side but needs
    updates according to the new documentation schemes

    :param unittest: _description_
    :return: _description_
    """

    locals().update(ret_dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__.update(ret_dict)

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def _test_get_runs(self):
        """
        Retrieve runs from prior mlruns directories
        """
        # for run_path in mlruns_paths:
        for run_path in run_paths:
            runs = modlee.get_runs(run_path)
            assert len(runs) > 0, f"No runs found in {run_path}"

    def test_cant_get_runs(self):
        """
        Should not be able to retrieve runs from garbage directories
        """
        run_paths = ["fasdfasf"]
        for run_path in run_paths:
            runs = modlee.get_runs(run_path)
            assert (
                len(runs) == 0
            ), f"Should not have found runs in {run_path}, but found {len(runs)}"

    def _test_get_model(self):
        """
        Retrieve models from prior runs
        """
        for run_path in run_paths:
            model = modlee.get_model(run_path)
            mlflow_model = mlflow.pytorch.load_model(f"{run_path}/artifacts/model")

            data_snapshot = modlee.get_data_snapshot(run_path)

            # set the modlee-loaded weights equal to the mlflow-loaded weights
            model.load_state_dict(mlflow_model.state_dict())

            # test that inference outputs are within a difference threshold
            model.eval(), mlflow_model.eval()
            param_thresh = 0.001
            with torch.no_grad():
                for x in data_snapshot:
                    y_modlee = model(torch.Tensor(x).unsqueeze(0))
                    y_mlflow = mlflow_model(torch.Tensor(x).unsqueeze(0))
                    diff_y = np.abs(y_modlee.numpy() - y_mlflow.numpy())
                    assert (
                        np.max(diff_y) < param_thresh
                    ), f"Difference between modlee- and mlflow-loaded model outputs is greater than threshold. Prediction difference: {diff_y}"

    def _test_get_data_snapshot(self):
        for run_path in run_paths:
            data_snapshot = modlee.get_data_snapshot(run_path)
            assert (
                data_snapshot is not None
            ), f"Could not retrieve data_snapshot.npy from {run_path}"
        pass
