import os
import unittest
import pathlib

import numpy as np
import torch

import modlee
import mlflow
from mlflow.client import MlflowClient
client = MlflowClient()


class RepTest(unittest.TestCase):

    run_dirs = [
        # '/Users/modlee/projects/modlee_pypi/modlee/tests/mlruns',
        # '/Users/modlee/projects/modlee_pypi/notebooks/mlruns',
        # '/Users/modlee/projects/modlee_pypi/notebooks/mlruns/0/c071dfc3a9d24a0cb148241af9af4e84',
        '/Users/modlee/projects/modlee_pypi/notebooks/mlruns/0/b65da0553bac46c1aa9b8b0d51e941d2',
    ]

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_get_runs(self):
        '''
        Retrieve runs from prior mlruns directories
        '''
        mlruns_dirs = [
            '/Users/modlee/projects/modlee_pypi/modlee/tests/mlruns',
            '/Users/modlee/projects/modlee_pypi/notebooks/mlruns',
        ]
        for run_dir in mlruns_dirs:
            runs = modlee.get_runs(run_dir)
            assert len(runs) > 0, \
                f"No runs found in {run_dir}"
            run = runs[0]

    def test_cant_get_runs(self):
        '''
        Should not be able to retrieve runs from garbage directories
        '''
        run_dirs = [
            'fasdfasf'
        ]
        for run_dir in run_dirs:
            runs = modlee.get_runs(run_dir)
            assert len(runs) == 0, \
                f"Should not have found runs in {run_dir}, but found {len(runs)}"

    def test_get_model(self):
        '''
        Retrieve models from prior runs
        '''
        for run_dir in self.run_dirs:
            modlee_model = modlee.get_model(run_dir)
            mlflow_model = mlflow.pytorch.load_model(
                f"{run_dir}/artifacts/model"
            )
            param_thresh = 0.001

            data_snapshot = modlee.get_data_snapshot(run_dir)

            # set the modlee-loaded weights equal to the mlflow-loaded weights
            modlee_model.load_state_dict(mlflow_model.state_dict())

            # test that inference outputs are within a difference threshold
            modlee_model.eval()
            mlflow_model.eval()
            with torch.no_grad():
                for x in data_snapshot:
                    y_modlee = modlee_model(torch.Tensor(x).unsqueeze(0))
                    y_mlflow = mlflow_model(torch.Tensor(x).unsqueeze(0))
                    diff_y = np.abs(y_modlee.numpy()-y_mlflow.numpy())
                    assert np.max(diff_y) < param_thresh, \
                        f"Difference between modlee- and mlflow-loaded model outputs is greater than threshold. Prediction difference: {diff_y}"

    def test_get_data_snapshot(self):
        for run_dir in self.run_dirs:
            data_snapshot = modlee.get_data_snapshot(run_dir)
            assert data_snapshot is not None, \
                f"Could not retrieve data_snapshot.npy from {run_dir}"
        pass
