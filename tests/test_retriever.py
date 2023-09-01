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

    mlruns_dirs = [
        '/Users/modlee/projects/modlee_pypi/modlee/tests/mlruns',
        '/Users/modlee/projects/modlee_pypi/notebooks/mlruns',
    ]
    run_dirs = [
        # '/Users/modlee/projects/modlee_pypi/notebooks/mlruns/0/b65da0553bac46c1aa9b8b0d51e941d2',
        # '/Users/modlee/projects/modlee_pypi/mlruns/0/603a2d573d3d4e95bd6a2f070a864b8d',
        # '/Users/modlee/projects/modlee_pypi/examples/mlruns/0/f2757107910e454489949d0c8e1da599',
        # '/Users/modlee/projects/modlee_pypi/examples/mlruns/0/12d92596d3a942c4868f9f6c18116692',
        # '/Users/modlee/projects/modlee_pypi/examples/mlruns/0/455cbd83c4fc44dab396a15be43fe9d4',
        '/Users/modlee/projects/modlee_pypi/examples/mlruns/0/635782e7b3114dbea4f66d7c81befb20',
        
        # this one will fail because the cached_vars was not logged
        # '/Users/modlee/projects/modlee_pypi/examples/mlruns/0/1942a15973c943a485e04738de628628',
        # '/Users/modlee/projects/modlee_pypi/examples/mlruns/0/8e5844784c8448b585b6ccc047eac271',
        
        # '/Users/modlee/projects/scratch/lightning_tutorials/lightning_examples/barlow-twins/mlruns/0/9792d5b6d83f44a2a2c45a4162353280',
        # '/Users/modlee/projects/scratch/lightning_tutorials/lightning_examples/basic-gan/mlruns/0/4b6593f253e4419594f1daf2903952ef',
        # '/Users/modlee/projects/scratch/lightning_tutorials/lightning_examples/mnist-hello-world/mlruns/0/ba6f693c464d42a8b083bcb53236c0ac',
        # '/Users/modlee/projects/scratch/lightning_tutorials/lightning_examples/cifar10-baseline/mlruns/0/8b711c778e4f48acb91bb2bb9d25ed17',
        # '/Users/modlee/projects/scratch/lightning_tutorials/lightning_examples/text-transformers/mlruns/0/1c1dc202b24c493fad5a60052dbac253',
        # '/Users/modlee/projects/scratch/lightning_tutorials/lightning_examples/augmentation_kornia/mlruns/0/fde2320f3c6b4fb395156034b58931ef',
        # '/Users/modlee/projects/scratch/lightning_tutorials/lightning_examples/datamodules/mlruns/0/013153e156ea41eda40ea39b0a19f7b2',
    ]
    fail_run_dirs = [
        '/Users/modlee/projects/modlee_pypi/examples/mlruns/0/1942a15973c943a485e04738de628628',
    ]

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_get_runs(self):
        '''
        Retrieve runs from prior mlruns directories
        '''
        for run_dir in self.mlruns_dirs:
            runs = modlee.get_runs(run_dir)
            assert len(runs) > 0, \
                f"No runs found in {run_dir}"

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

            data_snapshot = modlee.get_data_snapshot(run_dir)

            # set the modlee-loaded weights equal to the mlflow-loaded weights
            modlee_model.load_state_dict(mlflow_model.state_dict())

            # test that inference outputs are within a difference threshold
            modlee_model.eval(), mlflow_model.eval()
            param_thresh = 0.001
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
