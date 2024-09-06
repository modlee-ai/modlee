""" 
Test modlee.
"""

import os
import unittest
import pathlib
import modlee
import mlflow
import lightning.pytorch as pl


class ModleeTest(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_set_run_path(self):
        """
        Tracking URI should change
        """
        paths_to_test = [os.path.abspath("."), ".", ".."]
        for path_to_test in paths_to_test:
            modlee.set_run_path(path_to_test)
            run_path = os.path.abspath(f"{path_to_test}/mlruns")

            tracking_uri = mlflow.get_tracking_uri()
            assert (
                tracking_uri == pathlib.Path(run_path).as_uri()
            ), f"Tracking URI {tracking_uri} does not match path {path_to_test}/mlruns"

            got_run_path = modlee.get_run_path()

    def test_cant_set_run_path(self):
        """
        Should not be able to set to track garbage directory
        """
        paths_to_test = ["asdfadsfsd"]
        for path_to_test in paths_to_test:
            with self.assertRaises(FileNotFoundError):
                modlee.set_run_path(path_to_test)

    def test_init(self):
        init_args = [None, ".."]
        for init_arg in init_args:
            modlee.init(init_arg)
            run_path = modlee.get_run_path()
            if init_arg == None:
                init_arg = "."
            relative_mlruns_path = os.path.abspath(
                f"{os.path.join(os.getcwd(), init_arg)}/mlruns"
            )
            assert (
                run_path == relative_mlruns_path
            ), f"Initialized directory {run_path} does not match expected {relative_mlruns_path}"


    def test_training(self):
        with modlee.start_run() as run:
            trainer = pl.Trainer(max_epochs=1)
            breakpoint()
            trainer.fit(
                model=modlee_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )
            
