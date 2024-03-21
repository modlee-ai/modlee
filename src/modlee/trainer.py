""" 
Modlee Trainer.
"""
import os
import mlflow
from modlee.utils import last_run_path
from lightning import pytorch as pl

class Trainer(pl.Trainer):
    """ 
    Trainer that directs checkpoint files to the current run directory.
    """
    def __init__(self, *args, **kwargs):
        artifacts_dir = os.path.join(last_run_path(), 'artifacts')
        kwargs['default_root_dir'] = kwargs.get('default_root_dir', artifacts_dir)
        super().__init__(*args, **kwargs)