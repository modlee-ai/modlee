from importlib.machinery import SourceFileLoader
import json
import os
import pickle
import numpy as np

import mlflow
from mlflow.client import MlflowClient
from mlflow.entities import Run


class Rep(Run,):
    artifacts = ['modlee_model.py','data_stats']
    
    def __init__(self,run=None,**kwargs):
        self.__dict__.update(**kwargs)
        self.client = MlflowClient()
        
        if 'run_kwargs' in kwargs:
            Run.__init__(self, **kwargs['run_kwargs'])
            mlflow.set_tracking_uri(
                os.path.split(
                    self.info.artifact_uri
                )[0]
            )
        
    @classmethod
    def from_run(cls,run,**kwargs):
        run_keys = ['info','data','inputs']
        return cls(run_kwargs={f"run_{k}":getattr(run,k) for k in run_keys})

    def _init_model(self):
        model_path = self._get_artifact('model.py')
        self._modlee_model_module = SourceFileLoader(
            'modlee_mod',model_path,).load_module()
        self._model = self._modlee_model_module.ModleeModel()
        
    def _init_data_stats(self):
        data_stats_path = self._get_artifact('data_stats')
        with open(data_stats_path,'r') as data_stats_file:
            self._data_stats = json.load(data_stats_file)
            
    def _init_data_loader(self):
        data_loader_path = self._get_artifact('data_loader')
        with open(data_loader_path,'r') as data_loader_file:
            self.data_stats = pickle.loads(data_loader_file)
            
    def _init_snapshot(self,snapshot_type='data'):
        snapshot_path = self._get_artifact(f'{snapshot_type}_snapshot.npy')
        # with open(data_snapshot_path,'r') as data_snapshot_file:
        setattr(self,f"_{snapshot_type}_snapshot",
            np.load(snapshot_path))
        self._data_snapshot = np.load(snapshot_path)
        
    @property
    def model(self):
        if not hasattr(self, '_model'):
            self._init_model()
        return self._model      
    
    @property
    def data_stats(self):
        if not hasattr(self, '_data_stats'):
            self._init_data_stats()
        return self._data_stats     
    
    @property
    def data_loader(self):
        if not hasattr(self, '_data_stats'):
            self._init_data_loader()
        return self._data_stats     
    
    @property
    def data_snapshot(self):
        if not hasattr(self, '_data_snapshot'):
            self._init_snapshot('data')
        return self._data_snapshot    
    @property
    def targets_snapshot(self):
        if not hasattr(self, '_targets_snapshot'):
            self._init_snapshot('targets')
        return self._targets_snapshot     
    
    
    def _get_artifact(self, artifact):
        if not getattr(self, 'info', True): 
            raise Exception('Rep not initialized with a run object')
        return self.client.download_artifacts(self.info.run_id,
            artifact)